"""Lab-only graph timing, CUDA kernel trace, resource and optional Nsight profile."""
import json,time,statistics,shutil,subprocess,sys
from pathlib import Path
from benchmark import load
from test_public import read_flush

def run(candidate,root='/work'):
 import torch
 sys.path.insert(0,root);import reference
 mod=load(Path(root)/'candidate.py','profile_parent');weights=reference.generate_weights(82731)
 t=time.monotonic();st=mod.prepare(weights);print('PREPARED',time.monotonic()-t,flush=True)
 ncu=shutil.which('ncu') or next((str(p) for base in ('/opt/nvidia','/usr/local/cuda') for p in Path(base).glob('**/ncu') if p.is_file()),None)
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'ncu':ncu,'profiles':[],'resources':[]}
 flush=torch.ones(128*1024*1024,device='cuda',dtype=torch.uint8);sink=torch.empty(flush.numel()//4096,device='cuda',dtype=torch.int32)
 def clear():read_flush[(sink.numel(),)](flush,sink,flush.numel(),4096)
 clear();torch.cuda.synchronize()
 for layer,m in [('pooler',1),('pooler',8),('pooler',16),('pooler',32),('qkv_gate',1),('qkv_gate',16),('qkv_gate',32),('unpooler',16),('unpooler',32),('o',8)]:
  data=reference.generate_input(m,layer,500+m,'outliers')
  for _ in range(3):mod.kernel_fn(data,st)
  torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):clear();a.record();out=mod.kernel_fn(data,st);b.record()
  samples=[]
  for _ in range(40):g.replay();b.synchronize();samples.append(a.elapsed_time(b)*1000)
  ok,msg=reference.check_implementation(data,out,weights)
  if not ok:raise RuntimeError(msg)
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   for _ in range(3):g.replay()
   torch.cuda.synchronize()
  ev=[{'name':e.name,'us':e.device_time_total} for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA and 'read_flush' not in e.name]
  row={'layer':layer,'m':m,'graph_us':statistics.median(samples),'kernels':ev};result['profiles'].append(row);print('PROFILE',json.dumps(row),flush=True)
 for (layer,m),plan in st['v18']['plans'].items():
  x=reference.generate_input(m,layer,123)['x'];out=torch.empty((m,plan['rows']),device='cuda',dtype=torch.bfloat16)
  r=mod.v18s_launch_raw(st['v18']['ext'],plan['inst_id'],plan,x,out,plan['rows'],plan['cfg'],2)
  occ=mod.v18s_launch_raw(st['v18']['ext'],plan['inst_id'],plan,x,out,plan['rows'],plan['cfg'],1)
  result['resources'].append({'layer':layer,'m':m,'regs':r//1000,'local_bytes':r%1000,'occupancy_query':occ,'max_esc':plan['max_esc'],'packed_bytes':plan['wp'].numel()*plan['wp'].element_size(),'cfg':plan['cfg']})
 if ncu:
  code="import sys;sys.path.insert(0,'/work');import torch,reference;from benchmark import load;from pathlib import Path;m=load(Path('/work/candidate.py'),'ncu_parent');s=m.prepare(reference.generate_weights(82731));d=reference.generate_input(1,'pooler',9137);torch.cuda.synchronize();torch.cuda.profiler.start();m.kernel_fn(d,s);torch.cuda.synchronize();torch.cuda.profiler.stop()"
  try:
   r=subprocess.run([ncu,'--profile-from-start','off','--section','SpeedOfLight','--section','WarpStateStats','--csv',sys.executable,'-c',code],capture_output=True,text=True,timeout=220)
   result['ncu_returncode']=r.returncode;result['ncu_output']=(r.stdout+r.stderr)[-40000:]
  except subprocess.TimeoutExpired:result['ncu_output']='Timed out; no hardware-counter inference made.'
 else:result['ncu_output']='Nsight Compute not installed in the challenge image; graph traces and resource queries available, not hardware stall counters.'
 print('RESOURCES',json.dumps(result['resources']),flush=True)
 return result
