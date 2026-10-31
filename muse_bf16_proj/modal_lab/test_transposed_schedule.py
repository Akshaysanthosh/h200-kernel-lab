"""Bounded dense TMA scheduling experiment; protected parent comparator."""
import hashlib,json,statistics,sys,time,subprocess
from pathlib import Path
import numpy as np
from benchmark import load
from test_public import read_flush

def run(candidate,root='/work'):
 import torch
 from torch.utils.cpp_extension import load_inline
 sys.path.insert(0,root);import reference
 src=(Path(root)/'transposed_schedule.cu').read_text();start=time.time()
 ext=load_inline('okc_transposed_sched_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp_sources='torch::Tensor run(torch::Tensor,torch::Tensor,int,int,int);',cuda_sources=src,functions=['run'],extra_include_paths=['/opt/cutlass/include'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'],with_cuda=True,verbose=False)
 print('TRANSPOSE_COMPILED',time.time()-start,flush=True)
 mod=load(Path(root)/'candidate.py','schedule_parent')
 flush=torch.ones(128<<20,device='cuda',dtype=torch.uint8);sink=torch.empty(flush.numel()//4096,device='cuda',dtype=torch.int32)
 def clear():read_flush[(sink.numel(),)](flush,sink,flush.numel(),4096)
 clear();torch.cuda.synchronize()
 def graph(fn):
  for _ in range(3):fn()
  torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):clear();a.record();out=fn();b.record()
  return g,a,b,out
 def us(g):g[0].replay();g[2].synchronize();return g[1].elapsed_time(g[2])*1000
 def paired(a,b,reps):
  aa=[];bb=[]
  for i in range(reps):
   if i%2:bb.append(us(b));aa.append(us(a))
   else:aa.append(us(a));bb.append(us(b))
  rs=np.array([(aa[i]+aa[i+1])/(bb[i]+bb[i+1]) for i in range(0,reps,2)])
  boot=np.median(np.random.default_rng(17).choice(rs,(1000,len(rs))),axis=1)
  return {'ratio':float(np.median(rs)),'ci95':np.quantile(boot,[.025,.975]).tolist(),'parent_us':statistics.median(aa),'candidate_us':statistics.median(bb),'samples_us':[aa,bb]}
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda_source_sha256':hashlib.sha256(src.encode()).hexdigest(),'compile_seconds':time.time()-start,'shapes':[],'recheck':[],'nearby':[]}
 weights=reference.generate_weights(82731);state=mod.prepare(weights);winners=[]
 def pack(weights):
  packed={w.data_ptr():w.t().contiguous() for name,w in weights.items() if name in ('pooler','unpooler')}
  for name in ('pooler','unpooler'):
   w=weights[name];assert torch.equal(packed[w.data_ptr()].t(),w)
  return packed
 packed=pack(weights)
 def invoke(x,w,cfg,swizzle,raster):
  return ext.run(x,packed[w.data_ptr()] if cfg<8 else w,cfg,swizzle,raster)
 for layer in ('pooler','unpooler'):
  for m in (128,256):
   data=reference.generate_input(m,layer,9137+m,'outliers');x=data['x'];w=weights[layer]
   baseline=graph(lambda:mod.kernel_fn(data,state));aa=paired(baseline,graph(lambda:mod.kernel_fn(data,state)),40);rows=[]
   for cfg in range(10):
    for swizzle,raster in ((1,0),(1,1),(4,0),(4,1)):
     fn=lambda:invoke(x,w,cfg,swizzle,raster)
     out=fn();ok,msg=reference.check_implementation(data,out,weights)
     if not ok:raise RuntimeError(str((layer,m,cfg,swizzle,raster,msg)))
     g=graph(fn);row={'cfg':cfg,'swizzle':swizzle,'raster':raster,**paired(baseline,g,12)};rows.append(row)
   finalists=[]
   for row in sorted(rows,key=lambda z:-z['ratio'])[:2]:
    cfg,swizzle,raster=[row[k] for k in ('cfg','swizzle','raster')]
    g=graph(lambda:invoke(x,w,cfg,swizzle,raster))
    x.copy_(reference.generate_input(m,layer,7391+m,'normal')['x']);g[0].replay()
    ok,msg=reference.check_implementation(data,g[3],weights)
    if not ok:raise RuntimeError('fresh '+msg)
    finalists.append({**row,**paired(baseline,g,80)})
   winner=max(finalists,key=lambda r:r['ratio']);winners.append((layer,m,winner))
   result['shapes'].append({'layer':layer,'m':m,'aa':aa,'screen':rows,'finalists':finalists})
   print('TRANSPOSE_FINAL',layer,m,json.dumps([{k:v for k,v in z.items() if k!='samples_us'} for z in finalists]),flush=True)
 weights=reference.generate_weights(294817);state=mod.prepare(weights);packed=pack(weights)
 for layer,m,row in winners:
  cfg,swizzle,raster=[row[k] for k in ('cfg','swizzle','raster')];w=weights[layer]
  data=reference.generate_input(m,layer,31751,'outliers');x=data['x']
  fn=lambda:invoke(x,w,cfg,swizzle,raster)
  g=graph(fn);base=graph(lambda:mod.kernel_fn(data,state))
  for seed in (731,9817):
   x.copy_(reference.generate_input(m,layer,seed,'outliers')['x']);g[0].replay()
   ok,msg=reference.check_implementation(data,g[3],weights)
   if not ok:raise RuntimeError(msg)
  rr={'layer':layer,'m':m,'cfg':cfg,'swizzle':swizzle,'raster':raster,**paired(base,g,100)};result['recheck'].append(rr)
  print('TRANSPOSE_RECHECK',json.dumps({k:v for k,v in rr.items() if k!='samples_us'}),flush=True)
  # Nearby sizes validate tensor-edge handling. No tuning on these checks.
  for near in (m+2,m+7):
   data2=reference.generate_input(near,layer,98173,'outliers');out=invoke(data2['x'],w,cfg,swizzle,raster)
   ok,msg=reference.check_implementation(data2,out,weights);result['nearby'].append({'layer':layer,'m':near,'passed':ok,'message':msg})
   if not ok:raise RuntimeError(msg)
 result['resources']=subprocess.check_output(['cuobjdump','--dump-resource-usage',ext.__file__],text=True)[-24000:]
 result['promising_shapes']=[r for r in result['recheck'] if r['ci95'][0]>1.015 and r['ratio']>1.03]
 return result
