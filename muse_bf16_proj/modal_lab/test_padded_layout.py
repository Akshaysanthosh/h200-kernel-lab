"""Largest dense-cost shapes: profile and test weight leading-dimension padding."""
import hashlib,json,statistics,time,subprocess,shutil,sys,gc
from pathlib import Path
import numpy as np
from benchmark import load
from test_public import read_flush
from padded_layout import source


def run(candidate,root='/work'):
 import torch
 from torch.utils.cpp_extension import load_inline
 sys.path.insert(0,root);import reference
 mod=load(Path(root)/'candidate.py','padding_parent')
 weights=reference.generate_weights(82731);st=mod.prepare(weights)
 print('BASELINE_PREPARED',flush=True)
 src=source(mod)
 ext=load_inline('okc_pad_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp_sources=src,with_cuda=True,extra_cflags=['-O3','-std=c++20'],extra_ldflags=['-l:libcublasLt.so.13'],verbose=False)
 workspace=torch.empty(128<<20,device='cuda',dtype=torch.uint8);ctx=ext.Context(workspace)
 flush=torch.ones(128*1024*1024,device='cuda',dtype=torch.uint8);sink=torch.empty(flush.numel()//4096,device='cuda',dtype=torch.int32)
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
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'shapes':[],'recheck':[],'traces':[]}
 winners=[]
 for layer in ('pooler','unpooler'):
  n,k=weights[layer].shape
  for m in (128,256):
   data=reference.generate_input(m,layer,7271+m,'outliers');x=data['x'];out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
   baseline=graph(lambda:mod.kernel_fn(data,st));aa=paired(baseline,graph(lambda:mod.kernel_fn(data,st)),40)
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    baseline[0].replay();torch.cuda.synchronize()
   result['traces'].append({'layer':layer,'m':m,'kernels':[{'name':e.name,'us':e.device_time_total} for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA and 'read_flush' not in e.name]})
   layout,epi,descriptor=mod.tt_lt__LT_H200_FIXED[m,layer]
   rows=[];kept=[]
   # Packed layout is the parent's route. Both packed orientations are covered
   # by its row/column-major descriptor; padding never changes the logical GEMM.
   dense=weights[layer].t().contiguous() if layout>=2 else weights[layer]
   for pad in (0,8,16,32,64,128):
    backing=torch.empty((dense.shape[0],dense.shape[1]+pad),device='cuda',dtype=torch.bfloat16)
    backing.fill_(float('nan')) # Catch accidental reads of padding in output checks.
    view=backing[:,:dense.shape[1]];view.copy_(dense)
    assert torch.equal(view,dense)
    plan=ext.Plan(ctx,view,m,n,k,layout,epi)
    fixed=plan.restore(descriptor);plan.enumerate(False)
    indices=list(dict.fromkeys(([fixed] if fixed>=0 else [])+list(range(min(12,len(plan.metadata()))))))
    for idx in indices:
     rc=plan.run(x,out,idx)
     if rc:continue
     ok,msg=reference.check_implementation(data,out,weights)
     if not ok:raise RuntimeError(str((layer,m,pad,idx,msg)))
     def fn():
      status=plan.run(x,out,idx)
      if status:raise RuntimeError(str(status))
      return out
     g=graph(fn);timing=paired(baseline,g,12)
     row={'pad':pad,'layout':layout,'config':plan.metadata()[idx][:9],'index':idx,'padding_bytes':backing.numel()*2-dense.numel()*2,**timing}
     rows.append(row);kept.append((row,plan,idx,backing))
   final=[]
   # Include no-padding control and best padded alternatives.
   selected=sorted(kept,key=lambda r:-r[0]['ratio'])[:3]
   for row,plan,idx,backing in selected:
    def fn():
     rc=plan.run(x,out,idx)
     if rc:raise RuntimeError(str(rc))
     return out
    g=graph(fn);x.copy_(reference.generate_input(m,layer,9137+m,'normal')['x']);g[0].replay()
    ok,msg=reference.check_implementation(data,out,weights)
    if not ok:raise RuntimeError('fresh '+msg)
    record={**row,**paired(baseline,g,80),'correctness':msg}
    final.append(record)
   rec={'layer':layer,'m':m,'aa':aa,'tested':len(rows),'screen':rows,'finalists':final};result['shapes'].append(rec)
   top=max(final,key=lambda r:r['ratio']);winners.append((layer,m,top))
   print('PAD_FINAL',layer,m,json.dumps([{k:v for k,v in z.items() if k!='samples_us'} for z in final]),flush=True)
   del kept,selected,plan,backing,view;gc.collect()
 # Repeat selected layouts on an independent weight seed, no retuning.
 weights=reference.generate_weights(294817);st=mod.prepare(weights)
 for layer,m,row in winners:
  n,k=weights[layer].shape;layout=row['layout'];pad=row['pad']
  dense=weights[layer].t().contiguous() if layout>=2 else weights[layer]
  backing=torch.empty((dense.shape[0],dense.shape[1]+pad),device='cuda',dtype=torch.bfloat16);backing.fill_(float('nan'));view=backing[:,:dense.shape[1]];view.copy_(dense)
  plan=ext.Plan(ctx,view,m,n,k,layout,0);idx=plan.restore(row['config'])
  if idx<0:raise RuntimeError('failed to restore finalist')
  data=reference.generate_input(m,layer,18731+m,'outliers');x=data['x'];out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
  def fn():
   rc=plan.run(x,out,idx)
   if rc:raise RuntimeError(str(rc))
   return out
  g=graph(fn);base=graph(lambda:mod.kernel_fn(data,st))
  for seed in (371,9817):
   x.copy_(reference.generate_input(m,layer,seed,'outliers')['x']);g[0].replay();ok,msg=reference.check_implementation(data,out,weights)
   if not ok:raise RuntimeError(msg)
  rr={'layer':layer,'m':m,'pad':pad,'config':row['config'],**paired(base,g,100)};result['recheck'].append(rr)
  print('PAD_RECHECK',json.dumps({k:v for k,v in rr.items() if k!='samples_us'}),flush=True)
 # Profile the largest single dense operation without clock-control changes.
 ncu=shutil.which('ncu')
 if ncu:
  code="import sys;sys.path.insert(0,'/work');import torch,reference;from benchmark import load;from pathlib import Path;m=load(Path('/work/candidate.py'),'ncu_parent');s=m.prepare(reference.generate_weights(82731));d=reference.generate_input(256,'pooler',9137);torch.cuda.synchronize();torch.cuda.profiler.start();m.kernel_fn(d,s);torch.cuda.synchronize();torch.cuda.profiler.stop()"
  try:
   p=subprocess.run([ncu,'--clock-control','none','--profile-from-start','off','--section','SpeedOfLight','--section','WarpStateStats','--csv',sys.executable,'-c',code],capture_output=True,text=True,timeout=180)
   result['ncu_returncode']=p.returncode;result['ncu_output']=(p.stdout+p.stderr)[-40000:]
  except subprocess.TimeoutExpired:result['ncu_output']='Timed out; no counter conclusion.'
 return result
