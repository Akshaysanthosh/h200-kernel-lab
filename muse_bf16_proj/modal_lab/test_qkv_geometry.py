"""QKV output-dimension padding: exact weights and sliced output, lab-only."""
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
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'shapes':[],'recheck':[]}
 def setup(weights,m,npad,layout):
  w=weights['qkv_gate'];n,k=w.shape
  dense=torch.zeros((npad,k),device=w.device,dtype=w.dtype);dense[:n].copy_(w)
  assert torch.equal(dense[:n].view(torch.int16),w.view(torch.int16))
  if layout>=2:dense=dense.t().contiguous()
  out=torch.empty((m,npad),device=w.device,dtype=w.dtype)
  plan=ext.Plan(ctx,dense,m,npad,k,layout,0)
  return dense,out,plan
 winners=[]
 for m in (128,256):
  data=reference.generate_input(m,'qkv_gate',7271+m,'outliers');x=data['x'];n=2304
  base=graph(lambda:mod.kernel_fn(data,st));aa=paired(base,graph(lambda:mod.kernel_fn(data,st)),40)
  rows=[];kept=[]
  for npad in (2304,2560,3072,4096):
   for layout in (0,1,2,3):
    dense,out,plan=setup(weights,m,npad,layout)
    desc=mod.tt_lt__LT_H200_FIXED[m,'qkv_gate'][2]
    fixed=plan.restore(desc);plan.enumerate(False)
    indices=list(dict.fromkeys(([fixed] if fixed>=0 else [])+list(range(min(12,len(plan.metadata()))))))
    for idx in indices:
     if plan.run(x,out,idx):continue
     ok,msg=reference.check_implementation(data,out[:,:n],weights)
     if not ok:raise RuntimeError(str((m,npad,layout,idx,msg)))
     def fn():
      rc=plan.run(x,out,idx)
      if rc:raise RuntimeError(str(rc))
      return out[:,:n]
     g=graph(fn);row={'npad':npad,'layout':layout,'config':plan.metadata()[idx][:9],**paired(base,g,12)}
     rows.append(row);kept.append((row,dense,out,plan,idx))
  final=[]
  for row,dense,out,plan,idx in sorted(kept,key=lambda z:-z[0]['ratio'])[:3]:
   def fn():
    rc=plan.run(x,out,idx)
    if rc:raise RuntimeError(str(rc))
    return out[:,:n]
   g=graph(fn);x.copy_(reference.generate_input(m,'qkv_gate',9137+m,'normal')['x']);g[0].replay()
   ok,msg=reference.check_implementation(data,g[3],weights)
   if not ok:raise RuntimeError(msg)
   final.append({**row,**paired(base,g,100)})
  result['shapes'].append({'m':m,'aa':aa,'screen':rows,'finalists':final})
  winners.append((m,max(final,key=lambda z:z['ratio'])))
  print('QKV_FINAL',m,json.dumps([{k:v for k,v in z.items() if k!='samples_us'} for z in final]),flush=True)
  del kept;gc.collect()
 weights=reference.generate_weights(294817);st=mod.prepare(weights)
 for m,row in winners:
  dense,out,plan=setup(weights,m,row['npad'],row['layout']);idx=plan.restore(row['config'])
  assert idx>=0
  data=reference.generate_input(m,'qkv_gate',18731+m,'outliers');x=data['x']
  def fn():
   rc=plan.run(x,out,idx)
   if rc:raise RuntimeError(str(rc))
   return out[:,:2304]
  g=graph(fn);base=graph(lambda:mod.kernel_fn(data,st))
  for seed in (731,9817):
   x.copy_(reference.generate_input(m,'qkv_gate',seed,'outliers')['x']);g[0].replay()
   ok,msg=reference.check_implementation(data,g[3],weights)
   if not ok:raise RuntimeError(msg)
  record={k:v for k,v in row.items() if k not in ('samples_us','ratio','ci95','parent_us','candidate_us')}
  record.update(m=m,**paired(base,g,120));result['recheck'].append(record)
  print('QKV_RECHECK',json.dumps({k:v for k,v in record.items() if k!='samples_us'}),flush=True)
 return result
