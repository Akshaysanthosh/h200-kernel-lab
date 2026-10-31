"""Lab-only comparison of public warp-specialized correction kernels."""
import json,statistics,sys,time
from pathlib import Path
import numpy as np
from benchmark import load
from test_public import read_flush


def run(candidate,root='/work'):
 import torch
 sys.path.insert(0,root);import reference
 parent=load(Path(root)/'candidate.py','protected_parent')
 alt=load(Path(root)/'vjeux.py','vjeux_research')
 alt._get_ext()
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
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'timings':[],'controls':[],'nearby':[],'prepare_seconds':[]}
 for seed in (82731,294817):
  weights=reference.generate_weights(seed);t=time.monotonic();st=parent.prepare(weights)
  result['prepare_seconds'].append({'seed':seed,'parent':time.monotonic()-t})
  states={}
  for balance in (True,False):
   alt.SK2_BALANCE=balance
   state={'dummy':torch.zeros(2,device='cuda',dtype=torch.int32)}
   t=time.monotonic();alt.sk2_prepare(weights,state);states[balance]=state
   result['prepare_seconds'][-1][str(balance)]=time.monotonic()-t
  print('SPECIALIZED_PREPARED',seed,flush=True)
  for m in (8,16,32):
   for layer in ('qkv_gate','pooler','unpooler','o'):
    data=reference.generate_input(m,layer,seed+m,'outliers');x=data['x'];cfg=data['config']
    base=graph(lambda:parent.kernel_fn(data,st));aa=graph(lambda:parent.kernel_fn(data,st))
    result['controls'].append({'seed':seed,'m':m,'layer':layer,**paired(base,aa,60)})
    for balance in ((True,False) if layer=='qkv_gate' else (True,)):
     state=states[balance]
     fn=lambda:alt._run_sk2(x,layer,m,cfg['n'],cfg['k'],state)
     g=graph(fn);g[0].replay();ok,msg=reference.check_implementation(data,g[3],weights)
     if not ok:raise RuntimeError(str((m,layer,balance))+msg)
     rr={'seed':seed,'m':m,'layer':layer,'balance':balance,'correctness':msg,**paired(base,g,120)}
     x.copy_(reference.generate_input(m,layer,seed+m+100,'normal')['x']);g[0].replay()
     ok,msg=reference.check_implementation(data,g[3],weights)
     if not ok:raise RuntimeError('fresh '+msg)
     rr['fresh_correctness']=msg
     saved=fn();snapshot=saved.clone();other=fn()
     if not torch.equal(saved,snapshot):raise RuntimeError('output lifetime')
     result['timings'].append(rr)
     print('SPECIALIZED_TIMING',json.dumps({k:v for k,v in rr.items() if k!='samples_us'}),flush=True)
  for m in (7,9,15,17,31):
   for layer in ('qkv_gate','pooler','unpooler','o'):
    data=reference.generate_input(m,layer,seed+m+200,'outliers');cfg=data['config']
    out=alt._run_sk2(data['x'],layer,m,cfg['n'],cfg['k'],states[True]);ok,msg=reference.check_implementation(data,out,weights)
    result['nearby'].append({'seed':seed,'m':m,'layer':layer,'passed':ok,'message':msg})
    if not ok:raise RuntimeError('nearby '+str((m,layer))+msg)
 return result
