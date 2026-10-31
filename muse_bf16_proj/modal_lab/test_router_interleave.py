"""Cold-L2 paired tests of interleaved high/low router accumulation."""
import json,statistics,sys,time
from pathlib import Path
import numpy as np
from benchmark import load
from test_public import read_flush
import router_interleave as variant

def run(candidate,root='/work'):
 import torch
 sys.path.insert(0,root);import reference
 parent=load(Path(root)/'candidate.py','router_parent')
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
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'sweep':[],'confirmations':[],'controls':[]};selected={}
 began=time.monotonic()
 for seed in (82731,294817):
  weights=reference.generate_weights(seed);st=parent.prepare(weights);w=variant.pack(weights['router'])
  for m in (8,16,32,128,256):
   data=reference.generate_input(m,'router',seed+m,'outliers');x=data['x'];base=graph(lambda:parent.kernel_fn(data,st))
   aa=graph(lambda:parent.kernel_fn(data,st));result['controls'].append({'seed':seed,'m':m,**paired(base,aa,60)})
   if seed==82731:
    bm=16 if m<=16 else 32
    configs=[(bm,bn,bk,sk,4,3) for bn in (32,64) for bk in (64,128) for sk in (1,4,16)]
   else:configs=selected[m]
   rows=[]
   for cfg in configs:
    if time.monotonic()-began>760:result['truncated']=True;return result
    p=torch.empty((cfg[3],m,256),device='cuda',dtype=torch.float32);y=torch.empty((m,256),device='cuda',dtype=torch.float32)
    try:
     fn=lambda:variant.run(x,w,p,y,cfg)
     trial=graph(fn);trial[0].replay();ok,msg=reference.check_implementation(data,trial[3],weights)
     if not ok:raise RuntimeError('correctness '+msg)
     row={'seed':seed,'m':m,'config':cfg,'correctness':msg,**paired(base,trial,40 if seed==82731 else 120)}
     x.copy_(reference.generate_input(m,'router',seed+m+20,'normal')['x']);trial[0].replay()
     ok,msg=reference.check_implementation(data,trial[3],weights)
     if not ok:raise RuntimeError('fresh '+msg)
     row['fresh_correctness']=msg
     rows.append(row)
    except Exception as e:
     row={'seed':seed,'m':m,'config':cfg,'error':str(e)[-1800:]};result['passed']=False
    result['sweep' if seed==82731 else 'confirmations'].append(row)
   if seed==82731:
    selected[m]=[r['config'] for r in sorted(rows,key=lambda r:r['ratio'],reverse=True)[:2]]
    # Independent timing pass on the first seed after selection.
    for cfg in selected[m]:
     p=torch.empty((cfg[3],m,256),device='cuda',dtype=torch.float32);y=torch.empty((m,256),device='cuda',dtype=torch.float32)
     trial=graph(lambda:variant.run(x,w,p,y,cfg));row={'seed':seed,'m':m,'config':cfg,**paired(base,trial,120)}
     result['confirmations'].append(row)
   print('ROUTER_SHAPE',seed,m,json.dumps([{k:v for k,v in r.items() if k!='samples_us'} for r in result['confirmations'] if r['seed']==seed and r['m']==m]),flush=True)
 return result
