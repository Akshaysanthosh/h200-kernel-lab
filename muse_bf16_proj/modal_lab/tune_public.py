"""Lab-only cuBLASLt schedule search; no tuning code enters submissions."""
import json,statistics,time,gc
from pathlib import Path
from benchmark import load
from test_public import read_flush

def run(candidate,root='/work'):
 import torch,sys
 sys.path.insert(0,root);import reference
 mod=load(Path(root)/'candidate.py','public_tune');weights=reference.generate_weights(82731)
 t=time.monotonic();st=mod.prepare(weights);print('PREPARED',time.monotonic()-t,flush=True)
 lt=st['new_lt'];flush=torch.ones(128*1024*1024,device='cuda',dtype=torch.uint8);sink=torch.empty(flush.numel()//4096,device='cuda',dtype=torch.int32)
 def clear():read_flush[(sink.numel(),)](flush,sink,flush.numel(),4096)
 clear();torch.cuda.synchronize()
 def graph(fn):
  for _ in range(3):fn()
  torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):clear();a.record();out=fn();b.record()
  return g,a,b,out
 def us(g):g[0].replay();g[2].synchronize();return g[1].elapsed_time(g[2])*1000
 def paired(a,b,reps=20):
  aa=[];bb=[]
  for i in range(reps):
   if i%2:bb.append(us(b));aa.append(us(a))
   else:aa.append(us(a));bb.append(us(b))
  return statistics.median([(aa[i]+aa[i+1])/(bb[i]+bb[i+1]) for i in range(0,reps,2)]),statistics.median(aa),statistics.median(bb)
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'lt_version':lt['version'],'shapes':[]}
 for m,layer in [(128,'pooler'),(256,'pooler'),(128,'unpooler'),(256,'unpooler'),(128,'qkv_gate'),(256,'qkv_gate')]:
  data=reference.generate_input(m,layer,500+m,'outliers');x=data['x'];out=torch.empty((m,weights[layer].shape[0]),device='cuda',dtype=torch.bfloat16)
  baseline=graph(lambda:mod.kernel_fn(data,st));ranked=[];plans=[];tested=0
  def assess(plan,idx,layout):
   nonlocal tested
   rc=plan.run(x,out,idx)
   if rc:return
   # Quick rejection before timing; complete checks apply to finalists.
   if not bool(torch.isfinite(out).all()):return
   g=graph(lambda:(plan.run(x,out,idx),out)[1]);ratio,a,b=paired(baseline,g,6);tested+=1
   ranked.append((ratio,layout,idx,plan))
  for layout in range(4):
   plan=mod.tt_lt__lt_plan(lt,layer,m,layout,0);plans.append(plan)
   seeds=mod.tt_lt__LT_SEARCH_SEEDS.get((m,layer),[])+[mod.tt_lt__LT_H200_FIXED[m,layer]]
   for lay,epi,cfg in seeds:
    if lay==layout:plan.restore(cfg)
   plan.enumerate(False)
   for i in range(min(len(plan.metadata()),192)):assess(plan,i,layout)
   best=sorted([r for r in ranked if r[1]==layout],key=lambda r:-r[0])[:2]
   old=len(plan.metadata());plan.expand([r[2] for r in best])
   for i in range(old,min(len(plan.metadata()),old+96)):assess(plan,i,layout)
   print('SEARCH',m,layer,layout,tested,'best',max(r[0] for r in ranked),flush=True)
  finalists=[]
  for _,layout,idx,plan in sorted(ranked,key=lambda r:-r[0])[:4]:
   g=graph(lambda:(plan.run(x,out,idx),out)[1]);ratio,a,b=paired(baseline,g,60)
   ok,msg=reference.check_implementation(data,out,weights)
   if not ok:continue
   fresh=reference.generate_input(m,layer,9137+m,'normal');x.copy_(fresh['x']);g[0].replay();ok,msg=reference.check_implementation(data,out,weights)
   if not ok:continue
   finalists.append({'ratio':ratio,'baseline_us':a,'candidate_us':b,'layout':layout,'epilogue':0,'config':plan.metadata()[idx][:9],'correctness':msg})
  row={'m':m,'layer':layer,'tested':tested,'finalists':sorted(finalists,key=lambda r:-r['ratio'])};result['shapes'].append(row);print('FINALISTS',json.dumps(row),flush=True)
  del baseline,plans,ranked;gc.collect()
 return result
