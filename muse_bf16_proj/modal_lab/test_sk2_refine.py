"""Lab-only comparison of public warp-specialized correction kernels."""
import json,statistics,sys,time
from pathlib import Path
import numpy as np
from benchmark import load
from test_public import read_flush


def run(candidate,root='/work'):
 import torch
 sys.path.insert(0,root);import reference
 parent=load(Path(root)/'parent.py','protected_parent')
 alt=load(Path(root)/'candidate.py','narrow_candidate')
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
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'timings':[],'nearby':[],'controls':[]}
 original=list(alt.vj_SK2_CONFIGS);configs=[];keys=set()
 for c in original:
  for cap in (96,128,144,176,208):
   for accum in (1,2):
    cfg=list(c);cfg[0]=c[1]+'_a'+str(accum)+'_r'+str(cap);cfg[8]=accum
    hl,stages=alt.vj__sk2_geom(cfg,32);effective=1 if stages>=4 and cfg[4]==4 else accum
    key=(cfg[1],stages,effective)
    if key in keys:continue
    keys.add(key);cfg[8]=effective;configs.append(tuple(cfg))
 lines=[]
 for ci,c in enumerate(configs):
  name,layer,nrows,R,NT,NWR,WSK,UPW,NACC,CL=c
  _,stages=alt.vj__sk2_geom(c,32)
  lines.append(f'if (cfg=={ci} && tokr==32) {{ launch_sk2<2,32,{NT},{NWR},{WSK},{UPW},{stages},true,false,{NACC},{CL},__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg); return; }}')
 src=alt.vj_CUDA_SRC.replace('@@MBAR@@',alt.vj_MBAR_HELPERS).replace('@@BODY@@',alt.vj_WG_HELPERS)+alt.vj_MAP_SRC+alt.vj_SK2_SRC.replace('@@SK2_DISPATCH@@','\n'.join(lines))
 from torch.utils.cpp_extension import load_inline
 import hashlib
 cpp='#include <torch/extension.h>\nvoid sk2(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,torch::Tensor);'
 ext=load_inline('sk2_refine_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp_sources=cpp,cuda_sources=src,functions=['sk2'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],no_implicit_headers=True,verbose=False)
 result['configs']=configs;result['compiled_source_sha256']=hashlib.sha256(src.encode()).hexdigest()
 for seed in (82731,294817):
  weights=reference.generate_weights(seed);st=parent.prepare(weights);narrow=alt.prepare(weights)
  ts={'dummy':torch.zeros(2,device='cuda',dtype=torch.int32)}
  alt.vj_SK2_CONFIGS=configs
  try:alt.vj_sk2_prepare(weights,ts)
  finally:alt.vj_SK2_CONFIGS=original
  for layer in ('qkv_gate','pooler','unpooler'):
   data=reference.generate_input(32,layer,seed+32,'outliers');x=data['x'];c=data['config']
   base=graph(lambda:parent.kernel_fn(data,st));control=graph(lambda:alt.kernel_fn(data,narrow));aa=graph(lambda:parent.kernel_fn(data,st))
   result['controls'].append({'seed':seed,'layer':layer,'aa':paired(base,aa,80),'narrow':paired(base,control,120)})
   fresh=reference.generate_input(32,layer,seed+432,'normal')['x'];initial=x.clone()
   for cfg in configs:
    if cfg[1]!=layer:continue
    ci,stream,tab,pmax,b4,n=ts['sk2'][cfg[0]]
    def fn():
     out=torch.empty((32,c['n']),device=x.device,dtype=torch.bfloat16)
     ext.sk2(stream,x,out,tab,pmax,b4,32,c['n'],c['k'],ci,32,0,ts['dummy'])
     return out
    x.copy_(initial);g=graph(fn);g[0].replay();ok,msg=reference.check_implementation(data,g[3],weights)
    if not ok:raise RuntimeError(msg)
    rr={'seed':seed,'layer':layer,'config':cfg,'correctness':msg,'vs_parent':paired(base,g,80),'vs_narrow':paired(control,g,120)}
    x.copy_(fresh);g[0].replay();ok,msg=reference.check_implementation(data,g[3],weights)
    if not ok:raise RuntimeError('fresh '+msg)
    rr['fresh_correctness']=msg;result['timings'].append(rr)
   print('REFINE_LAYER',seed,layer,json.dumps([{k:v for k,v in r.items() if k not in ('correctness','fresh_correctness')} for r in result['timings'] if r['seed']==seed and r['layer']==layer],default=str),flush=True)
  for m in (17,24,31):
   for layer in ('qkv_gate','pooler','unpooler'):
    data=reference.generate_input(m,layer,seed+m+900,'outliers');base=graph(lambda:parent.kernel_fn(data,st));g=graph(lambda:alt.kernel_fn(data,narrow));g[0].replay();ok,msg=reference.check_implementation(data,g[3],weights)
    if not ok:raise RuntimeError('nearby '+msg)
    result['nearby'].append({'seed':seed,'m':m,'layer':layer,'correctness':msg,**paired(base,g,80)})
 return result
