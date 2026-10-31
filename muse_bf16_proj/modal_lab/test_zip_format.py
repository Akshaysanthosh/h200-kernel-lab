"""Bounded H200 grouped 11-bit format test against protected pooler M=1."""
import hashlib,json,statistics,time
from pathlib import Path
from benchmark import load
from test_public import read_flush
from zip_format import pack_bits,unpack,source,audit
import numpy as np


def run(candidate,root='/work'):
 import torch,sys
 from torch.utils.cpp_extension import load_inline
 sys.path.insert(0,root);import reference
 mod=load(Path(root)/'candidate.py','zip_parent')
 base,cfg=mod.V18_SMALL['pooler',1]
 configs=[cfg,(1,8,16,1),(2,8,16,1)]
 instances=[]
 for cc in configs:
  inst=list(base);inst[0]=0;inst[6]=cc[0]*cc[1]*32;inst[10:14]=cc
  instances.append(tuple(inst))
 src=source(mod,instances)
 ext=load_inline('okc_zip_'+hashlib.sha256(src.encode()).hexdigest()[:12],mod.V18S_CPP,cuda_sources=src,functions=['v18s_launch','v18s_router_launch'],extra_cuda_cflags=['-O3','-lineinfo'],no_implicit_headers=True,with_pytorch_error_handling=False)
 parent=mod.v18s_build([base],name='zip_control_'+hashlib.sha256(str(base).encode()).hexdigest()[:12])
 print('BUILT',len(instances),flush=True)
 flush=torch.ones(128*1024*1024,device='cuda',dtype=torch.uint8);sink=torch.empty(flush.numel()//4096,device='cuda',dtype=torch.int32)
 def clear():read_flush[(sink.numel(),)](flush,sink,flush.numel(),4096)
 clear();torch.cuda.synchronize()
 def capture(fn):
  for _ in range(3):fn()
  torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):clear();a.record();out=fn();b.record()
  return g,a,b,out
 def measure(g):g[0].replay();g[2].synchronize();return g[1].elapsed_time(g[2])*1000
 def paired(a,b):
  aa=[];bb=[]
  for i in range(80):
   if i%2:bb.append(measure(b));aa.append(measure(a))
   else:aa.append(measure(a));bb.append(measure(b))
  ratios=np.array([(aa[i]+aa[i+1])/(bb[i]+bb[i+1]) for i in range(0,80,2)])
  boots=np.median(np.random.default_rng(17).choice(ratios,(2000,len(ratios))),axis=1)
  return {'ratio':float(np.median(ratios)),'ratio_ci95':np.quantile(boots,[.025,.975]).tolist(),'parent_us':statistics.median(aa),'candidate_us':statistics.median(bb),'samples_us':[aa,bb], 'judge_4us_subtracted_ratio':statistics.median([(aa[i]+aa[i+1]-8)/(bb[i]+bb[i+1]-8) for i in range(0,80,2)])}
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'audit':audit(),'rounds':[],'format':'three exponent planes, compact common bytes and exact BF16 exceptions, 1024-value group'}
 for seed in (82731,294817):
  weights=reference.generate_weights(seed);w=weights['pooler'];bits=w.view(torch.int16).cpu().numpy().view(np.uint16)
  pk=mod.v18s_pack_format(w,1);sp=mod.v18s_stream(pk,1,*cfg)
  parentbytes=sum(sp[k].numel()*sp[k].element_size() for k in ('wp','esc'))
  data=reference.generate_input(1,'pooler',seed+1);x=data['x'];out0=torch.empty((1,4096),device='cuda',dtype=torch.bfloat16)
  def parent_fn():
   rc=mod.v18s_launch_raw(parent,0,sp,x,out0,4096,cfg)
   if rc:raise RuntimeError('parent launch '+str(rc))
   return out0
  g0=capture(parent_fn);gaa=capture(parent_fn)
  rr={'seed':seed,'parent_bytes':parentbytes,'aa':paired(g0,gaa),'candidates':[]}
  for idx,cc in enumerate(configs):
   t=time.monotonic();p=pack_bits(bits,cc)
   assert np.array_equal(unpack(p),p['ordered'])
   cand={'wp':torch.from_numpy(p['packed']).to('cuda'),'esc':torch.from_numpy(p['meta'].view(np.int32)).to('cuda'),'fmt':0,'t01':p['base'],'u01':0}
   out=torch.empty_like(out0)
   def fn():
    rc=mod.v18s_launch_raw(ext,idx,cand,x,out,4096,cc)
    if rc:raise RuntimeError('candidate launch '+str(rc))
    return out
   checks=[]
   # Fresh activation and one-hot reference checks, also nearby M=7 and M=8.
   for m in (1,7,8):
    for dist in ('normal','outliers'):
     dd=reference.generate_input(m,'pooler',seed+11+m,dist)
     yy=torch.empty((m,4096),device='cuda',dtype=torch.bfloat16)
     rc=mod.v18s_launch_raw(ext,idx,cand,dd['x'],yy,4096,cc)
     if rc:raise RuntimeError('correctness launch '+str(rc))
     ok,msg=reference.check_implementation(dd,yy,weights)
     if not ok:raise RuntimeError(str((cc,m,dist,msg)))
     checks.append({'m':m,'dist':dist,'message':msg})
   gg=capture(fn)
   fresh=reference.generate_input(1,'pooler',seed+123,'outliers');x.copy_(fresh['x'])
   for g in (g0,gg):
    g[0].replay();ok,msg=reference.check_implementation(data,g[3],weights)
    if not ok:raise RuntimeError('fresh graph '+msg)
   timing=paired(g0,gg)
   resource=mod.v18s_launch_raw(ext,idx,cand,x,out,4096,cc,2)
   row={'cfg':cc,'bpw':p['bpw'],'exception_rate':p['exception_rate'],'prepare_seconds':time.monotonic()-t,'registers':resource//1000,'local_bytes':resource%1000,'correctness':checks,**timing}
   rr['candidates'].append(row)
   print('ZIP_RESULT',json.dumps({k:v for k,v in row.items() if k not in ('samples_us','correctness')}),flush=True)
  result['rounds'].append(rr)
 return result
