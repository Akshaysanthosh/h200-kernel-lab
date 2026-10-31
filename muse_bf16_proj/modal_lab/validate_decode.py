"""Independent full-workload validation plus profile-directed memory-load test."""
def run(candidate,root='/work'):
 from test_public_refine import run as validate
 result=validate(candidate,root)
 import torch,sys,json,hashlib,statistics
 from pathlib import Path
 from benchmark import load
 from test_public import read_flush
 from torch.utils.cpp_extension import load_inline
 sys.path.insert(0,root);import reference
 mod=load(Path(root)/'parent.py','memory_parent');w=reference.generate_weights(193071);st=mod.prepare(w)
 inst,cfg=mod.V18_SMALL['pooler',1];pk=mod.v18s_pack_format(w['pooler'],1);sp=mod.v18s_stream(pk,inst[2],*cfg)
 d=reference.generate_input(1,'pooler',18327,'outliers');x=d['x'];out=torch.empty((1,4096),device='cuda',dtype=torch.bfloat16)
 flush=torch.ones(128*1024*1024,device='cuda',dtype=torch.uint8);sink=torch.empty(flush.numel()//4096,device='cuda',dtype=torch.int32)
 def clear():read_flush[(sink.numel(),)](flush,sink,flush.numel(),4096)
 clear();torch.cuda.synchronize()
 def graph(fn):
  for _ in range(3):fn()
  torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):clear();a.record();o=fn();b.record()
  return g,a,b,o
 def us(g):g[0].replay();g[2].synchronize();return g[1].elapsed_time(g[2])*1000
 base=graph(lambda:mod.kernel_fn(d,st));rows=[]
 original=mod.v18s_cuda_source([inst]);old='ld.global.nc.L1::no_allocate.v4.s32'
 for policy in [old,'ld.global.nc.L1::no_allocate.L2::128B.v4.s32','ld.global.nc.L1::no_allocate.L2::256B.v4.s32','ld.global.cg.v4.s32']:
  assert original.count(old)==1
  src=original.replace(old,policy)
  ext=load_inline('okc_mem_'+hashlib.sha256(src.encode()).hexdigest()[:12],mod.V18S_CPP,cuda_sources=src,functions=['v18s_launch','v18s_router_launch'],extra_cuda_cflags=['-O3','-lineinfo'],no_implicit_headers=True,with_pytorch_error_handling=False)
  def fn():
   rc=mod.v18s_launch_raw(ext,0,sp,x,out,4096,cfg)
   if rc:raise RuntimeError(str(rc))
   return out
  g=graph(fn)
  for seed in (1827,39271):
   x.copy_(reference.generate_input(1,'pooler',seed,'outliers')['x']);g[0].replay();ok,msg=reference.check_implementation(d,out,w)
   if not ok:raise RuntimeError('cache policy '+msg)
  aa=[];bb=[]
  for i in range(100):
   if i%2:bb.append(us(g));aa.append(us(base))
   else:aa.append(us(base));bb.append(us(g))
  ratio=statistics.median([(aa[i]+aa[i+1])/(bb[i]+bb[i+1]) for i in range(0,100,2)])
  row={'policy':policy,'ratio':ratio,'baseline_us':statistics.median(aa),'candidate_us':statistics.median(bb)};rows.append(row);print('MEMORY_POLICY',json.dumps(row),flush=True)
 result['memory_policy']=rows
 return result
