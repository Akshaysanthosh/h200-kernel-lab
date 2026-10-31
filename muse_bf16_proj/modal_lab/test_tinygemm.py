"""Lab-only FlashInfer TinyGEMM comparison; no harness imports enter candidates."""
import hashlib,json,statistics,time
from pathlib import Path
from benchmark import load
from test_public import read_flush

def run(candidate,root='/work'):
 import torch,sys
 from torch.utils.cpp_extension import load_inline
 sys.path.insert(0,root);import reference
 mod=load(Path(root)/'candidate.py','tiny_parent')
 weights=reference.generate_weights(82731)
 t=time.monotonic();st=mod.prepare(weights);print('PREPARED',time.monotonic()-t,flush=True)
 src=(Path(root)/'tinygemm.cu').read_text()
 cpp='#include <cstdint>\n#include <pybind11/pybind11.h>\nint launch(uint64_t,uint64_t,uint64_t,int,int,int,uint64_t,int);'
 ext=load_inline('okc_tiny_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp,cuda_sources=src,functions=['launch'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a'],extra_ldflags=['-lcuda'])
 print('TINY_BUILT',flush=True)
 flush=torch.ones(128*1024*1024,device='cuda',dtype=torch.uint8);sink=torch.empty(flush.numel()//4096,device='cuda',dtype=torch.int32)
 def clear():read_flush[(sink.numel(),)](flush,sink,flush.numel(),4096)
 clear();torch.cuda.synchronize()
 def graph(fn):
  for _ in range(3):fn()
  torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):clear();a.record();out=fn();b.record()
  return g,a,b,out
 def us(g):g[0].replay();g[2].synchronize();return g[1].elapsed_time(g[2])*1000
 def paired(a,b,reps=40):
  aa=[];bb=[]
  for i in range(reps):
   if i%2:bb.append(us(b));aa.append(us(a))
   else:aa.append(us(a));bb.append(us(b))
  return statistics.median([(aa[i]+aa[i+1])/(bb[i]+bb[i+1]) for i in range(0,reps,2)]),statistics.median(aa),statistics.median(bb)
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'approach':'FlashInfer TinyGEMM BF16 uncompressed','upstream_commit':'5d60811431d040ed62f85c7a7906a77a6e72cccc','kernel_sha256':hashlib.sha256(src.encode()).hexdigest(),'shapes':[]}
 for layer in ('pooler','qkv_gate','unpooler','o'):
  for m in (1,8,16,32):
   data=reference.generate_input(m,layer,500+m,'outliers');x=data['x'];n=reference.out_features(layer,m);k=x.shape[1];w=weights[layer][:n];out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16);stream=torch.cuda.current_stream().cuda_stream
   baseline=graph(lambda:mod.kernel_fn(data,st));rows=[]
   for stages in (4,8,16):
    def fn():
     rc=ext.launch(x.data_ptr(),w.data_ptr(),out.data_ptr(),m,n,k,torch.cuda.current_stream().cuda_stream,stages)
     if rc:raise RuntimeError('tiny launch '+str(rc))
     return out
    fn();ok,msg=reference.check_implementation(data,out,weights)
    if not ok:raise RuntimeError('tiny correctness '+str((layer,m,stages))+' '+msg)
    g=graph(fn);ratio,a,b=paired(baseline,g)
    for seed in (9137,19721):
     fresh=reference.generate_input(m,layer,seed+m,'outliers');x.copy_(fresh['x']);g[0].replay()
     ok,msg=reference.check_implementation(data,out,weights)
     if not ok:raise RuntimeError('fresh graph '+str((layer,m,stages,seed))+' '+msg)
    rows.append({'stages':stages,'ratio':ratio,'baseline_us':a,'candidate_us':b})
   rec={'layer':layer,'m':m,'results':rows};result['shapes'].append(rec);print('TINY_SHAPE',json.dumps(rec),flush=True)
 return result
