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
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'lt_version':lt['version'],'shapes':[],'pipeline':True,'revision':3}
 # Reuse the public lossless register decoder, extending its WGMMA tile
 # to 64 tokens. Weight bits are restored before multiplication.
 instances=[(bn,64,wk,pf,2,True,1) for bn in (64,128,256) for wk in (1,2) for pf in (1,2)]
 from torch.utils.cpp_extension import load_inline
 from pipeline_source import source
 import hashlib
 src=source(mod,instances,True)
 Path('/tmp/pipeline.cu').write_text(src)
 ext=load_inline('okc_pipe_'+hashlib.sha256(src.encode()).hexdigest()[:12],mod.dd_decodediet_tt_mw_CPP,cuda_sources=src,functions=['launch','info'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','--keep','-gencode=arch=compute_90a,code=sm_90a'])
 result['pipeline_source_sha256']=hashlib.sha256(src.encode()).hexdigest()
 result['resource_info']=[{'inst':inst,'registers':ext.info(i)//10000,'local_bytes':ext.info(i)%10000} for i,inst in enumerate(instances)]
 # Preserve a bounded assembly diagnostic before correctness checks.
 import subprocess,re
 try:
  sass=subprocess.check_output(['cuobjdump','--dump-sass',ext.__file__],text=True)
  parts=sass.split('Function :')
  part=next(x for x in parts if 'm32w_gemm' in x and 'HGMMA' in x)
  lines=part.splitlines();start=next(i for i,x in enumerate(lines) if 'HGMMA' in x)
  diagnostic='\n'.join(lines[max(0,start-4):start+100])
  result['sass_excerpt']=diagnostic
  print('SASS_EXCERPT',diagnostic,flush=True)
 except Exception as e:result['sass_error']=str(e)
 print('COMPRESSED_BUILT',len(instances),flush=True)
 for layer in ('unpooler','pooler','qkv_gate'):
  packed={}
  for wk in (1,2):
   for split in (1,2,4,8):
    packed[wk,split]=mod.dd_decodediet_tt_mw_pack_weight(weights[layer],64,wk,split,True)
  for m in (128,256):
   data=reference.generate_input(m,layer,500+m,'outliers');x=data['x'];out=torch.empty((m,weights[layer].shape[0]),device='cuda',dtype=torch.bfloat16)
   baseline=graph(lambda:mod.kernel_fn(data,st));ranked=[];tested=0
   for idx,inst in enumerate(instances):
    if ext.info(idx)%10000:continue
    for split in (1,2,4,8):
     pk=packed[inst[2],split]
     if pk is None:continue
     def fn():
      rc=mod.dd_decodediet_tt_mw_launch_raw(ext,idx,pk,x,out)
      if rc:raise RuntimeError('launch '+str(rc))
      return out
     fn();ok,msg=reference.check_implementation(data,out,weights)
     if not ok:raise RuntimeError('compressed correctness '+str(inst)+' '+str(split)+' '+msg)
     g=graph(fn);ratio,a,b=paired(baseline,g,12);tested+=1
     ranked.append({'ratio':ratio,'baseline_us':a,'candidate_us':b,'inst':inst,'split':split,'idx':idx,'max_esc':pk['max_esc'],'registers_per_thread':ext.info(idx)//10000})
   finalists=[]
   for row in sorted(ranked,key=lambda r:-r['ratio'])[:3]:
    idx=row['idx'];pk=packed[row['inst'][2],row['split']]
    g=graph(lambda:(mod.dd_decodediet_tt_mw_launch_raw(ext,idx,pk,x,out),out)[1]);ratio,a,b=paired(baseline,g,60)
    row.update(ratio=ratio,baseline_us=a,candidate_us=b)
    for seed in (9137,19721):
     fresh=reference.generate_input(m,layer,seed+m,'outliers');x.copy_(fresh['x']);g[0].replay()
     ok,msg=reference.check_implementation(data,out,weights)
     if not ok:raise RuntimeError('fresh graph '+msg)
    finalists.append(row)
   rec={'m':m,'layer':layer,'tested':tested,'finalists':finalists,'screen':ranked};result['shapes'].append(rec);print('FINALISTS',json.dumps(rec),flush=True)
  del packed;gc.collect()
 return result
