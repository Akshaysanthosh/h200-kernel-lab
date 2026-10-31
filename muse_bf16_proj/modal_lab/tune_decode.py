"""Bounded decoder/scheduling/reuse comparison against protected public parent."""
import hashlib,json,statistics,time,gc
from pathlib import Path
from benchmark import load
from test_public import read_flush

def run(candidate,root='/work'):
 import torch,sys
 from torch.utils.cpp_extension import load_inline
 from decode_variants import source,pack,audit_integer_decoder
 sys.path.insert(0,root);import reference
 mod=load(Path(root)/'candidate.py','decode_parent');weights=reference.generate_weights(82731)
 t=time.monotonic();st=mod.prepare(weights);print('PREPARED',time.monotonic()-t,flush=True)
 targets=[]
 for layer,m,base,cfg in [('pooler',1,*mod.V18_SMALL['pooler',1]),('o',8,*mod.V18_SMALL['o',8]),('qkv_gate',1,(1,1,1,2,2,0,256,0,4608,4096,1,4,16,4,0),(1,4,16,4))]:
  # QKV dimensions/config are derived from the actual tensor rather than assumed.
  if layer=='qkv_gate':
   n,k=weights[layer].shape;cfg=(1,4,k//(4*4*64),4);base=(1,1,1,2,2,0,256,0,n,k,*cfg,0)
  opts=[]
  for stage in (1,2,4):
   if cfg[2]%stage:continue
   for opt in (0,2,4,6):
    inst=list(base);inst[3]=stage;inst[14]=opt;opts.append((tuple(inst),cfg,'native'))
  if layer=='pooler':
   for wr,wk in ((1,8),(2,8),(1,16)):
    altcfg=(wr,wk,128//wk,1)
    for stage in (1,2):
     inst=list(base);inst[3]=stage;inst[6]=wr*wk*32;inst[10:14]=altcfg;inst[14]=6;opts.append((tuple(inst),altcfg,'occupancy'))
  targets.append((layer,m,opts))
 instances=list(dict.fromkeys(inst for _,_,opts in targets for inst,cfg,kind in opts))
 exts={}
 for variant in ('lookup','arithmetic'):
  src=mod.v18s_cuda_source(instances) if variant=='lookup' else source(mod,instances)
  exts[variant]=load_inline('okc_dec_'+hashlib.sha256(src.encode()).hexdigest()[:12],mod.V18S_CPP,cuda_sources=src,functions=['v18s_launch','v18s_router_launch'],extra_cuda_cflags=['-O3','-lineinfo'],no_implicit_headers=True,with_pytorch_error_handling=False)
  print('BUILT',variant,len(instances),flush=True)
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
  return statistics.median([(aa[i]+aa[i+1])/(bb[i]+bb[i+1]) for i in range(0,reps,2)]),statistics.median(aa),statistics.median(bb)
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'integer_audit':audit_integer_decoder(),'shapes':[],'skipped':[]}
 for layer,m,opts in targets:
  data=reference.generate_input(m,layer,400+m,'outliers');x=data['x'];n=reference.out_features(layer,m);out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
  baseline=graph(lambda:mod.kernel_fn(data,st));streams={};ranked=[]
  for variant in ('lookup','arithmetic'):
   pk=(mod.v18s_pack_format if variant=='lookup' else lambda w,f:pack(mod,w,f))(weights[layer][:n].contiguous(),1)
   if pk is None:continue
   for inst,cfg,kind in opts:
    key=(variant,inst[2],cfg)
    if key not in streams:streams[key]=mod.v18s_stream(pk,inst[2],*cfg)
    sp=streams[key]
    if sp is None:continue
    idx=instances.index(inst);ext=exts[variant]
    info=mod.v18s_launch_raw(ext,idx,sp,x,out,n,cfg,2);occ=mod.v18s_launch_raw(ext,idx,sp,x,out,n,cfg,1)
    if info<0 or info%1000 or occ<=0:
     result['skipped'].append({'layer':layer,'inst':inst,'variant':variant,'info':info,'occupancy':occ});continue
    def fn():
     rc=mod.v18s_launch_raw(ext,idx,sp,x,out,n,cfg)
     if rc:raise RuntimeError('launch '+str(rc)+' '+str(inst))
     return out
    fn();ok,msg=reference.check_implementation(data,out,weights)
    if not ok:raise RuntimeError('correctness '+str((layer,m,variant,inst))+' '+msg)
    g=graph(fn)
    fresh=reference.generate_input(m,layer,9137+m,'outliers');x.copy_(fresh['x']);g[0].replay();ok,msg=reference.check_implementation(data,out,weights)
    if not ok:raise RuntimeError('fresh graph '+msg)
    ratio,a,b=paired(baseline,g,20)
    ranked.append({'variant':variant,'inst':inst,'cfg':cfg,'ratio':ratio,'baseline_us':a,'candidate_us':b,'registers':info//1000,'occupancy':occ,'max_esc':sp['max_esc']})
  finalists=[]
  for row in sorted(ranked,key=lambda r:-r['ratio'])[:3]:
   ext=exts[row['variant']];inst=row['inst'];cfg=row['cfg'];idx=instances.index(inst);sp=streams[row['variant'],inst[2],cfg]
   def fn():
    rc=mod.v18s_launch_raw(ext,idx,sp,x,out,n,cfg)
    if rc:raise RuntimeError(str(rc))
    return out
   g=graph(fn);ratio,a,b=paired(baseline,g,80);row=dict(row,ratio=ratio,baseline_us=a,candidate_us=b);finalists.append(row)
  rec={'layer':layer,'m':m,'screen':ranked,'finalists':finalists};result['shapes'].append(rec);print('FINALISTS',json.dumps(rec),flush=True)
  del streams;gc.collect()
 # Reorder HMMAs to keep the same weight registers live across token groups.
 from reuse_variant import source as reuse_source
 src=reuse_source(mod)
 ext=load_inline('okc_reuse_'+hashlib.sha256(src.encode()).hexdigest()[:12],mod.b2p_tt_v10_m32x__CPP,cuda_sources=src,functions=['r3_pool32_pooler_d9_cap2_a_m32x_launch'],extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'],no_implicit_headers=True,with_pytorch_error_handling=False)
 plan=mod.b2p_tt_v10_m32x__make_plan(ext,weights['pooler'],1)
 if plan is None:raise RuntimeError('reuse packing failed')
 data=reference.generate_input(32,'pooler',8871,'outliers');x=data['x']
 def reuse_fn():return mod.b2p_tt_v10_m32x__run(x,plan)
 g=graph(reuse_fn);base=graph(lambda:mod.kernel_fn(data,st))
 for seed in (7217,29117):
  x.copy_(reference.generate_input(32,'pooler',seed,'outliers')['x']);g[0].replay();ok,msg=reference.check_implementation(data,g[3],weights)
  if not ok:raise RuntimeError('reuse '+msg)
 ratio,a,b=paired(base,g,100)
 info=mod.b2p_tt_v10_m32x_m32x_launch_raw(ext,plan['idx'],plan,x,g[3],4096,plan['cfg'],2)
 result['reuse']={'layer':'pooler','m':32,'ratio':ratio,'baseline_us':a,'candidate_us':b,'resource_query':info,'global_weight_bytes_changed':False}
 import subprocess,re
 try:
  sass=subprocess.check_output(['cuobjdump','--dump-sass',ext.__file__],text=True)
  lines=[l.strip() for l in sass.splitlines() if 'HMMA' in l]
  result['reuse']['hmma_excerpt']=lines[:20]
  result['reuse']['static_hmma_count']=len(lines)
  result['reuse']['hmma_reuse_flags']=sum(l.count('.reuse') for l in lines)
 except Exception as e:result['reuse']['sass_error']=str(e)
 print('REUSE',json.dumps(result['reuse']),flush=True)
 # Hardware-counter retry: never try to change clock settings in a shared cloud GPU.
 import shutil,sys
 ncu=shutil.which('ncu')
 if ncu:
  code="import sys;sys.path.insert(0,'/work');import torch,reference;from benchmark import load;from pathlib import Path;m=load(Path('/work/candidate.py'),'ncu_parent');s=m.prepare(reference.generate_weights(82731));d=reference.generate_input(1,'pooler',9137);torch.cuda.synchronize();torch.cuda.profiler.start();m.kernel_fn(d,s);torch.cuda.synchronize();torch.cuda.profiler.stop()"
  try:
   r=subprocess.run([ncu,'--clock-control','none','--profile-from-start','off','--section','SpeedOfLight','--section','WarpStateStats','--csv',sys.executable,'-c',code],capture_output=True,text=True,timeout=180)
   result['ncu_returncode']=r.returncode;result['ncu_output']=(r.stdout+r.stderr)[-40000:]
  except subprocess.TimeoutExpired:result['ncu_output']='Counter retry timed out; no stall-counter claims.'
 return result
