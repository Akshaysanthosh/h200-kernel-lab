"""Compare aligned-prefix OR decoding to the protected lookup decoder."""
import hashlib,json,statistics,subprocess,re
from pathlib import Path
from benchmark import load
from test_public import read_flush
from vector_decoder import source,pack,stream,audit
import numpy as np


def run(candidate,root='/work'):
 import torch,sys
 from torch.utils.cpp_extension import load_inline
 sys.path.insert(0,root);import reference
 mod=load(Path(root)/'candidate.py','vector_parent')
 targets=[('pooler',1,*mod.V18_SMALL['pooler',1]),('o',8,*mod.V18_SMALL['o',8])]
 bases=[t[2] for t in targets];instances=[];options=[]
 for layer,m,base,cfg in targets:
  configs=[cfg] if layer=='o' else [cfg,(1,16,8,1),(2,8,16,1)]
  opts=[]
  for cc in configs:
   for stage in (1,2):
    inst=list(base);inst[3]=stage;inst[6]=cc[0]*cc[1]*32;inst[10:14]=cc
    inst=tuple(inst)
    if inst not in instances:instances.append(inst)
    opts.append((instances.index(inst),cc,stage))
  options.append(opts)
 src=source(mod,instances)
 ext=load_inline('okc_vec_'+hashlib.sha256(src.encode()).hexdigest()[:12],mod.V18S_CPP,cuda_sources=src,functions=['v18s_launch','v18s_router_launch'],extra_cuda_cflags=['-O3','-lineinfo'],no_implicit_headers=True,with_pytorch_error_handling=False)
 parent=mod.v18s_build(bases,name='vec_control_'+hashlib.sha256(str(bases).encode()).hexdigest()[:12])
 print('BUILT',len(instances),flush=True)
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'audit':audit(),'rounds':[],'sass':{},'skipped':[]}
 for name,module in [('parent',parent),('aligned_or',ext)]:
  try:
   sass=subprocess.check_output(['cuobjdump','--dump-sass',module.__file__],text=True)
   funcs=[]
   for section in sass.split('Function : ')[1:]:
    if 'v18s_gemv' not in section.splitlines()[0]:continue
    instructions=[l for l in section.splitlines() if re.search(r'/\*[0-9a-f]+\*/',l)]
    funcs.append({'function':section.splitlines()[0],'LDG128':sum('LDG' in l and '.128' in l for l in instructions),'PRMT':sum('PRMT' in l for l in instructions),'LOP3':sum('LOP3' in l for l in instructions),'IADD3':sum('IADD3' in l for l in instructions),'SHF':sum('SHF' in l for l in instructions)})
   result['sass'][name]=funcs
  except Exception as e:result['sass'][name]={'error':str(e)}
 flush=torch.ones(128*1024*1024,device='cuda',dtype=torch.uint8);sink=torch.empty(flush.numel()//4096,device='cuda',dtype=torch.int32)
 def clear():read_flush[(sink.numel(),)](flush,sink,flush.numel(),4096)
 clear();torch.cuda.synchronize()
 def graph(fn):
  for _ in range(3):fn()
  torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):clear();a.record();out=fn();b.record()
  return g,a,b,out
 def us(g):g[0].replay();g[2].synchronize();return g[1].elapsed_time(g[2])*1000
 def paired(a,b):
  aa=[];bb=[]
  for i in range(80):
   if i%2:bb.append(us(b));aa.append(us(a))
   else:aa.append(us(a));bb.append(us(b))
  rs=np.array([(aa[i]+aa[i+1])/(bb[i]+bb[i+1]) for i in range(0,80,2)])
  boot=np.median(np.random.default_rng(17).choice(rs,(2000,len(rs))),axis=1)
  return {'ratio':float(np.median(rs)),'ci95':np.quantile(boot,[.025,.975]).tolist(),'parent_us':statistics.median(aa),'candidate_us':statistics.median(bb),'samples_us':[aa,bb]}
 for seed in (82731,294817):
  weights=reference.generate_weights(seed)
  for target_id,(layer,m,base,cfg) in enumerate(targets):
   w=weights[layer];n=w.shape[0]
   pk=mod.v18s_pack_format(w,1);parent_sp=mod.v18s_stream(pk,1,*cfg)
   apk=pack(mod,w);data=reference.generate_input(m,layer,seed+m);x=data['x'];out0=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
   def baseline():
    rc=mod.v18s_launch_raw(parent,target_id,parent_sp,x,out0,n,cfg)
    if rc:raise RuntimeError(str(rc))
    return out0
   gg=graph(baseline);control=graph(baseline)
   rr={'seed':seed,'layer':layer,'m':m,'aa':paired(gg,control),'base':apk['base'],'exceptions':apk['n_esc'],'baseline_exceptions':pk.get('n_esc'),'candidates':[]}
   streams={}
   for idx,cc,stage in options[target_id]:
    if cc not in streams:streams[cc]=stream(mod,apk,cc)
    sp=streams[cc]
    if sp is None:
     result['skipped'].append({'seed':seed,'layer':layer,'cfg':cc,'reason':'exception capacity exceeded; no timing'});continue
    out=torch.empty_like(out0)
    info=mod.v18s_launch_raw(ext,idx,sp,x,out,n,cc,2);occ=mod.v18s_launch_raw(ext,idx,sp,x,out,n,cc,1)
    if info<0 or occ<=0:
     result['skipped'].append({'seed':seed,'layer':layer,'cfg':cc,'reason':'unsupported resource use'});continue
    checks=[]
    for mm in (1,7,8):
     for dist in ('normal','outliers'):
      dd=reference.generate_input(mm,layer,seed+11+mm,dist);yy=torch.empty((mm,n),device='cuda',dtype=torch.bfloat16)
      rc=mod.v18s_launch_raw(ext,idx,sp,dd['x'],yy,n,cc)
      if rc:raise RuntimeError('launch '+str(rc))
      ok,msg=reference.check_implementation(dd,yy,weights)
      if not ok:raise RuntimeError(str((layer,cc,mm,dist,msg)))
      checks.append({'m':mm,'dist':dist,'message':msg})
    def fn():
     rc=mod.v18s_launch_raw(ext,idx,sp,x,out,n,cc)
     if rc:raise RuntimeError('launch '+str(rc))
     return out
    g=graph(fn);x.copy_(reference.generate_input(m,layer,seed+123,'outliers')['x'])
    for graph_ in (gg,g):
     graph_[0].replay();ok,msg=reference.check_implementation(data,graph_[3],weights)
     if not ok:raise RuntimeError('fresh graph '+msg)
    row={'cfg':cc,'stage':stage,'max_exceptions':sp['max_esc'],'registers':info//1000,'local_bytes':info%1000,'occupancy':occ,'bytes':sum(sp[k].numel()*sp[k].element_size() for k in ('wp','esc')),'correctness':checks,**paired(gg,g)}
    rr['candidates'].append(row)
    print('VECTOR_RESULT',seed,layer,json.dumps({k:v for k,v in row.items() if k not in ('samples_us','correctness')}),flush=True)
   result['rounds'].append(rr)
 return result
