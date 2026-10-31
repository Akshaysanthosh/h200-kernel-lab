"""QKV output-dimension padding: exact weights and sliced output, lab-only."""
import hashlib,json,statistics,time,subprocess,shutil,sys,gc
from pathlib import Path
import numpy as np
from benchmark import load
from test_public import read_flush
from padded_layout import source


def run(candidate,root='/work'):
 import torch
 from torch.utils.cpp_extension import load_inline
 sys.path.insert(0,root);import reference
 mod=load(Path(root)/'candidate.py','padding_parent')
 src=source(mod)
 ext=load_inline('okc_pad_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp_sources=src,with_cuda=True,extra_cflags=['-O3','-std=c++20'],extra_ldflags=['-l:libcublasLt.so.13'],verbose=False)
 workspace=torch.empty(128<<20,device='cuda',dtype=torch.uint8);ctx=ext.Context(workspace)
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
  rs=np.array([(aa[i]+aa[i+1])/(bb[i]+bb[i+1]) for i in range(0,reps,2)])
  boot=np.median(np.random.default_rng(17).choice(rs,(1000,len(rs))),axis=1)
  return {'ratio':float(np.median(rs)),'ci95':np.quantile(boot,[.025,.975]).tolist(),'parent_us':statistics.median(aa),'candidate_us':statistics.median(bb),'samples_us':[aa,bb]}
 selected=json.loads((Path(root)/'qkv_retest_configs.json').read_text())
 result={'passed':True,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'lt_version':ext.version(),'diagnostics':[],'retests':[]}
 for seed in (82731,294817):
  weights=reference.generate_weights(seed);st=mod.prepare(weights)
  print('CONTROL_PREPARED',seed,flush=True)
  lt=st['new_lt'];result.setdefault('lt_plan_metadata',[]).append({str(k):p.metadata()[idx] for k,(p,idx) in lt['configs'].items()})
  for m in (128,256):
   data=reference.generate_input(m,'qkv_gate',9137+m,'outliers');x=data['x'];w=weights['qkv_gate']
   parent=graph(lambda:mod.kernel_fn(data,st));out=parent[3]
   def trace(g):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
     g[0].replay();torch.cuda.synchronize()
    return [{'name':e.name,'us':e.device_time_total} for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA and 'read_flush' not in e.name]
   def checked(fn,label,reps=100,baseline=parent):
    g=graph(fn);g[0].replay();ok,msg=reference.check_implementation(data,g[3],weights)
    if not ok:raise RuntimeError(label+': '+msg)
    rr={'seed':seed,'m':m,'label':label,**paired(baseline,g,reps)}
    if seed==82731:rr['trace']=trace(g)
    return rr,g
   aa,_=checked(lambda:mod.kernel_fn(data,st),'full_parent_AA')
   result['diagnostics'].append(aa)
   if m==128:
    routeidx=st['v14_qkv']['routes'][m]
    actual=lambda:mod.tt_v14_qkv_run_qbig_candidate(x,st['v14_qkv'],routeidx)
   else:actual=lambda:mod.tt_lt_run_cublaslt(x,'qkv_gate',2304,lt)
   rr,_=checked(actual,'actual_route_control');result['diagnostics'].append(rr)
   plan,idx=lt['configs'][m,'qkv_gate']
   def direct(plan=plan,idx=idx,out=out):
    rc=plan.run(x,out,idx)
    if rc:raise RuntimeError(str(rc))
    return out
   rr,ltgraph=checked(direct,'saved_lt_same_output');result['diagnostics'].append(rr)
   # Compare modified extension to the parent's Lt using identical weights,
   # output and workspace. This isolates wrapper/descriptor changes.
   sharedctx=ext.Context(lt['workspace']);layout,epi,desc=mod.tt_lt__LT_H200_FIXED[m,'qkv_gate']
   mirror=ext.Plan(sharedctx,w,m,2304,8192,layout,epi);mi=mirror.restore(desc);assert mi>=0
   def mirrored():
    rc=mirror.run(x,out,mi)
    if rc:raise RuntimeError(str(rc))
    return out
   rr,_=checked(mirrored,'extension_same_pointers_vs_saved_lt',baseline=ltgraph);result['diagnostics'].append(rr)
   print('CONTROL_DIAGNOSTICS',seed,m,json.dumps([{k:v for k,v in z.items() if k not in ('samples_us','trace')} for z in result['diagnostics'][-4:]]),flush=True)
   for choice in selected[str(m)]:
    npad=choice['npad'];layout=choice['layout'];desc=choice['config']
    dense=torch.zeros((npad,8192),device=x.device,dtype=torch.bfloat16);dense[:2304].copy_(w)
    assert torch.equal(dense[:2304].view(torch.int16),w.view(torch.int16))
    if layout>=2:dense=dense.t().contiguous()
    trialout=torch.empty((m,npad),device=x.device,dtype=torch.bfloat16)
    tp=ext.Plan(sharedctx,dense,m,npad,8192,layout,0);ti=tp.restore(desc);assert ti>=0
    def fn():
     rc=tp.run(x,trialout,ti)
     if rc:raise RuntimeError(str(rc))
     return trialout[:,:2304]
    rr,g=checked(fn,'geometry',reps=120);rr.update(choice)
    x.copy_(reference.generate_input(m,'qkv_gate',seed+m,'normal')['x']);g[0].replay()
    ok,msg=reference.check_implementation(data,g[3],weights)
    if not ok:raise RuntimeError(msg)
    rr['fresh_input_check']=msg;result['retests'].append(rr)
    print('CONTROL_RETEST',json.dumps({k:v for k,v in rr.items() if k not in ('samples_us','trace')}),flush=True)
   # Existing, already-compiled native QKV alternatives, no new broad sweep.
   for ni in ((1,) if m==128 else (2,)):
    rr,g=checked(lambda:mod.tt_v14_qkv_run_qbig_candidate(x,st['v14_qkv'],ni),'native_alternative',reps=120)
    rr['native_config']=st['v14_qkv']['configs'][ni];rr['native_index']=ni
    x.copy_(reference.generate_input(m,'qkv_gate',seed+m+1,'outliers')['x']);g[0].replay()
    ok,msg=reference.check_implementation(data,g[3],weights)
    if not ok:raise RuntimeError(msg)
    rr['fresh_input_check']=msg;result['retests'].append(rr)
    print('CONTROL_NATIVE',json.dumps({k:v for k,v in rr.items() if k not in ('samples_us','trace')}),flush=True)
 return result
