"""Bounded correctness diagnostic, no performance claims."""
import hashlib,json
from pathlib import Path
from benchmark import load
from pipeline_source import source

def run(candidate,root='/work'):
 import torch,sys
 from torch.utils.cpp_extension import load_inline
 sys.path.insert(0,root);import reference
 mod=load(Path(root)/'candidate.py','pipe_debug')
 inst=[(64,64,1,1,2,True,1)]
 srcs={'serial':source(mod,inst,False),'pipeline':source(mod,inst,True)}
 exts={}
 for name,src in srcs.items():
  exts[name]=load_inline('okc_dbg_'+hashlib.sha256(src.encode()).hexdigest()[:12],mod.dd_decodediet_tt_mw_CPP,cuda_sources=src,functions=['launch','info'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','--keep','-gencode=arch=compute_90a,code=sm_90a'])
 result={'passed':False,'diagnostic_only':True,'tests':[]}
 torch.manual_seed(18372)
 for k in (64,128,256,512,4096):
  w=torch.randn((8192,k),device='cuda',dtype=torch.bfloat16);x=torch.randn((128,k),device='cuda',dtype=torch.bfloat16);x[0].zero_();x[0,3]=16
  pk=mod.dd_decodediet_tt_mw_pack_weight(w,64,1,1,True)
  if pk is None:raise RuntimeError('Packing failure')
  ref=x.double()@w.double().T
  for name,ext in exts.items():
   out=torch.empty((128,8192),device='cuda',dtype=torch.bfloat16)
   rc=mod.dd_decodediet_tt_mw_launch_raw(ext,0,pk,x,out);torch.cuda.synchronize()
   finite=bool(torch.isfinite(out).all());err=float((out.double()-ref).norm()/ref.norm()) if finite else None
   row={'k':k,'name':name,'rc':rc,'finite':finite,'error':err,'first_values':out[0,:8].float().tolist()};result['tests'].append(row);print('DEBUG',json.dumps(row),flush=True)
 # Retain PTX for the small failing specialization for local diagnosis.
 folder=Path(exts['pipeline'].__file__).parent
 result['ptx_files']={p.name:p.read_text() for p in folder.glob('*.ptx')}
 result['passed']=all(t['finite'] and t['error']<.01 for t in result['tests'])
 return result
