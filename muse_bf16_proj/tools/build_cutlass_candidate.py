import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'candidates/hopper/hybrid_v1.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='7ba00d91fae130f4cf6fd6bb3efc93dc4b3f98eb9ec5a5469bdc201af21aa03f'
s=base+'\n_CUTLASS_CUDA='+repr((ROOT/'cutlass_lab/pooler.cu').read_text())+'''

def prepare_cutlass(weights):
    state=prepare(weights)
    state['_cutlass']=load_inline(name='okc_cutlass_pooler_v3a',cpp_sources='torch::Tensor run(torch::Tensor,torch::Tensor,int,int);',cuda_sources=_CUTLASS_CUDA,functions=['run'],extra_include_paths=['/opt/cutlass/include'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],with_cuda=True,verbose=False)
    return state

def _cutlass_run(data,state,config,split):
    partial=state['_cutlass'].run(data['x'],state['pooler'],config,split)
    out=torch.empty((16,4096),device=data['x'].device,dtype=torch.bfloat16)
    _reduce[(triton.cdiv(16*4096,256),)](partial,out,16*4096,split,256)
    return out
'''
ast.parse(s);p=ROOT/'candidates/hopper/cutlass_v3a.py';p.write_text(s)
f=ROOT/'candidates/lt/manifest.json';j=json.loads(f.read_text());j=[i for i in j if i['file']!=str(p.relative_to(ROOT))];j.append({'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'gpu_validated':False,'development_only':True,'hypothesis':'CUTLASS SM90a cp.async/WGMMA BF16 pooler with batch-partitioned FP32 split-K'});f.write_text(json.dumps(j,indent=2)+'\n')
