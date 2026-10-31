import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'candidates/hopper/hybrid_v1.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='7ba00d91fae130f4cf6fd6bb3efc93dc4b3f98eb9ec5a5469bdc201af21aa03f'
codec=(ROOT/'kernels/packed_projection.py').read_text(); start=codec.index('def _pack_weights'); end=codec.index('\ndef ',start+5) if '\ndef ' in codec[start+5:] else len(codec)
base+='\n'+codec[start:end]
s=base+'\n_CUTLASS_CUDA='+repr((ROOT/'cutlass_decode/pooler.cu').read_text())+'''

def prepare_cutlass(weights):
    state=prepare(weights)
    state['_cutlass']=load_inline(name='okc_cutlass_pooler_decode_v2',cpp_sources='torch::Tensor run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int,int,bool);',cuda_sources=_CUTLASS_CUDA,functions=['run'],extra_include_paths=['/opt/cutlass_decode/include','/opt/cutlass/include'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],with_cuda=True,verbose=False)
    state['_packed']=_pack_weights(weights['pooler'])
    return state

def _cutlass_run(data,state,config,split,packed=True):
    partial=state['_cutlass'].run(data['x'],state['pooler'],*state['_packed'],config,split,packed)
    out=torch.empty((16,4096),device=data['x'].device,dtype=torch.bfloat16)
    _reduce[(triton.cdiv(16*4096,256),)](partial,out,16*4096,split,256)
    return out
'''
ast.parse(s);p=ROOT/'candidates/hopper/cutlass_decode_v2.py';p.write_text(s)
f=ROOT/'candidates/lt/manifest.json';j=json.loads(f.read_text());j=[i for i in j if i['file']!=str(p.relative_to(ROOT))];j.append({'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'gpu_validated':False,'development_only':True,'hypothesis':'CUTLASS SM90a cp.async/WGMMA BF16 pooler with batch-partitioned FP32 split-K'});f.write_text(json.dumps(j,indent=2)+'\n')
