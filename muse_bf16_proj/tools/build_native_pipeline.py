import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'candidates/hopper/hybrid_v1.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='7ba00d91fae130f4cf6fd6bb3efc93dc4b3f98eb9ec5a5469bdc201af21aa03f'
# Keep the existing dispatch as a control; expose prototype only to the external lab.
pack=(ROOT/'kernels/packed_projection.py').read_text().split('def _pack_weights')[1].split('def prepare')[0]
source=base+'\ndef _pack_weights'+pack+'\n_NATIVE_CUDA='+repr((ROOT/'native_pipeline/packed_pooler.cu').read_text())+'''

def prepare_native(weights):
    state=prepare(weights)
    state['pooler_packed']=_pack_weights(weights['pooler'])
    state['_native']=load_inline(name='okc_packed_pipeline_v3',cpp_sources='torch::Tensor run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int,int,bool);',cuda_sources=_NATIVE_CUDA,functions=['run'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],with_cuda=True,verbose=False)
    return state

def _native_run(data,state,cfg,split,packed=True):
    sm,ep,base=state['pooler_packed']
    partial=state['_native'].run(data['x'],state['pooler'],sm,ep,base,cfg,split,packed)
    out=torch.empty((16,4096),device=data['x'].device,dtype=torch.bfloat16)
    _reduce[(triton.cdiv(16*4096,256),)](partial,out,16*4096,split,256)
    return out
'''
ast.parse(source)
p=ROOT/'candidates/hopper/native_pipeline_v3.py';p.write_text(source)
f=ROOT/'candidates/lt/manifest.json';j=json.loads(f.read_text());j=[i for i in j if i['file']!=str(p.relative_to(ROOT))];j.append({'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'gpu_validated':False,'development_only':True,'hypothesis':'Native warp-specialized lossless decode overlapped with tensor-core consumption, pooler m16'});f.write_text(json.dumps(j,indent=2)+'\n')
