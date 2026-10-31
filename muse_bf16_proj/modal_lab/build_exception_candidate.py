"""Build a separate, provisional O8 predicated-decoder candidate."""
import ast,base64,zlib,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from exception_patch import source
ROOT=Path(__file__).resolve().parents[1]
parent=(ROOT/'candidates/public/nathen_v18_linear.py').read_text()
for node in ast.parse(parent).body:
 if isinstance(node,ast.Assign) and getattr(node.targets[0],'id','')=='V18S_CUDA_TEMPLATE':
  template=zlib.decompress(base64.b64decode(ast.literal_eval(node.value.args[0]))).decode()
inst=(1,1,1,2,0,0,256,0,8192,1024,1,8,2,1,6)
args=', '.join(map(str,inst))
case=f'case 0: return v18s_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);'
src=source(SimpleNamespace(v18s_cuda_source=lambda _:template.replace('@@CASES@@',case)),[inst])
extra='''
# Experimental predicated O8 decoder, derived from this file's v18d decoder.
# Original lossless packing is unchanged. Protected parent remains unchanged.
_DEC_V1_CUDA = SOURCE_LITERAL
_dec_v1_parent_prepare=prepare
_dec_v1_parent_kernel=kernel_fn

def prepare(weights: Weights) -> dict:
    st=_dec_v1_parent_prepare(weights)
    ext=load_inline('okc_patch_o8_'+hashlib.sha256(_DEC_V1_CUDA.encode()).hexdigest()[:12],V18S_CPP,cuda_sources=_DEC_V1_CUDA,functions=['v18s_launch','v18s_router_launch'],extra_cuda_cflags=['-O3','-lineinfo'],no_implicit_headers=True,with_pytorch_error_handling=False)
    pk=v18s_pack_format(weights['o'],1)
    sp=None if pk is None else v18s_stream(pk,1,1,8,2,1)
    if sp is not None:
        if not v18s_verify_onehot(ext,0,sp,weights['o'],(1,8,2,1),8):
            raise RuntimeError('Predicated O8 exact weight check failed')
        if not v18s_verify_random(ext,0,sp,weights['o'],(1,8,2,1),[1,7,8]):
            raise RuntimeError('Predicated O8 random check failed')
        st['dec_v1']=(ext,sp)
    return st

def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x=data['x'];c=data['config']
    if c['layer']=='o' and x.shape==(8,1024) and c['n']==8192 and x.is_cuda and x.is_contiguous() and x.dtype==torch.bfloat16 and x.data_ptr()%16==0 and 'dec_v1' in state:
        ext,sp=state['dec_v1']
        if x.device==sp['wp'].device:
            out=torch.empty((8,8192),device=x.device,dtype=torch.bfloat16)
            rc=v18s_launch_raw(ext,0,sp,x,out,8192,(1,8,2,1))
            if rc:raise RuntimeError('Predicated O8 launch failed: '+str(rc))
            return out
    return _dec_v1_parent_kernel(data,state)
'''.replace('SOURCE_LITERAL',repr(src))
p=ROOT/'candidates/public/v18_predicated_o8.py';p.write_text(parent+extra)
m=ROOT/'candidates/lt/manifest.json';d=json.loads(m.read_text());rel=str(p.relative_to(ROOT));d=[x for x in d if x['file']!=rel];d.append({'file':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'description':'Provisional predicated O8, independent validation required; no promotion'})
m.write_text(json.dumps(d,indent=2)+'\n');print(rel)
