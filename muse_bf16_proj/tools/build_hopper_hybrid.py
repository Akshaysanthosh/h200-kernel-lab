"""Combine offline-selected Hopper tiles with the preserved cuBLASLt candidate."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'candidates/lt/lt_hybrid_v2.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='21c8c63fb2af0390885050dd88f3955a09c852bb74d427b9fe2a78d02a18bd8f'
base=base.replace('def kernel_fn(', 'def _library_kernel_fn(',1)
kernel=(ROOT/'kernels/hopper_tma_v2.py').read_text().split('# BM, BN, BK')[0]
host=(ROOT/'kernels/hopper_tma_v2.py').read_text().split('def _hopper_run')[1].split('def kernel_fn')[0]
source=base+'\n'+kernel+'\ndef _hopper_run'+host+'''
# Fixed configurations selected externally; only metadata controls dispatch.
def kernel_fn(data,state):
    c=data['config']; x=data['x']; m=x.shape[0]
    if x.is_contiguous() and x.data_ptr()%16==0:
        if c['layer']=='qkv_gate' and 8<=m<=32:
            config=(16,64,128,4,4,4,True) if m<=16 else (32,64,128,4,4,4,True)
            return _hopper_run(data,state,config)
        if c['layer']=='pooler' and 8<=m<=16:
            return _hopper_run(data,state,(16,64,128,4,4,4,True))
    return _library_kernel_fn(data,state)
'''
ast.parse(source)
p=ROOT/'candidates/hopper/hybrid_v1.py';p.write_text(source)
f=ROOT/'candidates/lt/manifest.json';items=json.loads(f.read_text());items=[i for i in items if i['file']!=str(p.relative_to(ROOT))]
items.append({'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'gpu_validated':False,'hypothesis':'Small-batch TMA/WGMMA qkv and pooler; cuBLASLt hybrid v2 for larger shapes'})
f.write_text(json.dumps(items,indent=2)+'\n')
