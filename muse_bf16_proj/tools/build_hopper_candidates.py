import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'champions/2026-10-30_1.227412/submission.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='9ea10c81639eb23d6c9eb832465190328664fd655e89e9ff81732b94e10b8a36'
base=base.replace('def kernel_fn(', 'def _champion_kernel_fn(',1)
source=base+'\n'+(ROOT/'kernels/hopper_tma.py').read_text()
ast.parse(source)
p=ROOT/'candidates/hopper/tma_v1.py';p.write_text(source)
f=ROOT/'candidates/lt/manifest.json';items=json.loads(f.read_text());items=[i for i in items if i['file']!=str(p.relative_to(ROOT))]
items.append({'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'gpu_validated':False,'hypothesis':'TMA loads and tensor-core pipelining with bounded FP32 split-K reduction'})
f.write_text(json.dumps(items,indent=2)+'\n')
