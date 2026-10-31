"""Combine measured library choices; preserve champion on unproven shapes."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'candidates/lt/lt_original_h0.py').read_text()
source=source.replace("_LT_BATCHES = (4, 8, 16, 32, 64, 128, 256, 512, 1024)","_LT_BATCHES = (128, 256)")
source=source.replace("_LT_TARGETS = ('qkv_gate', 'pooler', 'unpooler')", "_LT_TARGETS = ('qkv_gate', 'pooler', 'unpooler')\n_LT_SELECTED = {('qkv_gate',128): 2, ('qkv_gate',256): 2, ('pooler',128): 0, ('unpooler',256): 0}")
source=source.replace("            for m in _LT_BATCHES:\n", "            for m in _LT_BATCHES:\n                if (name, m) not in _LT_SELECTED:\n                    continue\n")
source=source.replace('_LT_HEURISTIC_RANK, _LT_WORKSPACE_BUDGET)', '_LT_SELECTED[(name,m)], _LT_WORKSPACE_BUDGET)')
path=ROOT/'candidates/lt/lt_hybrid_v1.py'
ast.parse(source);path.write_text(source)
p=ROOT/'candidates/lt/manifest.json';manifest=json.loads(p.read_text());manifest=[x for x in manifest if x['file']!=str(path.relative_to(ROOT))]
manifest.append({'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'gpu_validated':False,'hypothesis':'Retain champion except four promising native paths; requires independent confirmation'})
p.write_text(json.dumps(manifest,indent=2)+'\n')
