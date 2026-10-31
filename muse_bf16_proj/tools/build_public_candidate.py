from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parents[1];src=R/'research/2026-10-30/nathen-source.py.txt';s=src.read_text()
license=(R/'research/2026-10-30/nathen-LICENSE').read_text()
header='# Derived from NathenFernandes/muse-bf16-proj, public v18.\n# Only adaptation: explicit PyTorch linear fallback for this standalone H200 lab.\n'+''.join('# '+l+'\n' for l in license.splitlines())
s=s.replace('# okc: problem=muse_bf16_proj\n','# okc: problem=muse_bf16_proj\n'+header,1)
s=s.replace('from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl','def _bf16_gemm_dispatch_impl(x, weight, bias):\n    return F.linear(x, weight, bias)',1)
ast.parse(s);p=R/'candidates/public/nathen_v18_linear.py';p.parent.mkdir(exist_ok=True);p.write_text(s)
f=R/'candidates/lt/manifest.json';j=json.loads(f.read_text());j=[e for e in j if e['file']!=str(p.relative_to(R))];j.append({'file':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'gpu_validated':False,'development_only':True,'hypothesis':'MIT public v18 with explicit linear fallback; requires independent validation and official confirmation','upstream_sha256':hashlib.sha256(src.read_bytes()).hexdigest()});f.write_text(json.dumps(j,indent=2)+'\n')
