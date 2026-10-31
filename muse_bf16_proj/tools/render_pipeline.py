import ast,types,sys,difflib,base64,zlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=ast.parse((root/'candidates/public/nathen_v18_linear.py').read_text())
names={'dd_decodediet_tt_mw_mma','dd_decodediet_tt_mw_cuda_source','dd_decodediet_tt_mw_CUDA_PREFIX','dd_decodediet_tt_mw_CUDA_CORE','dd_decodediet_tt_mw_CUDA_MAP','dd_decodediet_tt_mw_CUDA_HOST'}
nodes=[]
for n in s.body:
 if isinstance(n,ast.FunctionDef) and n.name in names:nodes.append(n)
 elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets):nodes.append(n)
g={'_zsrc':lambda v:zlib.decompress(base64.b64decode(v)).decode()};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audited-source-generator>','exec'),g)
sys.path.insert(0,str(root/'modal_lab'));from pipeline_source import source
mod=types.SimpleNamespace(**g);inst=[(bn,64,wk,pf,2,True,1) for bn in (64,128,256) for wk in (1,2) for pf in (1,2)]
a=source(mod,inst,False);b=source(mod,inst,True)
(root/'native_pipeline/pipeline_v3.cu').write_text(b)
(root/'native_pipeline/pipeline_v3.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='serial_wide.cu',tofile='pipeline_v3.cu')))
print('Saved CUDA and reviewable diff; bytes',len(b))
