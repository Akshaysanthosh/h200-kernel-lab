"""Extract only SK2 and its dependencies; preserve upstream source and provenance."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=(ROOT/'research/2026-10-31/vjeux/muse_bf16_proj.py').read_text();tree=ast.parse(src)
index={}
for node in tree.body:
 if isinstance(node,(ast.FunctionDef,ast.ClassDef)):index[node.name]=node
 elif isinstance(node,ast.Assign):
  for t in node.targets:
   if isinstance(t,ast.Name):index[t.id]=node
custom='''
_ext = None
def _get_ext():
 global _ext
 if _ext is None:
  cuda_src=CUDA_SRC.replace('@@MBAR@@',MBAR_HELPERS).replace('@@BODY@@',WG_HELPERS)+MAP_SRC+_sk2_src(True)
  cpp='#include <torch/extension.h>\\nvoid sk2(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,torch::Tensor);'
  _ext=load_inline(name='okc_sk2_'+hashlib.sha256(cuda_src.encode()).hexdigest()[:12],cpp_sources=cpp,cuda_sources=cuda_src,functions=['sk2'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],no_implicit_headers=True,verbose=False)
 return _ext
'''
for n in ast.parse(custom).body:
 index[n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id]=n
routes={('qkv_gate',32),('pooler',32),('unpooler',32)}
picks=ast.literal_eval(index['SK2_PICK'].value);picks={k:v for k,v in picks.items() if k in routes}
index['SK2_PICK'].value=ast.parse(repr(picks),mode='eval').body
# Only retained names are needed; avoid evaluating UNPOOLER_SLICE_ROWS in discarded config.
index['SK2_CONFIGS'].value.elts=[n for n in index['SK2_CONFIGS'].value.elts if ast.literal_eval(n.elts[0]) in picks.values()]
needed=set();todo=['sk2_prepare','_run_sk2']
while todo:
 name=todo.pop()
 if name in needed or name not in index:continue
 needed.add(name)
 todo.extend(n.id for n in ast.walk(index[name]) if isinstance(n,ast.Name) and n.id in index)
class Prefix(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id in needed:n.id='vj_'+n.id
  return n
 def visit_FunctionDef(self,n):
  self.generic_visit(n);n.name='vj_'+n.name;return n
 def visit_Global(self,n):n.names=['vj_'+x if x in needed else x for x in n.names];return n
nodes=[];seen=set()
for n in list(tree.body)+ast.parse(custom).body:
 name=n.name if isinstance(n,(ast.FunctionDef,ast.ClassDef)) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
 if name in needed and name not in seen:
  nodes.append(Prefix().visit(index[name]));seen.add(name)
assert needed==seen,(needed-seen)
extract=ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])))
text=(ROOT/'candidates/public/nathen_v18_linear.py').read_text()
text+='\n\n# Additional SK2 implementation: vjeux/okc-2026, Apache-2.0.\n# Upstream f35a41ee1c13e316dabbf772830d47b23b453393; extracted SK2 only, prefixed Python names.\n# Adaptation: selected routes, dedicated extension build, parent fallback.\n'
text+='\n'.join('# '+x for x in (ROOT/'research/2026-10-31/vjeux/LICENSE').read_text().splitlines())+'\n'
text+='\n_vj_parent_prepare=prepare\n_vj_parent_kernel=kernel_fn\n'+extract+'\n'
text+='''
def prepare(weights):
 state=_vj_parent_prepare(weights)
 vj__get_ext()
 sk={'dummy':torch.zeros(2,dtype=torch.int32,device=weights['o'].device)}
 vj_sk2_prepare(weights,sk)
 state['vj_sk2']=sk
 return state

def kernel_fn(data,state):
 x=data['x'];c=data['config'];m=c['m'];layer=c['layer']
 if 17<=m<=32 and layer in ('qkv_gate','pooler','unpooler'):
  sk=state.get('vj_sk2',{})
  name=vj__sk2_name(layer,m)
  if name in sk.get('sk2',{}):return vj__run_sk2(x,layer,m,c['n'],c['k'],sk)
 return _vj_parent_kernel(data,state)
'''
path=ROOT/'candidates/public/v18_sk2_m32.py';ast.parse(text);path.write_text(text)
manifest=ROOT/'candidates/lt/manifest.json';rows=json.loads(manifest.read_text());rows=[r for r in rows if r['file']!='candidates/public/v18_sk2_m32.py'];rows.append({'file':'candidates/public/v18_sk2_m32.py','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'DEVELOPMENT_ONLY','description':'Public v18 plus attributed vjeux SK2 for three batch-32 routes; requires full confirmation.'});manifest.write_text(json.dumps(rows,indent=2)+'\n')
print('extracted',sorted(needed));print('candidate',rows[-1]['sha256'])
