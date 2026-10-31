"""Static-only source inventory; never imports the downloaded module."""
import ast,base64,hashlib,json,re,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];src=R/'research/2026-10-30/nathen-source.py.txt';s=src.read_text();tree=ast.parse(s)
out=R/'research/2026-10-30/audit';out.mkdir(exist_ok=True)
strings={};imports=[]
for n in ast.walk(tree):
 if isinstance(n,(ast.Import,ast.ImportFrom)):imports.append(ast.unparse(n))
 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
  name=n.targets[0].id;v=n.value
  if isinstance(v,ast.Constant) and isinstance(v.value,str) and ('#include' in v.value or '__global__' in v.value):strings[name]=v.value
  if isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='_zsrc':strings[name]=zlib.decompress(base64.b64decode(ast.literal_eval(v.args[0]))).decode()
for name,val in strings.items():(out/(name+'.txt')).write_text(val)
# Inspection copy hides source literals but retains all Python control flow.
class Strip(ast.NodeTransformer):
 def visit_Assign(self,n):
  if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in strings:
   n.value=ast.Constant(value='<source in audit/'+n.targets[0].id+'.txt>')
  return n
(out/'python_control_flow.txt').write_text(ast.unparse(Strip().visit(tree)))
patterns=r'\b(system|popen|fork|execve|socket|connect|fopen|fstream|getenv|setenv|clock64|clock_gettime|cudaEventElapsedTime|cudaGraph|cuGetProcAddress)\b|/proc/|getframe|inspect\.|subprocess|requests\.|urllib|reference|benchmark|autotun|elapsed_time|perf_counter'
hits=[]
for name,val in [('python',s)]+list(strings.items()):
 for i,line in enumerate(val.splitlines(),1):
  if len(line)<3000 and re.search(patterns,line,re.I):hits.append({'source':name,'line':i,'text':line[:260]})
j={'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'imports':sorted(set(imports)),'embedded_sources':{k:{'bytes':len(v),'sha256':hashlib.sha256(v.encode()).hexdigest()} for k,v in strings.items()},'review_hits':hits}
(out/'inventory.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({'imports':j['imports'],'sources':len(strings),'hits':hits},indent=2))
