import json,sys
from pathlib import Path
for f in sys.argv[1:]:
 d=json.loads(Path(f).read_text()); r=d.get('result') or {}
 print(f, 'job',d.get('id'),'passed',r.get('passed'),'speedup',r.get('speedup'),'ci',r.get('speedup_ci'),'error',r.get('error'),'flags',r.get('flags'))
 for rec in r.get('records',[]):
  if rec.get('status') in ('fail','error'): print(rec)
 if not r.get('passed'): print(r.get('log_tail','')[-5000:])
