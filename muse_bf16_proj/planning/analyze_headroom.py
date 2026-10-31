import json, math
from pathlib import Path
from collections import defaultdict
root=Path(__file__).resolve().parent.parent
p=json.loads((root/'profiles/submission.35d40255820f4df4.json').read_text())
b=p['benchmarks']; groups=defaultdict(list)
for x in b:groups[x['case']['load']].append(x)
actual=p['summary']['speedup']
def score(fn):
 ratios=[]
 for rows in groups.values():
  old=sum(x['median_ns']*x['case']['weight'] for x in rows)
  new=sum(fn(x)*x['case']['weight'] for x in rows)
  ratios.append(old/new)
 return actual*math.exp(sum(map(math.log,ratios))/len(ratios))
base=lambda x:x['median_ns']
floor=lambda x:x['speed_of_light_ns']
scenarios={
 'current':actual,
 'router_free':score(lambda x:0 if x['case']['layer']=='router' else base(x)),
 'router_at_reported_floor':score(lambda x:floor(x) if x['case']['layer']=='router' else base(x)),
 'dense_20pct_less_time':score(lambda x:base(x)*(1 if x['case']['layer']=='router' else .8)),
 'dense_25pct_less_time':score(lambda x:base(x)*(1 if x['case']['layer']=='router' else .75)),
 'dense_30pct_less_time':score(lambda x:base(x)*(1 if x['case']['layer']=='router' else .7)),
 'all_at_reported_floor':score(floor),
 'dense_at_reported_floor':score(lambda x:base(x) if x['case']['layer']=='router' else floor(x))}
bylayer={}
for layer in ('qkv_gate','o','pooler','unpooler','router'):
 shares=[sum(base(x) for x in rows if x['case']['layer']==layer)/sum(base(x) for x in rows) for rows in groups.values()]
 bylayer[layer]={'mean_load_time_share':sum(shares)/len(shares),'score_if_layer_20pct_faster_time':score(lambda x:base(x)*(.8 if x['case']['layer']==layer else 1))}
loads=[]
for name, rows in groups.items():
 loads.append({'load':name,'current_us_per_layer_sum':sum(base(x) for x in rows)/1000,'reported_floor_us_per_layer_sum':sum(floor(x) for x in rows)/1000,'speedup_vs_baseline':sum(x['baseline_ns'] for x in rows)/sum(base(x) for x in rows)})
board=json.loads((root/'results/leaderboard-planning.json').read_text()); tenth=next(x for x in board['rows'] if x['rank']==10)
result={'current_speedup':actual,'current_rank':board['you']['rank'],'top10_speedup':tenth['speedup'],'required_additional_speed_factor':tenth['speedup']/actual,'required_time_reduction':1-actual/tenth['speedup'],'scenarios':scenarios,'layers':bylayer,'loads':loads,'caveat':'Analytical estimates holding other costs fixed. Reported hardware floors are optimistic reference bounds, not attainable targets or guarantees. Scoring uses six equally weighted load ratios via geometric mean.'}
(root/'planning/headroom.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
