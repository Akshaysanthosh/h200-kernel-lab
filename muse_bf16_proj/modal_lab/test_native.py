"""External bounded native-pipeline experiment. Never part of submission."""
import json,statistics,sys,time,subprocess,re,collections
from pathlib import Path
from benchmark import load

def run(candidate,root='/work'):
    import torch
    root=Path(root);sys.path.insert(0,str(root));import reference
    champion=load(root/'champion.py','champion');custom=load(root/'candidate.py','custom')
    weights=reference.generate_weights(571913);state=custom.prepare_native(weights)
    flush=torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda')
    result={'candidate':candidate,'gpu':torch.cuda.get_device_name(),'passed':True,'official_score':False,'shape':{'layer':'pooler','m':16},'trials':[]}
    sm,ep,base=state['pooler_packed'];w=weights['pooler'];n,k=w.shape
    delta=torch.stack((ep.int()&15,ep.int()>>4),-1).reshape(n,k);bb=base.int().repeat_interleave(256,dim=1)
    restored=(sm.int()&127)|((sm.int()&128)<<8)|((bb+delta)<<7);original=w.view(torch.int16).int()&65535
    assert torch.equal(torch.where(bb==255,original,restored),original)
    result['bit_exact_packing']=True;result['fallback_fraction']=float((base==255).float().mean())
    try:
        result['resources']=subprocess.check_output(['cuobjdump','--dump-resource-usage',state['_native'].__file__],text=True)[-16000:]
        sass=subprocess.check_output(['cuobjdump','--dump-sass',state['_native'].__file__],text=True)
        result['sass_instruction_counts']=dict(collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)',sass)))
    except Exception as exc:result['resource_error']=str(exc)
    def capture(fn):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):fn()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True);g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):flush.zero_();a.record();out=fn();b.record()
        return g,a,b,out
    def measure(g):
        g[0].replay();g[2].synchronize();return g[1].elapsed_time(g[2])*1000
    data=reference.generate_input(16,'pooler',478331,'outliers')
    bases=[capture(lambda:champion.kernel_fn(data,state)),capture(lambda:custom.kernel_fn(data,state))]
    began=time.monotonic()
    for cfg in (2,3,4,5,6,7):
        for split in (4,8,16):
            for packed in (False,True):
                if time.monotonic()-began>600:result['truncated']=True;break
                data['x'].copy_(reference.generate_input(16,'pooler',478331,'outliers')['x'])
                record={'config':cfg,'split':split,'packed':packed}
                out=custom._native_run(data,state,cfg,split,packed)
                ok,msg=reference.check_implementation(data,out,weights)
                record.update(correctness=ok,message=msg)
                if not ok:result['trials'].append(record);continue
                saved=out.clone();fresh=reference.generate_input(16,'pooler',77831,'normal')
                new=custom._native_run(fresh,state,cfg,split,packed)
                ok,msg=reference.check_implementation(fresh,new,weights)
                if not ok or not torch.equal(saved,out):raise RuntimeError('Fresh/lifetime failure '+msg)
                trial=capture(lambda:custom._native_run(data,state,cfg,split,packed))
                data['x'].copy_(fresh['x']);trial[0].replay();torch.cuda.synchronize()
                ok,msg=reference.check_implementation(data,trial[3],weights)
                if not ok:raise RuntimeError('Graph failure '+msg)
                vals=[[],[],[]];ratios=[]
                for rep in range(32):
                    order=(0,1,2) if rep%2==0 else (2,1,0)
                    row=[0,0,0]
                    for i in order:row[i]=measure((bases+[trial])[i]);vals[i].append(row[i])
                    ratios.append(row[1]/row[2])
                record.update(champion_us=statistics.median(vals[0]),best_us=statistics.median(vals[1]),native_us=statistics.median(vals[2]),relative_to_best=statistics.median(ratios))
                result['trials'].append(record);print('NATIVE_TRIAL',json.dumps(record),flush=True);del trial
    valid=[r for r in result['trials'] if r.get('correctness') and r.get('packed') and 'native_us'in r]
    result['best']=max(valid,key=lambda r:r['relative_to_best']) if valid else None
    result['meets_10_percent_time_reduction']=bool(result['best'] and result['best']['relative_to_best']>=1/.9)
    return result
