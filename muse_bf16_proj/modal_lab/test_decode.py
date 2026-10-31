"""Bounded cold-L2 shape evaluation, including independent-weight confirmation."""
import json,statistics,sys,subprocess,re,collections
from pathlib import Path
from benchmark import load

def run(candidate,root='/work'):
    import torch
    root=Path(root);sys.path.insert(0,str(root));import reference
    champion=load(root/'champion.py','champion');custom=load(root/'candidate.py','custom')
    weights=reference.generate_weights(537191);state=custom.prepare_cutlass(weights)
    def verify_codec(weights,state):
        w=weights['pooler'];sm,ep,base=state['_packed'];n,k=w.shape
        delta=torch.stack((ep.int()&15,ep.int()>>4),-1).reshape(n,k)
        bb=base.int().repeat_interleave(256,1)
        original=w.view(torch.int16).int()&65535
        restored=(sm.int()&127)|((sm.int()&128)<<8)|((bb+delta)<<7)
        assert torch.equal(torch.where(bb==255,original,restored),original)
        print('CODEC_BITEXACT fallback_fraction',float((base==255).float().mean()),flush=True)
    verify_codec(weights,state)
    flush=torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda')
    result={'candidate':candidate,'gpu':torch.cuda.get_device_name(),'passed':True,'official_score':False,'shape':{'layer':'pooler','m':16},'trials':[]}
    try:
        result['resources']=subprocess.check_output(['cuobjdump','--dump-resource-usage',state['_cutlass'].__file__],text=True)[-16000:]
        sass=subprocess.check_output(['cuobjdump','--dump-sass',state['_cutlass'].__file__],text=True)
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
    def evaluate(cfg,split,weights,state,seed):
        data=reference.generate_input(16,'pooler',seed,'outliers')
        out=custom._cutlass_run(data,state,cfg,split);ok,msg=reference.check_implementation(data,out,weights)
        record={'config':cfg,'split':split,'correctness':ok,'message':msg}
        if not ok:return record
        saved=out.clone();fresh=reference.generate_input(16,'pooler',seed+1,'normal')
        new=custom._cutlass_run(fresh,state,cfg,split);ok,msg=reference.check_implementation(fresh,new,weights)
        if not ok or not torch.equal(out,saved):raise RuntimeError('Fresh/lifetime failure '+msg)
        graphs=[capture(lambda:custom._cutlass_run(data,state,2,4,False)),capture(lambda:custom.kernel_fn(data,state)),capture(lambda:custom._cutlass_run(data,state,cfg,split))]
        data['x'].copy_(fresh['x'])
        for g in graphs:
            g[0].replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(data,g[3],weights)
            if not ok:raise RuntimeError('Graph correctness failure '+msg)
        vals=[[],[],[]];ratios=[]
        for rep in range(48):
            row=[0,0,0]
            for i in ((0,1,2) if rep%2==0 else (2,1,0)):row[i]=measure(graphs[i]);vals[i].append(row[i])
            ratios.append(row[1]/row[2])
        record.update(uncompressed_us=statistics.median(vals[0]),best_us=statistics.median(vals[1]),cutlass_us=statistics.median(vals[2]),relative_to_best=statistics.median(ratios),ratios=ratios)
        return record
    for cfg in (0,1,2,5):
        for split in (4,8,16):
            record=evaluate(cfg,split,weights,state,491738);result['trials'].append(record)
            print('CUTLASS_TRIAL',json.dumps({k:v for k,v in record.items() if k!='ratios'}),flush=True)
    valid=sorted([r for r in result['trials'] if r.get('correctness') and 'cutlass_us'in r],key=lambda r:r['relative_to_best'],reverse=True)
    result['confirmations']=[]
    # Confirm top two with different weights, fresh input and new graph allocations.
    weights2=reference.generate_weights(715913);state2=custom.prepare_cutlass(weights2);verify_codec(weights2,state2)
    for r in valid[:2]:
        rec=evaluate(r['config'],r['split'],weights2,state2,751931);result['confirmations'].append(rec)
        print('CUTLASS_CONFIRM',json.dumps({k:v for k,v in rec.items() if k!='ratios'}),flush=True)
    result['passed']=all(r.get('correctness',False) for r in result['trials']+result['confirmations'])
    result['best']=valid[0] if valid else None
    result['meets_10_percent_time_reduction']=any(r.get('relative_to_best',0)>=1/.9 and r.get('correctness') for r in result['confirmations'])
    return result
