"""Offline development search only; never embedded in submitted preparation."""
import json
import statistics
import sys
import time
from pathlib import Path
from benchmark import load


def run(candidate,root='/work'):
    import torch
    sys.path.insert(0,root)
    import reference
    from checks import compare
    root=Path(root)
    champion=load(root/'champion.py','champion')
    challenger=load(root/'candidate.py','challenger')
    weights=reference.generate_weights(246813)
    state=champion.prepare(weights)
    native_state=challenger.prepare(weights)
    Plan=type(next(iter(native_state['_lt_plans'].values())))
    flush=torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda')
    result={'candidate':candidate,'passed':True,'official_score':False,
            'method':'CUDA events captured inside graph after cache flush; two-stage search, fresh-seed validation',
            'gpu':torch.cuda.get_device_name(),'shapes':[]}
    began=time.monotonic()
    def capture(fn):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): fn()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        start=torch.cuda.Event(enable_timing=True,external=True)
        end=torch.cuda.Event(enable_timing=True,external=True)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            flush.zero_()
            start.record()
            out=fn()
            end.record()
        return g,start,end,out
    def measure(captured,repeats=12):
        g,a,b,_=captured
        values=[]
        for _ in range(repeats):
            g.replay();b.synchronize();values.append(a.elapsed_time(b)*1000)
        return statistics.median(values)
    for layer in ('pooler','qkv_gate','unpooler'):
        for m in (8,16,32,128,256):
            if layer=='unpooler' and m==8: continue
            if time.monotonic()-began>720:
                result['truncated']=True
                return result
            data=reference.generate_input(m,layer,50000+m,'outliers')
            n,k=weights[layer].shape
            ref=reference.ref_kernel(data,weights)
            probes=reference.probe_rows(data['x'])
            base=capture(lambda:champion.kernel_fn(data,state))
            base_us=measure(base,24)
            control=capture(lambda:champion.kernel_fn(data,state))
            control_ratio=base_us/measure(control,24)
            records=[];seen=set()
            for transposed in (False,True):
                weight=state[layer+'_t'] if transposed else state[layer]
                for rank in range(32):
                    plan=Plan(m,n,k,transposed,rank,32*1024*1024)
                    if not plan.ready(): break
                    identity=(transposed,plan.info())
                    if identity in seen: continue
                    seen.add(identity)
                    trial=capture(lambda:plan.run(data['x'],weight))
                    trial[0].replay();torch.cuda.synchronize()
                    ok,msg=compare(trial[3],ref,reference.REL_L2_TOL)
                    if len(probes):
                        err=(trial[3][probes].double()-ref[probes].double()).norm(dim=-1)/ref[probes].double().norm(dim=-1).clamp(min=1e-30)
                        ok=ok and bool((err<=reference.PROBE_TOL).all())
                    if not ok: raise RuntimeError(f'Algorithm correctness {layer} {m} {rank}: {msg}')
                    us=measure(trial)
                    records.append({'transposed':transposed,'rank':rank,'us':us,'info':plan.info()})
                    del trial,plan
            short=sorted(records,key=lambda x:x['us'])[:3]
            # Independent same-shape input before paired confirmation of finalists.
            fresh=reference.generate_input(m,layer,70000+m,'normal')
            data['x'].copy_(fresh['x'])
            confirmed=[]
            for choice in short:
                plan=Plan(m,n,k,choice['transposed'],choice['rank'],32*1024*1024)
                weight=state[layer+'_t'] if choice['transposed'] else state[layer]
                trial=capture(lambda:plan.run(data['x'],weight))
                trial[0].replay();torch.cuda.synchronize()
                ok,msg=reference.check_implementation(data,trial[3],weights)
                if not ok: raise RuntimeError(msg)
                ratios=[];base_times=[];trial_times=[]
                for rep in range(24):
                    if rep%2:
                        t=measure(trial,1);b=measure(base,1)
                    else:
                        b=measure(base,1);t=measure(trial,1)
                    ratios.append(b/t);base_times.append(b);trial_times.append(t)
                confirmed.append({**choice,'confirmed_ratio':statistics.median(ratios),
                    'base_us':statistics.median(base_times),'confirmed_us':statistics.median(trial_times)})
                del trial,plan
            row={'layer':layer,'m':m,'control_ratio':control_ratio,'base_us':base_us,
                 'tested':len(records),'choices':confirmed,'all_trials':records}
            result['shapes'].append(row)
            print('TUNE_SHAPE',json.dumps({k:v for k,v in row.items() if k!='all_trials'}),flush=True)
            del base,control
    return result
