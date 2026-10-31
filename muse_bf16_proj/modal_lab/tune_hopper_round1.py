"""Bounded external H200 tile search. Not part of submitted kernel code."""
import json,statistics,sys,time
from pathlib import Path
from benchmark import load

def run(candidate,root='/work'):
    import torch
    root=Path(root);sys.path.insert(0,str(root))
    import reference
    from checks import compare
    champion=load(root/'champion.py','champion')
    custom=load(root/'candidate.py','custom')
    weights=reference.generate_weights(624193);state=champion.prepare(weights)
    flush=torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda')
    result={'candidate':candidate,'passed':True,'official_score':False,'shapes':[],'gpu':torch.cuda.get_device_name()}
    began=time.monotonic()
    def capture(fn):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):fn()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            flush.zero_();a.record();out=fn();b.record()
        return g,a,b,out
    def measure(g,n=16):
        vals=[]
        for _ in range(n):
            g[0].replay();g[2].synchronize();vals.append(g[1].elapsed_time(g[2])*1000)
        return statistics.median(vals)
    for layer in ('pooler','qkv_gate'):
        for m in (32,128,8,16,256):
            if time.monotonic()-began>720:
                result['truncated']=True;return result
            data=reference.generate_input(m,layer,70000+m,'outliers')
            ref=reference.ref_kernel(data,weights);probes=reference.probe_rows(data['x'])
            base=capture(lambda:champion.kernel_fn(data,state));base_us=measure(base,32)
            bm=16 if m<=16 else (32 if m==32 else 64)
            configs=[]
            # Narrow batch: orientation, occupancy, and reduction trade-off.
            for bn in (64,128):
                for sk in (1,4,8):
                    configs.append((bm,bn,128,sk,4,3,True))
                    configs.append((bm,bn,64,sk,4,3,True))
            # Wider batch: tile M reuse, stages, and conventional orientation controls.
            if m>=128:
                for tile_m in (64,128):
                    for bn in (64,128):
                        for sk in (1,2,4):
                            configs.append((tile_m,bn,64,sk,4,3,False))
            records=[]
            for cfg in configs:
                if time.monotonic()-began>720:break
                try:
                    out,compiled=custom._hopper_run(data,state,cfg,diagnostic=True)
                    ok,msg=compare(out,ref,reference.REL_L2_TOL)
                    if len(probes):
                        err=(out[probes].double()-ref[probes].double()).norm(dim=-1)/ref[probes].double().norm(dim=-1).clamp(min=1e-30)
                        ok=ok and bool((err<=reference.PROBE_TOL).all())
                    if not ok:raise RuntimeError('CORRECTNESS '+msg)
                    trial=capture(lambda:custom._hopper_run(data,state,cfg))
                    us=measure(trial)
                    ptx=compiled.asm.get('ptx','')
                    row={'config':cfg,'us':us,'ratio':base_us/us,'registers':compiled.n_regs,'spills':compiled.n_spills,
                         'shared_bytes':compiled.metadata.shared,'wgmma':'wgmma.mma_async' in ptx,
                         'tma':'cp.async.bulk.tensor' in ptx}
                    records.append(row);del trial
                except Exception as exc:
                    if 'CORRECTNESS' in str(exc): raise
                    records.append({'config':cfg,'error':str(exc)[-1600:]})
            valid=sorted([r for r in records if 'us' in r],key=lambda r:r['us'])[:2]
            confirmations=[]
            fresh=reference.generate_input(m,layer,80000+m,'normal');data['x'].copy_(fresh['x'])
            for best in valid:
                cfg=best['config'];trial=capture(lambda:custom._hopper_run(data,state,cfg))
                trial[0].replay();torch.cuda.synchronize()
                ok,msg=reference.check_implementation(data,trial[3],weights)
                if not ok:raise RuntimeError(msg)
                ratios=[]
                for rep in range(24):
                    if rep%2:t=measure(trial,1);b=measure(base,1)
                    else:b=measure(base,1);t=measure(trial,1)
                    ratios.append(b/t)
                confirmations.append({**best,'confirmed_ratio':statistics.median(ratios)})
                del trial
            row={'layer':layer,'m':m,'base_us':base_us,'trials':records,'best':confirmations}
            result['shapes'].append(row)
            print('HOPPER_SHAPE',json.dumps({k:v for k,v in row.items() if k!='trials'}),flush=True)
            del base
    return result
