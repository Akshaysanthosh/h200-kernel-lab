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
    weights=reference.generate_weights(624193);state=custom.prepare(weights)
    flush=torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda')
    result={'candidate':candidate,'passed':True,'official_score':False,'shapes':[],'gpu':torch.cuda.get_device_name()}
    result['packing']={}
    for name in ('pooler','qkv_gate','unpooler','o'):
        w=state[name];sm,ep,base=state[name+'_packed'];n,k=w.shape
        delta=torch.stack((ep.to(torch.int32)&15,ep.to(torch.int32)>>4),-1).reshape(n,k)
        bb=base.to(torch.int32).repeat_interleave(256,dim=1)
        bits=(sm.to(torch.int32)&127)|((sm.to(torch.int32)&128)<<8)|((bb+delta)<<7)
        original=w.view(torch.int16).to(torch.int32)&65535
        restored=torch.where(bb==255,original,bits)
        assert torch.equal(original,restored), 'Lossless codec verification failed'
        result['packing'][name]={'bit_exact':True,'fallback_fraction':float((base==255).float().mean()),'packed_fraction':(sm.numel()+ep.numel()+base.numel())/(w.numel()*2)}
    print('PACKING',json.dumps(result['packing']),flush=True)
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
    for layer in ('pooler','qkv_gate','unpooler','o'):
        for m in (1,8,32):
            if time.monotonic()-began>720:
                result['truncated']=True;return result
            data=reference.generate_input(m,layer,70000+m,'outliers')
            ref=reference.ref_kernel(data,weights);probes=reference.probe_rows(data['x'])
            base=capture(lambda:champion.kernel_fn(data,state));base_us=measure(base,32)
            bm=16 if m<=16 else (32 if m==32 else 64)
            configs=[]
            if m==1: configs=[('vector',r,w) for r in (1,2,4) for w in (4,8)]
            else:
                configs=[('dot',16 if m<=16 else 32,bn,128,sk,4,3) for bn in (32,64) for sk in (1,4,16)]
            configs=[cfg for cfg in configs if cfg[0]=='vector' or data['x'].shape[1]%(cfg[3]*cfg[4])==0]
            records=[]
            for cfg in configs:
                if time.monotonic()-began>720:break
                try:
                    out,compiled=custom._packed_run(data,state,cfg,diagnostic=True)
                    ok,msg=reference.check_implementation(data,out,weights)
                    if not ok:raise RuntimeError('CORRECTNESS '+msg)
                    trial=capture(lambda:custom._packed_run(data,state,cfg))
                    us=measure(trial)
                    ptx=compiled.asm.get('ptx','')
                    row={'config':cfg,'us':us,'ratio':base_us/us,'registers':compiled.n_regs,'spills':compiled.n_spills,
                         'shared_bytes':compiled.metadata.shared,'wgmma':'wgmma.mma_async' in ptx,
                         'tma':'cp.async.bulk.tensor' in ptx}
                    records.append(row);del trial
                except Exception as exc:
                    records.append({'config':cfg,'error':str(exc)[-1600:]})
            valid=sorted([r for r in records if 'us' in r],key=lambda r:r['us'])[:2]
            confirmations=[]
            fresh=reference.generate_input(m,layer,80000+m,'normal');data['x'].copy_(fresh['x'])
            for best in valid:
                cfg=best['config'];trial=capture(lambda:custom._packed_run(data,state,cfg))
                trial[0].replay();torch.cuda.synchronize()
                ok,msg=reference.check_implementation(data,trial[3],weights)
                if not ok:
                    records.append({'config':cfg,'error':'fresh-input correctness: '+msg});del trial;continue
                ratios=[]
                for rep in range(24):
                    if rep%2:t=measure(trial,1);b=measure(base,1)
                    else:b=measure(base,1);t=measure(trial,1)
                    ratios.append(b/t)
                confirmations.append({**best,'confirmed_ratio':statistics.median(ratios)})
                del trial
            row={'layer':layer,'m':m,'base_us':base_us,'trials':records,'best':confirmations}
            result['shapes'].append(row)
            print('PACKED_SHAPE',json.dumps({k:v for k,v in row.items() if k!='trials'}),flush=True)
            del base
    return result
