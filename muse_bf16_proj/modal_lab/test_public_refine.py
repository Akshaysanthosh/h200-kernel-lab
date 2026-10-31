"""Independent public-baseline audit, correctness, and timing calibration. Lab only."""
import json,math,statistics,subprocess,time,gc
from pathlib import Path
from benchmark import load
import triton
import triton.language as tl

@triton.jit
def read_flush(X,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    v=tl.load(X+i,i<N,0,cache_modifier='.cg').to(tl.int32)
    tl.store(Y+tl.program_id(0),tl.sum(v,0))

def run(candidate,root='/work'):
    import torch,sys
    root=Path(root);sys.path.insert(0,str(root));import reference
    champion=load(root/'parent.py','parent');hybrid=load(root/'champion.py','original');public=load(root/'candidate.py','public')
    result={'candidate':candidate,'official_score':False,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'rounds':[],'passed':False,'image':'lmsysorg/sglang:dev-muse-spark-open'}
    try:result['hardware']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit,memory.total','--format=csv'],text=True)
    except Exception as e:result['hardware_error']=str(e)
    flush=torch.ones(128*1024*1024,dtype=torch.uint8,device='cuda');sink=torch.empty(flush.numel()//4096,dtype=torch.int32,device='cuda')
    def flush_read():read_flush[(sink.numel(),)](flush,sink,flush.numel(),4096)
    flush_read();torch.cuda.synchronize()
    def capture(fn,kind='read'):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):fn()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True);g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if kind=='read':flush_read()
            elif kind=='write':flush.zero_()
            a.record();out=fn();b.record()
        return g,a,b,out
    def measure(g):g[0].replay();g[2].synchronize();return g[1].elapsed_time(g[2])*1000
    def check(data,out,w):
        ok,msg=reference.check_implementation(data,out,w)
        if not ok:raise RuntimeError(str(data['config'])+': '+msg)
        return msg
    for round_id,seed in enumerate((82731,294817)):
        weights=reference.generate_weights(seed);states=[];prep=[]
        for name,mod in [('champion',champion),('hybrid',hybrid),('public',public)]:
            t=time.monotonic();states.append(mod.prepare(weights));torch.cuda.synchronize();prep.append(time.monotonic()-t);print('PREPARED',round_id,name,prep[-1],flush=True)
        rr={'seed':seed,'prepare_seconds':prep,'correctness':[],'timings':[], 'public_lt_version':states[2].get('new_lt',{}).get('version'), 'public_lt_configs':[str(k) for k in states[2].get('new_lt',{}).get('configs',{})]}
        mods=[champion,hybrid,public,champion];ss=states+[states[0]]
        tests=[(1,'qkv_gate'),(7,'pooler'),(33,'o'),(130,'unpooler'),(5,'unpooler'),(8,'router'),(9,'router'),(512,'qkv_gate'),(1000,'unpooler')]
        tests += [(m,l) for m in (1,8,16,32,128,256) for l in reference.LAYERS]
        tests += [(m,l) for m in (15,17,31,33,127,129,255,257) for l in ('pooler','qkv_gate','router')]
        for i,(m,l) in enumerate(tests):
            data=reference.generate_input(m,l,seed+100+i,'outliers');out=public.kernel_fn(data,states[2]);msg=check(data,out,weights);saved=out.clone()
            fresh=reference.generate_input(m,l,seed+1000+i,'normal');check(fresh,public.kernel_fn(fresh,states[2]),weights)
            if not torch.equal(out,saved):raise RuntimeError('Retained output modified')
            rr['correctness'].append({'m':m,'layer':l,'message':msg})
        print('CORRECTNESS',round_id,len(tests),'passed',flush=True)
        totals=[[],[],[],[]]
        for m in (1,16,32,8,128,256):
            load_total=[0.]*4
            for l in reference.LAYERS:
                data=reference.generate_input(m,l,seed+2000+m)
                gs=[capture(lambda mod=mod,st=st:mod.kernel_fn(data,st)) for mod,st in zip(mods,ss)]
                fresh=reference.generate_input(m,l,seed+3000+m,'outliers');data['x'].copy_(fresh['x'])
                for g in gs:g[0].replay();check(data,g[3],weights)
                samples=[[] for _ in gs]
                for rep in range(40):
                    for j in ((0,1,2,3) if rep%2==0 else (3,2,1,0)):samples[j].append(measure(gs[j]))
                # Pair adjacent reversed-order repetitions, then median their ratios.
                ratios=[statistics.median([(samples[0][i]+samples[0][i+1])/(v[i]+v[i+1]) for i in range(0,40,2)]) for v in samples]
                anchor=statistics.median(samples[0]);times=[anchor/r for r in ratios]
                load_total=[a+b for a,b in zip(load_total,times)]
                rec={'m':m,'layer':l,'parent_us':times[0],'original_us':times[1],'public_us':times[2],'aa_us':times[3],'paired_ratios':ratios,'samples_us':samples};rr['timings'].append(rec)
                print('PUBLIC_TIMING',round_id,json.dumps({k:v for k,v in rec.items() if k!='samples_us'}),flush=True)
                if round_id==0 and m==16 and l=='pooler':
                    cal={}
                    for kind in ('read','write','none'):
                        gg=[capture(lambda mod=mod,st=st:mod.kernel_fn(data,st),kind) for mod,st in zip(mods[:3],ss[:3])]
                        vals=[[] for _ in gg]
                        for _ in range(30):
                            for j,g in enumerate(gg):
                                if kind=='none':flush_read()
                                vals[j].append(measure(g))
                        cal[kind]=[statistics.median(v) for v in vals]
                    # Historical external-event method, without a guessed subtraction.
                    old=[]
                    for mod,st in zip(mods[:3],ss[:3]):
                        g=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):o=mod.kernel_fn(data,st)
                        vals=[]
                        for _ in range(30):
                            flush.zero_();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000)
                        old.append(statistics.median(vals))
                    cal['historical_external_events']=old
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for mod,st in zip(mods[:3],ss[:3]):mod.kernel_fn(data,st)
                        torch.cuda.synchronize()
                    cal['kernel_events']=[{'name':e.name,'device_us':e.device_time_total} for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
                    rr['calibration']=cal
                    print('CALIBRATION',json.dumps(cal),flush=True)
            for i in range(4):totals[i].append(load_total[i])
        rr['load_times']=totals
        rr['relative_to_parent']=math.exp(sum(math.log(a/b) for a,b in zip(totals[0],totals[2]))/6)
        rr['relative_to_original']=math.exp(sum(math.log(a/b) for a,b in zip(totals[1],totals[2]))/6)
        rr['aa_ratio']=math.exp(sum(math.log(a/b) for a,b in zip(totals[0],totals[3]))/6)
        result['rounds'].append(rr);print('ROUND_RESULT',round_id,rr['relative_to_parent'],rr['relative_to_original'],rr['aa_ratio'],flush=True)
        del states,ss,weights,gs;gc.collect();torch.cuda.empty_cache()
    result['passed']=True;result['relative_to_parent']=min(r['relative_to_parent'] for r in result['rounds'])
    return result
