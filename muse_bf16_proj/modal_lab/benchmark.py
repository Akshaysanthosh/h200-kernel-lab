"""Development-only evaluator. Never included in a competition submission."""
import importlib.util
import math
from pathlib import Path
import statistics
import sys


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(candidate, root='/work', seed=82731):
    import torch
    root = Path(root)
    sys.path.insert(0, str(root))
    import reference
    champion = load(root / 'champion.py', 'champion')
    challenger = load(root / 'candidate.py', 'challenger')
    weights = reference.generate_weights(seed)
    states = [champion.prepare(weights), challenger.prepare(weights)]
    modules = [champion, challenger]
    result = {'candidate': candidate, 'gpu': torch.cuda.get_device_name(),
              'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'weight_seed':seed, 'official_score': False, 'correctness': [], 'timings': []}
    result['native_plans'] = {str(k):v.info() for k,v in states[1].get('_lt_plans',{}).items()}
    assert 'H200' in result['gpu'], result['gpu']
    tests = [(1,'qkv_gate','normal'), (7,'pooler','normal'), (33,'o','normal'),
             (130,'unpooler','normal'), (5,'unpooler','normal'), (8,'router','normal'),
             (9,'router','outliers'), (512,'qkv_gate','outliers'), (1000,'unpooler','outliers')]
    # Exercise native paths as well as the public correctness cases.
    tests += [(m,l,'outliers') for m in (8,16,32,128,256) for l in ('qkv_gate','pooler','unpooler')]
    tests += [(m,l,'outliers') for m in (9,15,17,31,33) for l in ('qkv_gate','pooler')]
    for i,(m,layer,dist) in enumerate(tests):
        data = reference.generate_input(m,layer,91000+i,dist)
        output = challenger.kernel_fn(data,states[1])
        ok,msg = reference.check_implementation(data,output,weights)
        result['correctness'].append({'m':m,'layer':layer,'passed':ok,'message':msg})
        if not ok:
            result['passed'] = False
            return result
        # Fresh input and old-output lifetime check, outside timed sections.
        saved = output.clone()
        fresh = reference.generate_input(m,layer,92000+i,dist)
        new = challenger.kernel_fn(fresh,states[1])
        ok,msg = reference.check_implementation(fresh,new,weights)
        if not ok or not torch.equal(output,saved):
            raise RuntimeError(f'Fresh input/output lifetime failure: {layer} {m}: {msg}')
    # 128 MiB exceeds H200 L2. Flush outside the measured graph replay.
    flush = torch.empty(128*1024*1024, dtype=torch.uint8, device='cuda')
    load_ratios = []
    for m in (1,16,32,8,128,256):
        totals = [0.,0.]
        for layer in reference.LAYERS:
            data = reference.generate_input(m,layer,100000+m)
            graphs, outputs = [], []
            for mod,state in zip(modules,states):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3): mod.kernel_fn(data,state)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = mod.kernel_fn(data,state)
                graphs.append(graph); outputs.append(output)
            # Replays must read new values, not captured results.
            fresh = reference.generate_input(m,layer,110000+m)
            data['x'].copy_(fresh['x'])
            for graph,output in zip(graphs,outputs):
                graph.replay()
                ok,msg = reference.check_implementation(data,output,weights)
                if not ok: raise RuntimeError(f'Graph replay failure {layer} {m}: {msg}')
            samples = [[],[]]
            for rep in range(20):
                for j in ((0,1) if rep%2==0 else (1,0)):
                    flush.zero_()
                    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record(); graphs[j].replay(); end.record(); end.synchronize()
                    samples[j].append(start.elapsed_time(end)*1000)
            times = [statistics.median(s) for s in samples]
            totals = [a+b for a,b in zip(totals,times)]
            result['timings'].append({'m':m,'layer':layer,'champion_us':times[0],
                'candidate_us':times[1],'ratio':times[0]/times[1], 'samples_us':samples})
        load_ratios.append(totals[0]/totals[1])
    result['passed'] = True
    result['relative_to_champion'] = math.exp(sum(map(math.log,load_ratios))/len(load_ratios))
    result['timing_note'] = 'Paired cold-L2 CUDA graph event timing; no fixed-overhead subtraction. Development estimate, not official score.'
    return result
