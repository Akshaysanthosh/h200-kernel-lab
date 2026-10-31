"""Assemble self-contained candidates without importing PyTorch or using a GPU."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / 'champions/2026-10-30_1.227412/submission.py'
EXPECTED = '9ea10c81639eb23d6c9eb832465190328664fd655e89e9ff81732b94e10b8a36'

TAIL = '''

# Experimental native path; the preserved champion handles other shapes.
from torch.utils.cpp_extension import load_inline

_LT_LAYOUT_TRANSPOSED = {transposed}
_LT_HEURISTIC_RANK = {rank}
_LT_WORKSPACE_BUDGET = 32 * 1024 * 1024
_LT_BATCHES = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
_LT_TARGETS = ('qkv_gate', 'pooler', 'unpooler')
_LT_CPP = {cpp}


def prepare(weights):
    state = _champion_prepare(weights)
    native = load_inline(
        name={module_name!r}, cpp_sources=_LT_CPP,
        with_cuda=True, extra_cflags=['-O3'],
        extra_ldflags=['-lcublasLt', '-lcublas'], verbose=False)
    plans = {{}}
    with torch.cuda.device(weights['pooler'].device):
        for name in _LT_TARGETS:
            n, k = weights[name].shape
            for m in _LT_BATCHES:
                # Small unpooler uses only a slice: retain the champion there.
                if name == 'unpooler' and m <= 8:
                    continue
                plan = native.Plan(m, n, k, _LT_LAYOUT_TRANSPOSED,
                                   _LT_HEURISTIC_RANK, _LT_WORKSPACE_BUDGET)
                if plan.ready():
                    plans[(name, m)] = plan
    if not plans:
        raise RuntimeError('No eligible cuBLASLt plans; candidate is not validated')
    state['_lt_plans'] = plans
    print('OKC_LT_READY', native.version(), 'layout', _LT_LAYOUT_TRANSPOSED,
          'rank', _LT_HEURISTIC_RANK, 'plans', len(plans))
    for m in (8, 16, 32, 128, 256):
        plan = plans.get(('pooler', m))
        print('OKC_LT_POOLER', m, plan.info() if plan else 'fallback')
    return state


def kernel_fn(data, state):
    x, c = data['x'], data['config']
    plan = state['_lt_plans'].get((c['layer'], x.shape[0]))
    if plan is not None:
        weight = state[c['layer'] + '_t'] if _LT_LAYOUT_TRANSPOSED else state[c['layer']]
        # Pointer alignment/shape are metadata, not values or cached input results.
        if (x.is_contiguous() and weight.is_contiguous()
                and c['n'] == state[c['layer']].shape[0]
                and x.data_ptr() % 256 == 0 and weight.data_ptr() % 256 == 0):
            return plan.run(x, weight)
    return _champion_kernel_fn(data, state)
'''


def build():
    champion = CHAMPION.read_bytes()
    if hashlib.sha256(champion).hexdigest() != EXPECTED:
        raise RuntimeError('Champion hash changed; refusing to generate candidates')
    base = champion.decode().replace('def prepare(', 'def _champion_prepare(', 1)
    base = base.replace('def kernel_fn(', 'def _champion_kernel_fn(', 1)
    cpp = (ROOT / 'cuBLASLt/lt_plan.cpp').read_text()
    module_name = 'okc_lt_' + hashlib.sha256(cpp.encode()).hexdigest()[:12]
    directory = ROOT / 'candidates/lt'
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    # Diagnostic first: original storage, first eligible algorithm.
    for transposed in (False, True):
        for rank in range(3):
            name = f'lt_{"transposed" if transposed else "original"}_h{rank}.py'
            source = base + TAIL.format(transposed=transposed, rank=rank,
                                       cpp=repr(cpp), module_name=module_name)
            ast.parse(source, filename=name)
            path = directory / name
            path.write_text(source)
            manifest.append({'file': str(path.relative_to(ROOT)),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'transposed': transposed, 'heuristic_rank': rank,
                             'workspace_budget_bytes': 32 * 1024 * 1024,
                             'gpu_validated': False})
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Built {len(manifest)} candidates. Champion unchanged. GPU validation pending.')


if __name__ == '__main__':
    build()
