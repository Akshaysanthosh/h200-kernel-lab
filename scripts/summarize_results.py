"""Recalculate headline metrics from the saved scored result, without GPU imports."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'muse_bf16_proj'
meta = json.loads((root / 'champions/2026-10-31_1.520670/metadata.json').read_text())
result = json.loads((root / 'champions/2026-10-31_1.520670/result.json').read_text())
speedup = meta['speedup']
print(json.dumps({
    'status': 'historical official scored result',
    'job': meta['job'],
    'speedup': speedup,
    'speedup_ci': meta['speedup_ci'],
    'lower_time_vs_baseline_percent': 100 * (1 - 1 / speedup),
    'rank_at_submission': meta['rank_at_submission'],
    'ranked_at_submission': meta['ranked_at_submission'],
    'reported_integrity_checks': result['result']['checks'],
    'source_sha256': meta['sha256'],
}, indent=2))
