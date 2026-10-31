"""One quota-gated official free check. Never submits, promotes, or rents compute."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CLI = Path('/Users/akshay/.okc/bin/okc')
NOT_BEFORE = datetime(2026, 10, 31, 0, 0, tzinfo=timezone.utc)
DAILY_CAP = 8  # Initial cuBLASLt phase, not an unattended full-quota sweep.
EXPECTED_CHAMPION = '9ea10c81639eb23d6c9eb832465190328664fd655e89e9ff81732b94e10b8a36'


def eligible(now, quota, used_here):
    if now < NOT_BEFORE:
        return False, 'Waiting for October 31 UTC reset'
    if quota.get('left', 0) <= 0:
        return False, 'No official free checks remain'
    if used_here >= DAILY_CAP:
        return False, 'Local initial-phase daily cap reached'
    return True, 'Eligible for one free check'


def candidate_path(name):
    path = (ROOT / name).resolve()
    entries = json.loads((ROOT / 'candidates/lt/manifest.json').read_text())
    found = next((e for e in entries if (ROOT / e['file']).resolve() == path), None)
    if found is None or hashlib.sha256(path.read_bytes()).hexdigest() != found['sha256']:
        raise RuntimeError('Candidate is missing from manifest or has changed; rebuild first')
    return path


def verify_champion():
    path = ROOT / 'champions/2026-10-30_1.227412/submission.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_CHAMPION:
        raise RuntimeError('Preserved champion hash mismatch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', help='Manifest-listed path relative to the problem folder')
    parser.add_argument('--execute', action='store_true', help='Run exactly one official check')
    args = parser.parse_args()
    path = candidate_path(args.candidate)
    verify_champion()
    now = datetime.now(timezone.utc)
    outdir = ROOT / 'results/lt' / now.strftime('%Y-%m-%d')
    outdir.mkdir(parents=True, exist_ok=True)
    # Count reserved attempts, including interrupted runs; never blindly resubmit them.
    used = len(list(outdir.glob('*.attempt.json')))
    problems = json.loads(subprocess.check_output([str(CLI), 'problems', '--json'], text=True))
    problem = next(p for p in problems['problems'] if p['name'] == 'muse_bf16_proj')
    quota = problem['limits']['checks']
    ok, reason = eligible(now, quota, used)
    print(json.dumps({'eligible': ok, 'reason': reason, 'quota': quota,
                      'local_attempts_today': used, 'execute': args.execute}), flush=True)
    if not args.execute or not ok:
        return 0 if not args.execute else 2
    lock = ROOT / 'results/lt/.check.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError('Another check may be running. Inspect jobs/lock; do not resubmit blindly.')
    with os.fdopen(descriptor, 'w') as file:
        file.write(json.dumps({'pid': os.getpid(), 'candidate': str(path), 'started': now.isoformat()}))
    try:
        # Re-evaluate reservations after the lock, since another local run may have finished.
        if len(list(outdir.glob('*.attempt.json'))) >= DAILY_CAP:
            raise RuntimeError('Local phase cap reached')
        prefix = outdir / (now.strftime('%H%M%S') + '_' + path.stem)
        attempt = prefix.with_suffix('.attempt.json')
        attempt.write_text(json.dumps({'candidate': str(path), 'started': now.isoformat(),
                                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))
        with prefix.with_suffix('.json').open('w') as output, prefix.with_suffix('.stderr').open('w') as err:
            proc = subprocess.run([str(CLI), 'check', str(path), '--json'],
                                  cwd=ROOT, stdout=output, stderr=err)
        data = json.loads(prefix.with_suffix('.json').read_text())
        result = data.get('result') or {}
        print(json.dumps({'job': data.get('id'), 'passed': result.get('passed'),
                          'speedup': result.get('speedup'), 'flags': result.get('flags'),
                          'result_file': str(prefix.with_suffix('.json'))}), flush=True)
        verify_champion()
        return proc.returncode
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
