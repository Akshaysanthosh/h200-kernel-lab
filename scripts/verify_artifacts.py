"""CPU-only checks: never import candidate modules or contact external services."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'muse_bf16_proj/champions/2026-10-30_1.227412/submission.py': '9ea10c81639eb23d6c9eb832465190328664fd655e89e9ff81732b94e10b8a36',
    'muse_bf16_proj/champions/2026-10-31_1.520670/submission.py': '6faa2ae8519e3b9c19b27c2fdf47a2a057c7be89c72583ecfd270175c2144ae7',
    'muse_bf16_proj/candidates/public/nathen_v18_linear.py': '6faa2ae8519e3b9c19b27c2fdf47a2a057c7be89c72583ecfd270175c2144ae7',
    'muse_bf16_proj/candidates/public/v18_sk2_hybrid.py': '4ee96b46c21a7fe096c456684519b32353ce472a4e369b4aa08023c2bb83f2c9',
    'muse_bf16_proj/candidates/public/v18_sk2_m32.py': '11822cbfddd312660603b72e21d4f2c87591995bd4829f11d30b86a6c2fb4f7b',
}

def main():
    for name, expected in EXPECTED.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f'Hash mismatch: {name}')
    counts = {'protected_hashes': len(EXPECTED), 'python_files': 0, 'json_files': 0, 'doc_links': 0}
    for path in ROOT.rglob('*'):
        if any(part.startswith('.') or part == 'vendor' for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
            counts['python_files'] += 1
        elif path.suffix == '.json':
            json.loads(path.read_text())
            counts['json_files'] += 1
    for path in [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md'))]:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            if not (path.parent / target.split('#')[0]).exists():
                raise SystemExit(f'Broken link: {path.relative_to(ROOT)} -> {target}')
            counts['doc_links'] += 1
    print(json.dumps({'passed': True, **counts}, indent=2))
    print('Artifact checks only. No GPU compilation, correctness or performance test performed.')

if __name__ == '__main__':
    main()
