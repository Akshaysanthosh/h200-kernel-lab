"""Summarize paired development measurements without calling them official scores."""
import json
import sys
from pathlib import Path
for filename in sys.argv[1:]:
    data=json.loads(Path(filename).read_text())
    print(filename, 'passed=',data['passed'],'relative=',data.get('relative_to_champion'))
    for row in data.get('timings',[]):
        print(f"{row['layer']:10} m={row['m']:4} champion={row['champion_us']:7.2f}us candidate={row['candidate_us']:7.2f}us ratio={row['ratio']:.3f}")
