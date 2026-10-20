"""Quantify the two causes using only immutable historical records."""
from collections import deque
import gzip
import json
from pathlib import Path

selected = {1110000, 1110002, 1110012, 1110013, 1110014}
groups = {seed: [] for seed in selected}
with gzip.open('experiments/escape-flight-validation-2026-10-08/run/traces.jsonl.gz', 'rt') as stream:
    for line in stream:
        row = json.loads(line)
        if row['seed'] in selected and row['intervention'] == 'intact':
            groups[row['seed']].append(row)
diagnosis = []
for seed, rows in groups.items():
    small = deque(maxlen=15)
    first_net_clearance = None
    for row in rows:
        size = row['threat']['size_degrees']
        if size <= 10:
            small.append(size)
        else:
            small.clear()
        if (len(small) == 15 and (small[-1] - small[0]) / .28 <= 5 and first_net_clearance is None):
            first_net_clearance = row['end_seconds']
    recovery = next((r['end_seconds'] for r in rows if r['recovered']), None)
    collision = next((r['end_seconds'] for r in rows if r['collided']), None)
    late_small = [r for r in rows if r['end_seconds'] >= 4 and r['threat']['size_degrees'] <= 10]
    diagnosis.append(dict(seed=seed, recovered_seconds=recovery, collision_seconds=collision,
        first_net_clearance_on_original_path=first_net_clearance,
        late_small_samples=len(late_small),
        late_small_samples_rejected_by_rectified_growth=sum(r['threat']['positive_growth_degrees_s'] > 5 for r in late_small),
        late_small_samples_with_original_weak=sum(r['threat']['weak_continuous'] for r in late_small)))
result = dict(neural_steps=0, changed_records=0, cases=diagnosis,
    limit='Counterfactual clearance flag on original poses, not a new closed-loop performance result.')
path = Path('artifacts/escape-repair-2026-10-08/saved-diagnosis.json')
with path.open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
