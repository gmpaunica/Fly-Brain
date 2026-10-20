"""Extra saved-only review: sustained progress and full commanded flee speed.

This strengthens the user-level no-inward-turn check beyond the recorded scorer's
2 mm per-step tolerance. It never alters those results or advances a simulation.
The additional delivery requirement is declared before completing development.
"""
import gzip
import json
import math
import numpy as np
from flybrain.escape_edge.validation import ART,BASE
from flybrain.flight3d.anatomy import digest
from flybrain.escape_diagnostic_recovery.runner import write

checked = []
for phase in ('development','robustness'):
    result = json.loads((BASE/phase/'results.json').read_text())
    for index,row in enumerate(result['closed_loop']):
        if row['intervention']!='intact':
            continue
        with gzip.open(BASE/phase/'traces'/f'{index:03}.jsonl.gz','rt') as stream:
            rows = [json.loads(line) for line in stream]
        triggered = next(r for r in rows if r['triggers'])
        theta = triggered['pre_orientation'][2]+(math.pi/2 if row['side']=='right' else -math.pi/2)
        direction = np.array((math.sin(theta),math.cos(theta)))
        moving = [r for r in rows if r['escape_state'] in ('fleeing','edge_braking')]
        progress = np.array([np.array(r['position'][:2])@direction for r in moving])
        retreat = float((np.maximum.accumulate(progress)-progress).max())
        full_speed = all(abs(np.linalg.norm(r['desired_velocity'][:2])-.12)<=1e-12
                         for r in rows if r['escape_state']=='fleeing')
        checked.append(dict(phase=phase,index=index,seed=row['seed'],maximum_retreat_m=retreat,
                            full_commanded_flee_speed=full_speed,passed=retreat<=.005 and full_speed))
assert len(checked)==48 and all(r['passed'] for r in checked)
write(ART/'sustained-review.json',dict(status='passed',individual_neural_steps_added=0,
    cases=48,maximum_retreat_m=max(r['maximum_retreat_m'] for r in checked),
    retreat_limit_m=.005,full_commanded_flee_speed_m_s=.12,checked=checked,
    receipts={p:digest(BASE/p/'receipt.json') for p in ('development','robustness')}))
print('All 48 intact cases pass cumulative progress and full commanded flee-speed review.')
