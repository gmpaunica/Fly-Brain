"""Activate only after saved plot/UI review, audits and exact replay."""
import json
from pathlib import Path
from flybrain.escape_diagnostic_recovery.runner import write,verify_hashes
from flybrain.escape_edge.validation import ART,BASE
from flybrain.escape_edge.session import source_hashes
from flybrain.escape_edge.delivery import verified_candidate
from flybrain.flight3d.anatomy import digest

preflight = json.loads((ART/'preflight.json').read_text())
verify_hashes(preflight['bound'])
verify_hashes(preflight['protected'])
assert preflight['edge_sources']==source_hashes()
review = json.loads((ART/'visual-review.json').read_text())
assert review['status']=='reviewed' and review['offscreen']=='reviewed' and review['plots']==['trajectories.png','hover.png']
sustained = json.loads((ART/'sustained-review.json').read_text())
assert sustained['status']=='passed' and sustained['cases']==48 and sustained['individual_neural_steps_added']==0
proofs = dict(preflight['proofs'])
for name in ('development-audit.json','robustness-audit.json','replay.json','plots.json',
             'trajectories.png','hover.png','visual-review.json'):
    p = ART/name
    proofs[p.as_posix()] = digest(p)
p = ART/'sustained-review.json'
proofs[p.as_posix()] = digest(p)
for phase in ('development','robustness'):
    for name in ('receipt.json','results.json','config.json'):
        p = BASE/phase/name
        proofs[p.as_posix()] = digest(p)
value = dict(status='passed',candidate=preflight['candidate'],edge_sources=source_hashes(),
    bound=preflight['bound'],proofs=proofs,runs={p:(BASE/p).as_posix() for p in ('development','robustness')},
    historical_direction=preflight['historical_direction'],optimizer_updates=0)
write(ART/'delivery.json',value)
assert verified_candidate() is not None
print('Source-bound edge launcher activated; historical launchers unchanged.')
