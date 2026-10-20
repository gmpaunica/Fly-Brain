"""Read-only reconciliation of the one authorized run and preservation inventory."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
DEV=Path(__file__).parent
RUN=ROOT/'experiments/flight3d-landing-2026-10-01'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    result=read(RUN/'results.json');protocol=read(RUN/'protocol.json')
    assert result['status'] in ('closed_failed_gate','accepted','incomplete')
    progress=[json.loads(line) for line in (RUN/'progress.jsonl').read_text().splitlines()]
    episodes=[json.loads(line) for line in (RUN/'episodes.jsonl').read_text().splitlines()] if (RUN/'episodes.jsonl').exists() else []
    assert episodes==result['episodes']
    assert len(episodes)==sum(r['event']=='episode_completed' for r in progress)
    updates=[r for r in progress if r['event']=='optimizer_update_completed']
    assert [r['actual_optimizer_updates'] for r in updates]==list(range(1,result['actual_optimizer_updates']+1))
    assert sum(r['event']=='optimizer_update_started' for r in progress)==len(updates)
    assert sha(RUN/'protocol.json')==read(RUN/'protocol.sha256.json')['sha256']
    receipt=read(ROOT/'experiments/.flight3d-visual-repair-2026-10-01-consumed.json')
    assert receipt['consumed'] and receipt['protocol_sha256']==sha(RUN/'protocol.json')
    for name,value in protocol['source_sha256'].items():assert sha(RUN/'source'/name)==value
    differences=[name for name,value in protocol['source_sha256'].items() if sha(ROOT/'src/flybrain/flight3d'/name)!=value]
    assert set(differences)<=set(read(DEV/'post-freeze-ui.json')['changes'])
    before=read(DEV/'preservation-before.json')['files']
    protected={p:h for p,h in before.items() if not p.startswith('src/flybrain/flight3d/')}
    assert all(sha(ROOT/p)==value for p,value in protected.items())
    assert sha(ROOT/'data/flight3d/circuit.json')==protocol['circuit_sha256']==sha(RUN/'circuit.json')
    valid={split:{c['seed'] for c in cases} for split,cases in protocol['cases'].items()}
    for row in episodes:
        split=row['label'].split('-')[-1];assert row['seed'] in valid[split]
        with np.load(RUN/'trajectories'/f"{row['label']}-{row['seed']}.npz") as data:
            trace=data['trace'];assert len(trace)==1500
            # trace: XYZ, velocity, orientation, five commands, foot/food/crash/dwell
            success=bool(trace[-1,15] and not trace[-1,16] and trace[-1,17]>=2 and np.linalg.norm(trace[-1,3:6])<1e-4)
            assert success==row['success']
    learned={}
    if updates:
        curves=read(RUN/'learning_curves.json');assert len(curves)==len(updates)==2000
        assert all(r['gradient_norm']>0 for r in updates)
        trained=torch.load(RUN/'trained.pt',weights_only=True,map_location='cpu')
        original=torch.load(RUN/'untrained.pt',weights_only=True,map_location='cpu')
        candidate=torch.load(RUN/'candidate.pt',weights_only=True,map_location='cpu')
        assert all(torch.equal(v,candidate['state'][k]) for k,v in trained['state'].items())
        assert sha(RUN/'candidate.pt')==updates[-1]['checkpoint_sha256']
        assert trained['seed']==original['seed']==940404
        with np.load(RUN/'training.npz') as data:
            x=torch.from_numpy(data['features'])
            assert torch.equal(trained['state']['mean'],x.mean(0))
            assert torch.equal(trained['state']['scale'],x.std(0).clamp_min(1))
        deltas={k:float((v-original['state'][k]).norm()) for k,v in trained['state'].items() if k.startswith('heads.')}
        recorded=read(RUN/'parameter_changes.json')
        assert all(np.isclose(v,recorded[k]) and v>0 for k,v in deltas.items())
        assert read(RUN/'reload.json')=={'exact_outputs':True,'frozen_parameters':True}
        learned={'changed_parameter_tensors':len(deltas),'normalization_train_only':True,'checkpoint_reload_exact':True}
    record={'verified':True,'status':result['status'],'stage':result['stage'],
        'actual_optimizer_updates':len(updates),'episodes_by_label':dict(Counter(r['label'] for r in episodes)),
        'preserved_files':len(protected),'current_ui_only_differences':differences,
        'frozen_sources_intact':True,'protocol_receipt_bound':True,'circuit_intact':True,**learned}
    (DEV/'reconciliation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__':main()
