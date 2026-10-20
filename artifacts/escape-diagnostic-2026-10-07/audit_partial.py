"""Verify the consumed first baseline chunk and reproduce the comparison error."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('experiments/escape-diagnostic-2026-10-07')
OUT=Path('artifacts/escape-diagnostic-2026-10-07')
calls=Counter()
def guard(frame,event,arg):
    if event!='call': return
    filename=frame.f_code.co_filename.replace('\\','/')
    if '/src/flybrain/' not in filename: return
    key=filename.split('/src/flybrain/')[1]+':'+frame.f_code.co_name; calls[key]+=1
    if not filename.endswith('/escape_diagnostic/audit.py') and frame.f_code.co_name!='<module>':
        raise RuntimeError('Forbidden simulation/sensor call: '+key)


def main():
    """All arithmetic derives from saved observations, adding zero kernel steps."""
    sys.setprofile(guard)
    from flybrain.escape_diagnostic.audit import audit,check_chunk,exact
    sealed=audit(ROOT)
    assert sealed['status']=='closed-incomplete'
    receipt=sealed['receipt']; assert receipt['recorded_intervals']==1 and receipt['reserved_kernel_steps']==40
    with gzip.open(ROOT/'00-1100000/intervals.jsonl.gz','rt') as f: records=[json.loads(line) for line in f]
    assert len(records)==1; record=records[0]
    with gzip.open('experiments/turnflee-2026-10-06/traces.jsonl.gz','rt') as f:
        for line in f:
            original=json.loads(line)
            if (original['phase'],original['candidate'],original['seed'],original['interval'])==('validation',1,1100000,0): break
        else: raise ValueError('original identity missing')
    assert record['original']==original
    arrays=dict(np.load(ROOT/'00-1100000/000.npz',allow_pickle=False))
    graph=dict(np.load(ROOT/'graph.npz',allow_pickle=False))
    measured=check_chunk(arrays,graph,1)
    assert arrays['step_index'].tolist()==list(range(1,21))
    assert hashlib.sha256(arrays['rgb'].tobytes()).hexdigest()==original['eyes_sha256']
    assert not arrays['rgb'].any()
    for label,diag in [('actual',original['pathway']),('reference',record['reference_pathway'])]:
        exact(measured[label]['counts'],original[label+'_counts'],label+' counts')
        for field in ('current_sum','current_peak','discarded_current','peak_voltage','peak_phase'):
            exact(measured[label][field],diag[field],label+' '+field)
        exact(measured[label]['counts'],diag['spike_counts'],label+' diagnostic counts')
        first=[int(v)*.001 if v>=0 else None for v in measured[label]['first']]
        assert first==diag['first_spike_seconds']==diag['episode_first_spike_seconds']
    exact(original['actual_counts'],original['reference_counts'],'paired baseline')
    exact(record['entered_increment'],np.zeros(589),'entry')
    exact(record['rolling_all'],np.zeros(589),'rolling')
    exact(record['cumulative_increment'],np.zeros(589),'cumulative')
    assert record['expired_interval'] is None and original['selected_side'] is None
    # asdict() preserves tuple fields; reading original JSON produces lists.
    runtime=dict(record['original'])
    runtime['pathway']={k:tuple(v) if isinstance(v,list) else v for k,v in runtime['pathway'].items()}
    assert runtime!=original
    assert json.loads(json.dumps(runtime,allow_nan=False))==original
    affected=[k for k in runtime['pathway'] if runtime['pathway'][k]!=original['pathway'][k]]
    sys.setprofile(None)
    result=dict(status='partial-record-audit-passed',hypothesis='unresolved',completed_presentations=0,
        started_presentations=1,rendered_threat_intervals=0,simulated_seconds=.02,
        actual_kernel_steps=20,reference_kernel_steps=20,total_kernel_steps=40,
        all_neurons=589,all_nonself_edges=len(graph['pre']),saved_original_interval_exact=True,
        complete_step_recurrence_exact=True,brain_sensor_simulation_calls=0,audit_kernel_steps=0,
        cause='Runtime asdict tuple fields compared directly with JSON lists; saved serialized record equals original exactly.',
        tuple_list_fields=affected,original_receipt_unchanged=True,retry_authorized=False,project_calls=dict(calls))
    (OUT/'partial-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='project_calls'},indent=2))


if __name__=='__main__': main()
