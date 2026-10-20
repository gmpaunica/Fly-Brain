"""Saved-only independent behavioral measurements beyond state-label scoring."""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import numpy as np
from flybrain.escape_recovery.audit import audit
from flybrain.flight3d.anatomy import digest

RUN=Path('experiments/escape-recovery-execution-2026-10-07/run')
ART=Path('artifacts/escape-recovery-execution-2026-10-07')


def physical(row:dict,records:list[dict]) -> dict:
    """Independent vectorized checks from raw poses, not controller flags."""
    selected=[r for r in records if row['trigger_seconds'] is not None and r['end_seconds']>=row['trigger_seconds']-1e-12]
    if not selected:return dict(measured_turn=False,measured_takeoff=row['state']!='resting',separation_gain_m=0.,measured_braking=False,
                                collision=any(r['collided'] for r in records),weak_reached=False,recovered=False,stable_recovery=False,physical_success=False)
    sign=1 if row['side']=='right' else -1;anchor=selected[0]['pre_orientation'][2]
    states=[r['escape_state'] for r in selected]
    limit=next((k for k,s in enumerate(states) if s in ('fleeing','braking','idle')),len(selected)-1)
    turn=any(states[k] in ('turning','fleeing') and abs(sign*(selected[k]['orientation'][2]-anchor)-math.pi/2)<=math.radians(3) for k in range(limit+1))
    pos=np.array([r['position'] for r in selected]);pred=np.array([r['predator_position'] for r in selected]);v=np.array([r['velocity'] for r in selected])
    weak=[r['threat']['weak_continuous'] for r in selected]
    weak_index=next((k for k,b in enumerate(weak) if b),len(selected)-1)
    start=float(np.linalg.norm(np.array(selected[0]['pre_position'])-pred[0]))
    separation=max(0.,float(np.max(np.linalg.norm(pos[:weak_index+1]-pred[:weak_index+1],axis=1)))-start)
    streak=0;takeoff=row['state']!='resting'
    for k,r in enumerate(selected):
        streak=streak+1 if not r['foot_contact'] and pos[k,2]>=.008 else 0
        takeoff|=streak>=3
    brake=any(any(weak[:k+1]) and states[k] in ('braking','idle') and np.linalg.norm(v[k,:2])<.02 and abs(v[k,2])<.02 for k in range(len(selected)))
    recovery=False;stable=False;streak=0
    for k,r in enumerate(selected):
        recovery|=bool(r['recovered'])
        safe=recovery and states[k]=='idle' and r['behavior_state']=='cruising' and not r['collided'] and .075<=pos[k,2]<=.125 and abs(v[k,2])<.02 and r['triggers']==1
        streak=streak+1 if safe else 0;stable|=streak>=15
    collision=any(r['collided'] for r in records)
    return dict(measured_turn=bool(turn),measured_takeoff=bool(takeoff),separation_gain_m=separation,
        measured_braking=bool(brake),collision=collision,weak_reached=bool(any(weak)),recovered=bool(recovery),stable_recovery=bool(stable),
        physical_success=bool(turn and takeoff and separation>=.02-1e-12 and not collision and any(weak) and brake and recovery and stable))


if __name__=='__main__':
    result=audit(RUN)
    summary=json.loads((RUN/'results.json').read_text());groups=defaultdict(list)
    if summary['closed_loop']:
        with gzip.open(RUN/'traces.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line)
                if r['phase']=='closed-loop':groups[(r['seed'],r['state'],r['intervention'])].append(r)
        for row in summary['closed_loop']:
            measured=physical(row,groups[(row['seed'],row['state'],row['intervention'])])
            for k,v in measured.items():
                if row[k]!=v:raise ValueError('independent physical mismatch: '+k)
    result.update(independent_physical_cases=len(summary['closed_loop']),receipt_sha256=digest(RUN/'receipt.json'),brain_steps=0)
    (ART/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
