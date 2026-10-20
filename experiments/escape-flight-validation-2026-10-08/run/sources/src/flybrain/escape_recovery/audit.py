"""Independent saved-count temporal replay and measured trajectory audit."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import math
import numpy as np
from flybrain.flight3d.anatomy import load_manifest,digest
from flybrain.turnflee.audit import audit_counts,_correct
from flybrain.turnflee.protocol import validation_gate,closed_gate
from .protocol import trajectory_metrics,closed_success


def temporal_audit(records:list[dict],manifest:dict) -> tuple[float|None,str|None]:
    """Reconstruct fixed rule directly from saved counts, without a controller."""
    indices=[next(i for i,n in enumerate(manifest['nodes']) if n['metadata'].get('side')==s and n['metadata'].get('cell_type')==cell)
        for s in ('left','right') for cell in ('i1','b3','b1')]
    c=np.zeros(6,dtype=np.int64);first=None;streak=0;candidate=None;side=None;trigger=None;holding=False
    for tick,r in enumerate(records,1):
        if holding and r['threat']['weak_continuous']:c[:]=0
        else:
            holding=False;c+=np.array(r['actual_counts'])[indices]-np.array(r['reference_counts'])[indices]
        l=int(c[0]+c[1]+c[5]);rr=int(c[3]+c[4]+c[2]);d=rr-l;committed=False
        if r['threat']['strong'] and first is None and not holding:first=tick
        deadline=None if first is None else first+25
        timed=deadline is not None and tick>deadline
        active=side is not None
        if not active and deadline is not None and not timed and not holding:
            eligible='right' if d>=1 and rr>=1 else 'left' if d<=-1 and l>=1 else None
            if eligible is None:candidate=None;streak=0
            elif candidate==eligible:streak+=1
            else:candidate=eligible;streak=1
            if streak>=2:
                side=candidate;committed=True
                if trigger is None:trigger=r['end_seconds'];selected=side
        elif timed and side is None:candidate=None;streak=0
        expected=dict(cumulative=c.tolist(),left=l,right=rr,strength=math.sqrt(float(c@c)),candidate=candidate,
            streak=streak,side=side,committed=committed,first_strong_tick=first,deadline=deadline,timed_out=timed)
        # Recovery callback occurs after the published readout decision in session.
        if r['temporal']!=expected:raise ValueError('temporal record mismatch')
        if r.get('recovered'):
            c[:]=0;first=None;candidate=None;streak=0;side=None;holding=True
    return trigger,selected if trigger is not None else None


def audit(directory:Path) -> dict:
    """Verify immutable receipts, physiology parity, temporal rule and gates."""
    receipt=json.loads((directory/'receipt.json').read_text());result=json.loads((directory/'results.json').read_text())
    seal=json.loads((directory.parent/'escape-recovery-seal.json').read_text())
    if seal['receipt_sha256']!=digest(directory/'receipt.json'):raise ValueError('receipt seal mismatch')
    for name,h in receipt['outputs'].items():
        if digest(directory/name)!=h:raise ValueError('saved output hash mismatch')
    if receipt['status']!=result['status']:raise ValueError('terminal status mismatch')
    config=json.loads((directory/'config.json').read_text())
    for name,h in config['candidate']['source_sha256']:
        if digest(Path(name))!=h or digest(directory/'sources'/name)!=h:raise ValueError('candidate source archive mismatch')
    for name,h in config['protected'].items():
        if digest(Path(name))!=h:raise ValueError('historical preservation mismatch')
    if result['status'] in ('failed','interrupted'):return dict(status='closed-incomplete',experiment_status=result['status'],audit_steps=0)
    groups=defaultdict(list)
    key=lambda r:(r['phase'],r['seed'],r.get('state'),r.get('intervention'))
    with gzip.open(directory/'traces.jsonl.gz','rt') as f,(directory/'traces.csv').open(newline='') as t:
        reader=csv.DictReader(t)
        for line in f:
            r=json.loads(line);row=next(reader)
            if any(row[k]!=('' if r.get(k) is None else str(r[k])) for k in reader.fieldnames):raise ValueError('CSV/JSON mismatch')
            groups[key(r)].append(r)
        if next(reader,None) is not None:raise ValueError('extra CSV rows')
    manifest=load_manifest();intervals=0
    for summary,case in zip(result['validation'],config['validation']):
        if any(summary[k]!=v for k,v in case.items()):raise ValueError('direction identity mismatch')
        records=groups[key(case)];intervals+=len(records)
        if len(records)!=75:raise ValueError('direction duration mismatch')
        counts=audit_counts(records,manifest);trigger,side=temporal_audit(records,manifest)
        onset=next((r['end_seconds'] for r in records if r['threat']['strong']),None)
        success=trigger is not None and onset is not None and trigger-onset<=.5+1e-12 and _correct(case['azimuth'],side)
        success=bool(success and counts['baseline_equal'] and counts['finite'] and counts['max_downstream_hz']<=240)
        if (summary['trigger_seconds'],summary['side'],summary['success'])!=(trigger,side,success):raise ValueError('direction result mismatch')
    if len(result['validation'])!=96:raise ValueError('direction count mismatch')
    if not validation_gate(result['validation']):
        if result['closed_loop'] or result['status']!='closed-direction-failed':raise ValueError('continued after direction failure')
    else:
        if len(result['closed_loop'])!=48:raise ValueError('closed coverage missing')
        for summary,case in zip(result['closed_loop'],config['closed_loop']):
            records=groups[key(dict(case,phase='closed-loop'))];intervals+=len(records)
            if len(records)!=300:raise ValueError('closed duration mismatch')
            audit_counts(records,manifest);temporal_audit(records,manifest)
            metrics=trajectory_metrics(case,summary,records)
            if any(summary[k]!=v for k,v in metrics.items()):raise ValueError('trajectory metrics mismatch')
            if summary['success']!=closed_success(summary):raise ValueError('physical classification mismatch')
        expected='passed' if closed_gate(result['closed_loop']) else 'closed-physical-failed'
        if result['status']!=expected:raise ValueError('physical gate mismatch')
    steps=intervals*40
    if steps!=result['reserved_kernel_steps'] or steps>864000:raise ValueError('scientific budget mismatch')
    return dict(status='audit-passed',experiment_status=result['status'],cases=result['cases'],intervals=intervals,
        neural_steps=steps,validation_successes=result['validation_successes'],intact_successes=result['intact_successes'],
        silent_triggers=result['silent_triggers'],audit_steps=0)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(audit(a.directory),indent=2))
