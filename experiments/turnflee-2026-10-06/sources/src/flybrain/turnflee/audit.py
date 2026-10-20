"""Independent saved-record reconciliation; never instantiate or advance a brain."""
import argparse
from collections import defaultdict, deque
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
from flybrain.flight3d.anatomy import CACHE, load_manifest


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition: raise ValueError(reason)


def audit_counts(records: list[dict], manifest: dict) -> dict:
    """Recompute signed 200 ms windows and both downstream sustained rates."""
    wings=[i for i,n in enumerate(manifest['nodes']) if set(n['roles'])&{'wing_power','wing_steering'}]
    downstream=np.array([not bool(set(n['roles'])&{'odor','vision','optic_flow','motion','taste','touch'})
                         for n in manifest['nodes']])
    windows=deque(maxlen=10); counts=deque(maxlen=10); maximum=0.; baseline=True
    for interval,r in enumerate(records):
        a=np.array(r['actual_counts']); b=np.array(r['reference_counts'])
        require(a.shape==b.shape==(589,) and a.dtype.kind in 'iu' and b.dtype.kind in 'iu'
                and (a>=0).all() and (b>=0).all(),'invalid paired counts')
        require(r['interval']==interval and abs(r['end_seconds']-(interval+1)*.02)<1e-12,'interval clock mismatch')
        windows.append((a-b)[wings]); counts.append(np.stack((a,b)))
        require(np.array_equal(np.sum(windows,axis=0),r['wing_vector']),'rolling wing vector mismatch')
        if interval<25: baseline &= bool(np.array_equal(a,b))
        if len(counts)==10: maximum=max(maximum,float(np.max(np.sum(counts,axis=0)[:,downstream]))/.2)
        require(maximum==r['max_downstream_hz'],'downstream rate mismatch')
    return dict(intervals=len(records),baseline_equal=baseline,max_downstream_hz=maximum,
                finite=all(r['finite'] for r in records))


def _correct(azimuth: float, side: str | None) -> bool:
    if side not in ('left','right'): return False
    angle=azimuth%360
    return min(angle,360-angle,abs(angle-180))<=15 or side==('left' if 0<angle<180 else 'right')


def audit_presentation(records: list[dict], row: dict, manifest: dict) -> dict:
    counts=audit_counts(records,manifest)
    wings=[n for n in manifest['nodes'] if set(n['roles'])&{'wing_power','wing_steering'}]
    entries=[(j,n['metadata']['side'],n['metadata']['cell_type']) for j,n in enumerate(wings)
             if n['metadata']['cell_type'] in ('i1','b3','b1')]
    onset=None; trigger=None; side=None; pending=-1; rng=random.Random(row['seed'])
    for tick,r in enumerate(records,1):
        score={'left':0.,'right':0.}; norms={'left':0.,'right':0.}
        for j,s,cell in entries:
            value=r['wing_vector'][j]
            score[('right' if s=='left' else 'left') if cell=='b1' else s]+=value
            norms[s]+=value*value
        strength=math.sqrt(sum(norms.values())); bilateral=all(v>0 for v in norms.values())
        expected=dict(left=score['left'],right=score['right'],strength=strength,bilateral=bilateral)
        require(r['wing_scores']==expected,'wing score mismatch')
        if r['threat']['strong']:
            pending=tick+25
            if onset is None: onset=r['end_seconds']
        if trigger is None and tick<=pending and strength>=1 and (score['left']!=score['right'] or bilateral):
            difference=score['right']-score['left']
            side=('right' if difference>0 else 'left') if difference else rng.choice(('left','right'))
            trigger=r['end_seconds']
        if trigger is None: require(r['triggers']==0,'ungated recorded trigger')
        elif r['end_seconds']==trigger:
            require(r['triggers']==1 and r['selected_side']==side,'spike-selected side mismatch')
    require(row['first_strong_seconds']==onset and row['trigger_seconds']==trigger and row['side']==side,
            'response timing/side summary mismatch')
    for k in ('baseline_equal','finite','max_downstream_hz'):
        require(row[k]==counts[k],f'{k} summary mismatch')
    success=(onset is not None and trigger is not None and trigger>=onset
             and (row.get('phase')=='calibration' or trigger-onset<=.5+1e-12)
             and _correct(row['azimuth'],side) and counts['baseline_equal'] and counts['finite']
             and counts['max_downstream_hz']<=240)
    require(row['success']==success,'presentation success mismatch')
    return counts


def audit(directory: Path) -> dict:
    """Verify receipts, sources, case identity, count windows and strict stops."""
    receipt=json.loads((directory/'receipt.json').read_text())
    for name,value in receipt['sha256'].items(): require(sha(directory/name)==value,f'receipt mismatch: {name}')
    config=json.loads((directory/'config.json').read_text()); result=json.loads((directory/'results.json').read_text())
    require(receipt['status']==result['status'],'receipt status mismatch')
    lock=json.loads((directory/'lock.json').read_text())
    require(sha(directory/'config.json')==lock['config_sha256'],'protocol lock mismatch')
    for name,value in config['source_sha256'].items():
        require(sha(directory/'sources'/name)==value,f'archive source mismatch: {name}')
        require(sha(Path(name))==value,f'current source differs: {name}')
    for name,value in config['history_audit']['files'].items(): require(sha(Path(name))==value,'historical protocol changed')
    require(sha(CACHE/'circuit.json')==config['circuit_sha256'],'circuit changed')
    require(config['gain_pairs']==[[2,1],[4,2],[8,4]],'candidate list changed')
    manifest=load_manifest(); groups=defaultdict(list)
    key=lambda r:(r['phase'],r['candidate'],r['azimuth'],r.get('elevation'),r.get('profile'),r['seed'])
    with gzip.open(directory/'traces.jsonl.gz','rt',encoding='utf-8') as trace, (directory/'traces.csv').open(newline='') as table:
        reader=csv.DictReader(table); intervals=0
        for line in trace:
            record=json.loads(line); csv_row=next(reader,None)
            require(csv_row is not None,'missing CSV interval')
            for field in reader.fieldnames:
                value=record.get(field)
                require(csv_row[field]==('' if value is None else str(value)),f'CSV mismatch: {field}')
            groups[key(record)].append(record); intervals+=1
        require(next(reader,None) is None,'extra CSV interval')
    calibration=json.loads((directory/'calibration.json').read_text())
    require(calibration==result['calibration'] and len(calibration)==3,'calibration record mismatch')
    expected=set(); baseline_intervals=0
    for index,candidate in enumerate(calibration):
        require(candidate['gains']==config['gain_pairs'][index],'gain ordering mismatch')
        require(len(candidate['presentations'])==16,'calibration case count mismatch')
        for row,case in zip(candidate['presentations'],config['calibration']):
            require(all(row[k]==v for k,v in case.items()),'calibration case mismatch')
            k=key(dict(row,phase='calibration',candidate=index)); expected.add(k)
            require(len(groups[k])==75,'presentation duration mismatch')
            audit_presentation(groups[k],row,manifest); baseline_intervals+=25
        require(candidate['eligible']==all(r['success'] for r in candidate['presentations']),'eligibility mismatch')
        require(candidate['successes']==sum(r['success'] for r in candidate['presentations']),'calibration score mismatch')
    selected=next((r['gains'] for r in calibration if r['eligible']),None)
    require(result['selected_gains']==selected,'lowest eligible selection mismatch')
    validation=result['validation']; closed=result['closed_loop']
    if selected is None:
        require(not validation and not closed and result['status']=='closed-calibration-failed','continued after calibration failure')
    else:
        require(len(validation)==96,'validation count mismatch')
        index=config['gain_pairs'].index(selected)
        for row,case in zip(validation,config['validation']):
            require(all(row[k]==v for k,v in case.items()),'validation case mismatch')
            k=key(dict(row,candidate=index)); expected.add(k)
            require(len(groups[k])==75,'validation duration mismatch')
            audit_presentation(groups[k],row,manifest); baseline_intervals+=25
        passed=(sum(r['success'] for r in validation)>=90 and
                all(sum(r['success'] for r in validation if r['azimuth']==az)>=5
                    for az in {r['azimuth'] for r in validation}))
        if not passed: require(not closed and result['status']=='closed-validation-failed','continued after validation failure')
        else:
            require(len(closed)==48,'closed case count mismatch')
            for row,case in zip(closed,config['closed_loop']):
                require(all(row[k]==v for k,v in case.items()),'closed case mismatch')
                k=key(dict(row,phase='closed-loop',candidate=-1)); expected.add(k)
                records=groups[k]; require(len(records)==300,'closed duration mismatch')
                audit_counts(records,manifest)
                require(row['triggers']==records[-1]['triggers'],'closed trigger mismatch')
                require(row['collision']==any(r['collided'] for r in records),'closed collision mismatch')
                # Response/weak/recovery and physical success remain recorded independently.
                weak=any(r['triggers'] and r['threat']['weak_continuous'] for r in records)
                recovered=any(r['recovered'] for r in records)
                streak=0; stable=False; seen=False
                for r in records:
                    seen |= r['recovered']
                    safe=(seen and r['escape_state']=='idle' and not r['collided'] and
                          .075<=r['position'][2]<=.125 and abs(r['velocity'][2])<.02)
                    streak=streak+1 if safe else 0; stable |= streak>=15
                require((row['weak_reached'],row['recovered'],row['stable_recovery'])==(bool(weak),recovered,stable),'closed recovery mismatch')
    require(set(groups)==expected,'unexpected scientific episode')
    count=48+len(validation)+len(closed); seconds=72+1.5*len(validation)+6*len(closed)
    require((result['cases'],result['simulated_seconds'],result['neural_steps'])==(count,seconds,int(seconds*2000)),'budget accounting mismatch')
    require(count<=192 and seconds<=504 and result['neural_steps']<=1008000,'budget exceeded')
    require(result['optimizer_updates']==0 and not result['retry_authorized'],'unauthorized learning/retry')
    return dict(status='audit-passed',experiment_status=result['status'],cases=count,
                intervals=intervals,baseline_intervals=baseline_intervals,
                paired_neural_steps=result['neural_steps'],audit_neural_steps=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('directory',type=Path)
    print(json.dumps(audit(parser.parse_args().directory),indent=2))
