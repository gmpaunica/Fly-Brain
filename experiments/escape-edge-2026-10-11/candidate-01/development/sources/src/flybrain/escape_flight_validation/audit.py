"""Independent saved-record temporal/physical audit; never advances neurons."""
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
from flybrain.flight3d.anatomy import digest,load_manifest
from flybrain.escape_recovery.audit import temporal_audit
from flybrain.turnflee.audit import audit_counts
from flybrain.escape_diagnostic_recovery.runner import verify_hashes
from .acceptance import verify_exception


def require(condition:bool,reason:str) -> None:
    if not condition:raise ValueError(reason)


def audit_temporal(records:list[dict],manifest:dict) -> tuple[float|None,str|None]:
    """Retained independent count arithmetic plus observed trigger reconciliation."""
    trigger,side=temporal_audit(records,manifest)
    cumulative_triggers=0;first=None;selected=None;latest=None
    for r in records:
        if r['temporal']['committed']:
            cumulative_triggers+=1
            latest=r['temporal']['side']
            if first is None:first=r['end_seconds'];selected=r['temporal']['side']
        require(r['triggers']==cumulative_triggers,'recorded trigger count mismatch')
        require(not r['triggers'] or r['selected_side']==latest,'recorded selected side mismatch')
    require((trigger,side)==(first,selected),'temporal onset mismatch')
    return trigger,side


def reconstruct_physical(case:dict,trigger:float|None,side:str|None,records:list[dict]) -> dict:
    """Scalar reconstruction independent of the runner's numpy metric function."""
    from .metrics import validate_measurements
    for r in records:validate_measurements(r)
    collision=any(r['collided'] for r in records)
    eligible=[r for r in records if trigger is not None and r['end_seconds']>=trigger-1e-12]
    empty=dict(measured_turn=False,measured_takeoff=case['state']!='resting',separation_gain_m=0.,
        measured_braking=False,collision=collision,weak_reached=False,recovered=False,
        stable_recovery=False,physical_success=False,turn_seconds=None,first_flee_seconds=None,
        weak_seconds=None,takeoff_seconds=None,braking_seconds=None,max_stable_recovery_intervals=0,
        pre_command_yaw=None,initial_predator_separation_m=None)
    if not eligible:return empty
    start=eligible[0];anchor=start['pre_orientation'][2];sign=1 if side=='right' else -1
    distance=lambda a,b:math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
    original_distance=distance(start['pre_position'],start['predator_position'])
    flee_index=next((i for i,r in enumerate(eligible) if r['escape_state']=='fleeing'),None)
    turn_candidates=eligible if flee_index is None else eligible[:flee_index+1]
    turn_events=[]
    for r in turn_candidates:
        if r['escape_state'] not in ('turning','fleeing'):continue
        yaw=r['pre_orientation'][2] if r['escape_state']=='fleeing' else r['orientation'][2]
        if abs(sign*(yaw-anchor)-math.pi/2)<=math.radians(3)+1e-12:turn_events.append(r['end_seconds'])
    weak_index=next((i for i,r in enumerate(eligible) if r['threat']['weak_continuous']),None)
    before_weak=eligible if weak_index is None else eligible[:weak_index]
    separation=max([0.]+[distance(r['position'],r['predator_position'])-original_distance for r in before_weak])
    takeoff=case['state']!='resting';takeoff_time=None;takeoff_run=0
    stable_run=0;max_stable=0;seen_recovery=False;brake_events=[]
    for i,r in enumerate(eligible):
        if case['state']=='resting':
            takeoff_run=takeoff_run+1 if not r['foot_contact'] and r['position'][2]>=.008-1e-12 else 0
            if takeoff_run==3 and takeoff_time is None:takeoff_time=r['end_seconds'];takeoff=True
        vx,vy,vz=r['velocity']
        if weak_index is not None and i>=weak_index and r['escape_state'] in ('braking','idle') and math.hypot(vx,vy)<.02 and abs(vz)<.02:
            brake_events.append(r['end_seconds'])
        seen_recovery=seen_recovery or r['recovered']
        ok=(seen_recovery and r['escape_state']=='idle' and r['behavior_state']=='cruising'
            and not r['collided'] and .075<=r['position'][2]<=.125 and abs(vz)<.02 and r['triggers']==1)
        stable_run=stable_run+1 if ok else 0;max_stable=max(max_stable,stable_run)
    passed=bool(turn_events and flee_index is not None and takeoff and separation>=.02-1e-12
        and not collision and weak_index is not None and brake_events and seen_recovery and max_stable>=15)
    return dict(measured_turn=bool(turn_events),measured_takeoff=takeoff,separation_gain_m=separation,
        measured_braking=bool(brake_events),collision=collision,weak_reached=weak_index is not None,
        recovered=bool(seen_recovery),stable_recovery=max_stable>=15,physical_success=passed,
        turn_seconds=turn_events[0] if turn_events else None,
        first_flee_seconds=eligible[flee_index]['end_seconds'] if flee_index is not None else None,
        weak_seconds=eligible[weak_index]['end_seconds'] if weak_index is not None else None,
        takeoff_seconds=takeoff_time,braking_seconds=brake_events[0] if brake_events else None,
        max_stable_recovery_intervals=max_stable,pre_command_yaw=anchor,initial_predator_separation_m=original_distance)


def close_enough(a,b) -> bool:
    if isinstance(a,float) and isinstance(b,(float,int)):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b


def audit_completed(directory:Path,config:dict,result:dict) -> dict:
    """Independently check order, all measurements, classifications and budget."""
    from .runner import case_order,protocol
    require(config['closed_loop']==case_order(),'case ordering mismatch')
    require(all(config[k]==v for k,v in protocol().items()),'predeclared physical protocol mismatch')
    require(len(result['closed_loop'])==48,'closed case count mismatch')
    key=lambda r:(r['seed'],r['state'],r['intervention'])
    groups=defaultdict(list);seen=[]
    with gzip.open(directory/'traces.jsonl.gz','rt') as trace,(directory/'traces.csv').open(newline='') as table:
        reader=csv.DictReader(table)
        for line in trace:
            r=json.loads(line);row=next(reader,None)
            require(row is not None and all(row[k]==('' if r.get(k) is None else str(r[k])) for k in reader.fieldnames),'CSV/JSON mismatch')
            require(r['phase']=='closed-loop' and r['candidate']==-1,'unexpected scientific phase')
            k=key(r)
            if not seen or seen[-1]!=k:seen.append(k)
            groups[k].append(r)
        require(next(reader,None) is None,'extra CSV interval')
    require(seen==[key(c) for c in case_order()] and len(groups)==48,'trace case ordering mismatch')
    manifest=load_manifest();reconstructed=[];intervals=0
    for case,saved in zip(config['closed_loop'],result['closed_loop']):
        require(all(saved[k]==v for k,v in case.items()),'summary case mismatch')
        records=groups[key(case)];intervals+=len(records)
        require(len(records)==300,'closed duration mismatch')
        require(all(all(r[k]==v for k,v in case.items()) for r in records),'trace case identity mismatch')
        counts=audit_counts(records,manifest);trigger,side=audit_temporal(records,manifest)
        require(counts['finite'] and counts['max_downstream_hz']<=240,'neural health mismatch')
        if case['intervention']=='visual-silent':
            vision=[i for i,n in enumerate(manifest['nodes']) if 'vision' in n['roles']]
            require(all(r[field][i]==0 for r in records for field in ('actual_counts','reference_counts') for i in vision),'visual-silent intervention mismatch')
        onset=next((r['end_seconds'] for r in records if r['threat']['strong']),None)
        angle=case['azimuth']%360
        correct=(side in ('left','right') and (angle in (0,180) or side==('left' if 0<angle<180 else 'right')))
        timely=bool(trigger is not None and onset is not None and 0<=trigger-onset<=.5+1e-12 and correct)
        metrics=reconstruct_physical(case,trigger,side,records)
        row=dict(**case,first_strong_seconds=onset,trigger_seconds=trigger,side=side,
            triggers=records[-1]['triggers'],finite=counts['finite'],max_downstream_hz=counts['max_downstream_hz'],
            timely_correct_side=timely,**metrics,success=bool(timely and metrics['physical_success']))
        require(all(k in saved and close_enough(v,saved[k]) for k,v in row.items()),'physical summary mismatch')
        require(saved['baseline_equal'] is None,'new motion-only baseline claim')
        action=directory/'actions'/f'{case["seed"]}-{case["state"]}-{case["intervention"]}.json'
        envelope=json.loads(action.read_text());original=action.parent/envelope['recovery_log']
        require(envelope['case']==case and envelope['candidate_identity']==config['candidate_identity'],'action envelope mismatch')
        require(digest(original)==envelope['recovery_log_sha256'],'action hash mismatch')
        actions=json.loads(original.read_text())
        require(actions['intervention']==case['intervention'] and actions['seed']==case['seed']
            and actions['candidate']==config['candidate'] and actions['candidate_identity']==config['candidate_identity']
            and actions['dt_seconds']==.02 and len(actions['steps'])==300,'action configuration mismatch')
        reconstructed.append(row)
    require(intervals==14400,'interval budget mismatch')
    progress=json.loads((directory/'progress.json').read_text());require(progress['limit']==576000,'reservation limit mismatch')
    with (directory/progress['journal']).open() as f:
        reservations=[json.loads(line) for line in f]
    require(len(reservations)==14400 and all(r==dict(reserved_steps=40*(i+1),limit=576000) for i,r in enumerate(reservations)),'reservation journal mismatch')
    intact=[r for r in reconstructed if r['intervention']=='intact'];silent=[r for r in reconstructed if r['intervention']=='visual-silent']
    successes=sum(r['success'] for r in intact)
    per_state={s:sum(r['success'] for r in intact if r['state']==s) for s in ('resting','hovering','cruising')}
    triggers=sum(r['triggers'] for r in silent)
    passed=successes>=22 and all(v>=7 for v in per_state.values()) and triggers==0
    status='passed' if passed else 'closed-physical-failed'
    expected=dict(status=status,cases=48,recorded_intervals=14400,reserved_kernel_steps=576000,
        simulated_seconds=288.,intact_successes=successes,per_state=per_state,silent_triggers=triggers,
        optimizer_updates=0,retry_authorized=False)
    require(all(result.get(k)==v for k,v in expected.items()),'terminal physical gate/budget mismatch')
    return dict(status='audit-passed',experiment_status=status,cases=48,intervals=14400,neural_steps=576000,
        intact_successes=successes,per_state=per_state,silent_triggers=triggers,audit_steps=0)


def audit(directory:Path) -> dict:
    """Validate terminal seals, complete output inventories and frozen sources."""
    receipt=json.loads((directory/'receipt.json').read_text());result=json.loads((directory/'results.json').read_text())
    seal=json.loads((directory.parent/'flight-validation-seal.json').read_text())
    require(seal['receipt_sha256']==digest(directory/'receipt.json'),'receipt seal mismatch')
    verify_hashes({str(directory/n):h for n,h in receipt['outputs'].items()})
    require({p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name!='receipt.json'}==set(receipt['outputs']),'saved output inventory mismatch')
    require(receipt['status']==result['status']==seal['status'],'terminal status mismatch')
    config=json.loads((directory/'config.json').read_text()) if (directory/'config.json').exists() else None
    if config:
        candidate=verify_exception(config['exception'])
        require(candidate.identity==receipt['candidate_identity']==config['candidate_identity'],'candidate receipt mismatch')
        require(config['candidate']==json.loads(json.dumps(__import__('dataclasses').asdict(candidate))),'candidate config mismatch')
        bound={**dict(candidate.source_sha256),**config['source_sha256']}
        verify_hashes(bound);verify_hashes({str(directory/'sources'/n):h for n,h in bound.items()})
        verify_hashes(config['protected'])
        require(digest(directory/'circuit.json')==candidate.response.circuit_sha256,'archived circuit mismatch')
        from .runner import source_hashes
        require(config['source_sha256']==source_hashes(),'adjacent source inventory mismatch')
    if result['status'] in ('failed','interrupted'):
        return dict(status='closed-incomplete',experiment_status=result['status'],reserved_kernel_steps=result['reserved_kernel_steps'],audit_steps=0)
    require(config is not None,'missing configuration')
    return audit_completed(directory,config,result)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    print(json.dumps(audit(parser.parse_args().directory),indent=2))
