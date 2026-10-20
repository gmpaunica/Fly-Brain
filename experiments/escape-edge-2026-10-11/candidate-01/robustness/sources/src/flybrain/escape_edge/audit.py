"""Saved-only independent reconstruction; no brain or world is advanced."""
import csv
import gzip
import json
import math
from pathlib import Path
import numpy as np

from flybrain.escape_diagnostic_recovery.runner import write, verify_hashes
from flybrain.escape_flight_validation.audit import close_enough, require
from flybrain.escape_recovery.audit import temporal_audit
from flybrain.flight3d.anatomy import digest, load_manifest
from flybrain.turnflee.audit import audit_counts
from .validation import cases, INTERVALS, LIMIT
from .session import source_hashes


def reconstruct_motion(case: dict, trigger: float | None, side: str | None,
                       records: list[dict]) -> dict:
    """Recompute trajectory gates directly from arrays, independent of the scorer."""
    xyz = np.asarray([r['position'] for r in records])
    before = np.asarray([r['pre_position'] for r in records])
    velocity = np.asarray([r['velocity'] for r in records])
    predators = np.asarray([r['predator_position'] for r in records])
    require(xyz.shape == before.shape == velocity.shape == predators.shape == (len(records),3), 'motion shape')
    require(all(np.isfinite(v).all() for v in (xyz,before,velocity,predators)), 'nonfinite motion')
    collision = any(r['collided'] for r in records)
    start = next((i for i,r in enumerate(records) if trigger is not None and r['end_seconds']+1e-12 >= trigger), None)
    turn = False
    takeoff = case['state'] != 'resting'
    separation = 0.
    reversal = premature = unsafe = False
    arrival = None
    longest = 0
    if start is not None:
        yaw0 = records[start]['pre_orientation'][2]
        flee = next((i for i in range(start,len(records)) if records[i]['escape_state'] != 'turning'), None)
        if flee is not None:
            signed_yaw = (records[flee]['pre_orientation'][2]-yaw0)*(1 if side=='right' else -1)
            turn = abs(signed_yaw-math.pi/2) <= math.radians(3)+1e-12
        theta = yaw0+(math.pi/2 if side=='right' else -math.pi/2)
        direction = np.array([math.sin(theta),math.cos(theta)])
        gain = np.linalg.norm(xyz[start:]-predators[start:],axis=1)-np.linalg.norm(before[start]-predators[start])
        separation = max(0.,float(gain.max()))
        airborne = [not r['foot_contact'] and r['position'][2] >= .008 for r in records[start:]]
        takeoff |= any(all(airborne[i:i+3]) for i in range(len(airborne)-2))
        wall = np.minimum(xyz[:,:2], .6-xyz[:,:2]).min(axis=1)
        speed = np.linalg.norm(velocity[:,:2],axis=1)
        stable = 0
        for i in range(start,len(records)):
            r = records[i]
            moving = r['escape_state'] in ('fleeing','edge_braking')
            if moving:
                reversal |= bool(((xyz[i]-before[i])[:2]@direction) < -.002)
                if flee is not None and r['end_seconds'] > records[flee]['end_seconds']+.4 and wall[i] > .06:
                    premature |= bool(speed[i] < .04)
            hovering = r['escape_state']=='edge_hover'
            safe = (hovering and .005 <= wall[i] <= .025 and speed[i] <= .01
                    and abs(xyz[i,2]-.1) <= .015 and not r['collided'])
            unsafe |= hovering and not safe
            stable = stable+1 if safe else 0
            longest = max(longest,stable)
            if safe and arrival is None:
                arrival = r['end_seconds']
    passed = bool(turn and takeoff and separation >= .02-1e-12 and not collision
        and not reversal and not premature and not unsafe and arrival is not None
        and arrival <= 10.+1e-12 and longest >= 100)
    return dict(measured_turn=turn, measured_takeoff=bool(takeoff), separation_gain_m=separation,
        collision=collision, premature_stop=premature, inward_reversal=reversal,
        arrival_seconds=arrival, stable_hover_seconds=longest*.02,
        hover_unsafe=bool(unsafe), physical_success=passed)


def check_case(case: dict, records: list[dict], saved: dict, manifest: dict) -> None:
    """Reconcile original paired count/readout arithmetic and actual edge motion."""
    require(len(records)==INTERVALS, 'case duration mismatch')
    require(all(all(r[k]==v for k,v in case.items()) for r in records), 'case identity mismatch')
    counts = audit_counts(records,manifest)
    trigger,side = temporal_audit(records,manifest)
    require(counts['finite'] and counts['max_downstream_hz'] <= 240, 'neural health')
    onset = next((r['end_seconds'] for r in records if r['threat']['strong']),None)
    az = case['azimuth']%360
    correct = side in ('left','right') and (az in (0,180) or side==('left' if 0<az<180 else 'right'))
    timely = bool(trigger is not None and onset is not None and 0 <= trigger-onset <= .5+1e-12 and correct)
    motion = reconstruct_motion(case,trigger,side,records)
    expected = dict(**case,trigger_seconds=trigger,side=side,first_strong_seconds=onset,
        timely_correct_side=timely,**motion,success=bool(timely and motion['physical_success']),
        triggers=records[-1]['triggers'],finite=counts['finite'],max_downstream_hz=counts['max_downstream_hz'])
    require(all(k in saved and close_enough(v,saved[k]) for k,v in expected.items()), 'case summary mismatch')
    if case['intervention']=='visual-silent':
        vision = [i for i,n in enumerate(manifest['nodes']) if 'vision' in n['roles']]
        require(all(r[k][i]==0 for r in records for k in ('actual_counts','reference_counts') for i in vision), 'silent cells')
    for r in records:
        require(not r['home_guard'] and not r['recovered'], 'unexpected home/recovery handoff')
        require(r['escape_state'] in ('idle','turning','fleeing','edge_braking','edge_hover'), 'escape state')
        if r['escape_state'] in ('edge_braking','edge_hover'):
            require(r['braking_reason']=='edge_endpoint', 'braking reason')
        if r['endpoint_estimate'] is not None:
            endpoint = np.array(r['endpoint_estimate'])
            require(endpoint.shape==(2,) and np.isfinite(endpoint).all() and (np.abs(endpoint)<=.285+1e-12).all(), 'endpoint bounds')
        require(np.isfinite(r['desired_velocity']).all() and np.isfinite(r['estimated_displacement']).all(), 'nonfinite controller state')


def audit(directory: Path) -> dict:
    """Check source archives, hashes, cases, CSV, actions and durable step budget."""
    receipt = json.loads((directory/'receipt.json').read_text())
    config = json.loads((directory/'config.json').read_text())
    result = json.loads((directory/'results.json').read_text())
    seal = json.loads((directory.parent/f'{config["phase"]}-seal.json').read_text())
    require(seal['receipt_sha256']==digest(directory/'receipt.json'), 'seal mismatch')
    require(receipt['status']==result['status']==seal['status'], 'terminal status')
    require(config['edge_sources']==source_hashes(), 'edge source inventory')
    verify_hashes(config['bound'])
    verify_hashes(config['protected'])
    verify_hashes(config['proofs'])
    for name,h in receipt['outputs'].items():
        require(digest(directory/name)==h, 'output hash mismatch')
    for name,h in config['bound'].items():
        require(digest(directory/'sources'/name)==h, 'archive hash mismatch')
    expected = cases(config['phase'],config['seed_offset'])
    require(config['cases']==expected and config['max_kernel_steps']==LIMIT and config['seconds_per_case']==12, 'protocol mismatch')
    require(config['optimizer_updates']==0 and config['gains']==[4,2], 'learning/gains changed')
    require(len(result['closed_loop'])==48 and result['cases']==48, 'incomplete results')
    with (directory/'results.csv').open(newline='') as stream:
        csv_rows = [{k:json.loads(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    require(csv_rows==result['closed_loop'], 'CSV mismatch')
    manifest = load_manifest()
    for index,(case,row) in enumerate(zip(expected,result['closed_loop'])):
        with gzip.open(directory/'traces'/f'{index:03}.jsonl.gz','rt',encoding='utf-8') as stream:
            records = [json.loads(line) for line in stream]
        check_case(case,records,row,manifest)
        actions = json.loads((directory/'actions'/f'{index:03}.json').read_text())
        require(actions['schema']=='escape-edge-actions-v1' and actions['seed']==case['seed']
                and actions['intervention']==case['intervention'] and actions['initial_state']==case['state']
                and actions['candidate_identity']==config['candidate_identity']
                and actions['candidate']==config['candidate'] and actions['edge_sources']==config['edge_sources']
                and actions['dt_seconds']==.02 and len(actions['steps'])==INTERVALS, 'action envelope')
        require(all(v==[0.,0.,0.] for v in actions['steps']), 'unexpected predator actions')
        require(len(actions['spawns'])==1 and actions['spawns'][0]['step']==0
                and actions['spawns'][0]['kind']=='predator'
                and actions['spawns'][0]['position']==records[0]['predator_position'], 'spawn mismatch')
    reservations = [json.loads(line) for line in (directory/'progress.reservations.jsonl').read_text().splitlines()]
    require(len(reservations)==48*INTERVALS and all(r['reserved_steps']==40*(i+1) and r['limit']==LIMIT
        for i,r in enumerate(reservations)) and result['reserved_kernel_steps']==LIMIT, 'budget mismatch')
    intact = sum(r['success'] for r in result['closed_loop'] if r['intervention']=='intact')
    silent = sum(r['triggers'] for r in result['closed_loop'] if r['intervention']=='visual-silent')
    require(result['intact_successes']==intact and result['silent_triggers']==silent, 'aggregate mismatch')
    require((result['status']=='passed')==(intact==24 and silent==0), 'gate mismatch')
    return dict(status='audit-passed',experiment_status=result['status'],cases=48,
        intact_successes=intact,silent_triggers=silent,neural_steps_added=0,
        receipt_sha256=digest(directory/'receipt.json'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    value = audit(args.directory)
    write(args.output,value)
    print(json.dumps(value,indent=2))
