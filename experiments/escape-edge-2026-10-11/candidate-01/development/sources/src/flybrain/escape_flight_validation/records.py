"""Retained closed setup, measured intervals and named-start replay envelope."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from flybrain.escape_recovery.session import RecoverySession
from flybrain.escape_recovery.config import CandidateConfig
from flybrain.retreat3d.world import PredatorInput
from flybrain.escape_diagnostic_recovery.runner import write
from .metrics import validate_measurements


def setup_case(case:dict,candidate:CandidateConfig) -> RecoverySession:
    """Use the frozen closed-case calibration; world initialization is separate."""
    session=RecoverySession(case['seed'],candidate)
    session.controller.calibrated_start(case['state'])
    if case['state']!='resting':
        session.world.position[2]=.1;session.world.foot_contact=False
        if case['state']=='cruising':session.world.velocity[1]=.06
    else:
        session.controller.idle.transition('resting');session.controller.idle.deadline=10.
    az=math.radians(case['azimuth']);origin=session.world.position.copy()
    predator=(origin[0]+.075*math.sin(az),origin[1]+.075*math.cos(az),max(.021,origin[2]))
    session.spawn('predator',tuple(float(v) for v in predator))
    session.set_intervention(case['intervention'])
    return session


def advance_record(session:RecoverySession,case:dict,interval:int,predator:PredatorInput|None=None) -> dict:
    """Record geometry outside control; neural/sensory/policy logic is retained."""
    pre_position=session.world.position.tolist();pre_orientation=session.world.orientation.tolist()
    pre_velocity=session.world.velocity.tolist();pre_contact=session.world.foot_contact
    eyes_hash=hashlib.sha256(session.world.observe().eyes.tobytes()).hexdigest()
    snapshot=session.advance(predator);threat=session.threat.value;escape=session.controller.escape
    record=dict(**case,phase='closed-loop',candidate=-1,interval=interval,end_seconds=(interval+1)*.02,
        actual_counts=session.brain.last_counts.tolist(),reference_counts=session.pair.reference.last_counts.tolist(),
        wing_vector=session.pair.last_vector.tolist(),wing_scores=asdict(session.scores(session.pair.last_vector)),
        threat=asdict(threat),selected_side=escape.side,escape_state=escape.state,triggers=escape.trigger_count,
        pre_position=pre_position,pre_orientation=pre_orientation,pre_velocity=pre_velocity,pre_contact=pre_contact,
        position=list(snapshot.position),velocity=list(snapshot.velocity),orientation=list(snapshot.orientation),
        foot_contact=snapshot.foot_contact,behavior_state=snapshot.behavior_state,
        predator_position=list(snapshot.object_position),command=asdict(snapshot.command),
        collided=snapshot.crashed,recovered=escape.recovered,home_guard=escape.home_guard,
        temporal=asdict(session.readout.last),pathway=asdict(session.brain.diagnostic),
        finite=session.pair.finite,max_downstream_hz=session.pair.max_downstream_hz,eyes_sha256=eyes_hash)
    validate_measurements(record)
    if not record['finite'] or record['max_downstream_hz']>240:raise ValueError('neural health gate failed')
    return record


def store_interval(trace,writer,record:dict) -> None:
    """Finite chronological JSON and matching CSV; fail before missing data flush."""
    validate_measurements(record)
    if 'command' not in record or not all(math.isfinite(v) for v in record['command'].values()):
        raise ValueError('missing/nonfinite command measurement')
    line=json.dumps(record,allow_nan=False,separators=(',',':'))
    trace.write(line+'\n');trace.flush()
    writer.writerow({k:record.get(k) for k in writer.fieldnames})


def save_actions(session:RecoverySession,case:dict,path:Path) -> None:
    """Preserve original actions beside an immutable calibrated-start envelope."""
    original=path.with_suffix('.recovery.json');session.log.save(original)
    from flybrain.flight3d.anatomy import digest
    write(path,dict(schema='escape-flight-actions-v1',case=case,candidate_identity=session.candidate.identity,
        recovery_log=original.name,recovery_log_sha256=digest(original)))


def load_actions(path:Path) -> tuple[dict,CandidateConfig,list[PredatorInput]]:
    """Validate the envelope and original intervention/config/spawn binding."""
    from flybrain.flight3d.anatomy import digest
    from flybrain.turnflee.protocol import closed_cases
    envelope=json.loads(path.read_text());case=envelope['case']
    if envelope['schema']!='escape-flight-actions-v1' or case not in closed_cases():raise ValueError('closed action identity mismatch')
    original=path.parent/envelope['recovery_log']
    if original.parent!=path.parent or digest(original)!=envelope['recovery_log_sha256']:raise ValueError('closed action hash mismatch')
    r=json.loads(original.read_text());candidate=CandidateConfig.from_record(r['candidate'])
    if (r['schema']!='escape-recovery-actions-v1' or r['dt_seconds']!=.02
        or r['seed']!=case['seed'] or r['intervention']!=case['intervention']
        or candidate.identity!=r['candidate_identity'] or candidate.identity!=envelope['candidate_identity']):
        raise ValueError('closed action configuration mismatch')
    expected=setup_case(case,candidate)
    if r['spawns']!=json.loads(json.dumps([asdict(v) for v in expected.log.spawns])):
        raise ValueError('closed action start mismatch')
    return case,candidate,[PredatorInput(tuple(v)) for v in r['steps']]


def replay_actions(path:Path) -> RecoverySession:
    """Restore named world/controller starts then apply the retained actions."""
    case,candidate,actions=load_actions(path);session=setup_case(case,candidate)
    for action in actions:session.advance(action)
    return session
