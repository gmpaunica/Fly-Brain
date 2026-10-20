"""One fixed readout experiment; source-locked, bounded, no scientific retries."""
import argparse
from dataclasses import asdict
import csv
import gzip
import json
import hashlib
from pathlib import Path
import platform
import importlib.metadata
import numpy as np
import torch
from flybrain.flight3d.anatomy import CACHE,load_manifest,digest
from flybrain.flight3d.contracts import FlightObservation
from flybrain.encounters3d.scene import SceneSpec,encounter_eyes
from flybrain.retreat3d.controller import KinematicEstimate
from flybrain.retreat3d.threat import ThreatMonitor
from flybrain.turnflee.control import EscapePolicy,WingScores,WingResponse
from flybrain.turnflee.neural import ResponsePair
from flybrain.turnflee.experiment import presentation_success,write_interval as original_write_interval
from flybrain.escape_diagnostic_journal import ReservationJournal
from flybrain.escape_diagnostic_recovery.runner import verify_hashes,write
from .config import CandidateConfig,POLICY
from .readout import TemporalReadout
from .session import RecoverySession
from .protocol import cases,closed_cases,validation_gate,closed_gate,closed_success,trajectory_metrics

RUN=Path('experiments/escape-recovery-2026-10-07')
ART=Path('artifacts/escape-recovery-2026-10-07')


def write_interval(trace,writer,record:dict) -> None:
    """Flush each immutable interval; no progress-file replacement."""
    original_write_interval(trace,writer,record);trace.flush()


def health(pair:ResponsePair) -> None:
    """Stop immediately on nonfinite model state or sustained downstream excess."""
    if not pair.finite or not pair.stable:raise ValueError('neural health gate failed')


def validate_history(record:dict,old:dict) -> None:
    """Only decoder/actuator outcomes may change in fixed-body presentations."""
    fields=('phase','azimuth','elevation','profile','seed','interval','end_seconds','eyes_sha256',
            'actual_counts','reference_counts','wing_vector','wing_scores','threat','finite','max_downstream_hz','pathway')
    actual=json.loads(json.dumps({k:record[k] for k in fields},allow_nan=False))
    if actual!={k:old[k] for k in fields}:raise ValueError('unchanged-model historical mismatch')


def consume(directory:Path) -> None:
    if directory.exists():raise FileExistsError(directory)
    directory.parent.mkdir(parents=True,exist_ok=True)
    write(directory.parent/'escape-recovery-once.json',dict(directory=str(directory.resolve()),consumed=True,retry_authorized=False))
    directory.mkdir()


def historical_records() -> dict[int,list[dict]]:
    selected={c['seed']:[] for c in cases('validation')}
    with gzip.open('experiments/turnflee-2026-10-06/traces.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['phase']=='validation' and r['candidate']==1:selected[r['seed']].append(r)
    if any(len(r)!=75 for r in selected.values()):raise ValueError('historical coverage missing')
    return selected


def preflight() -> dict:
    evidence=json.loads((ART/'acceptance.json').read_text())
    if evidence['status']!='passed':raise ValueError('engineering preflight missing')
    verify_hashes(evidence['source_hashes']);verify_hashes(evidence['verification_hashes'])
    candidate=CandidateConfig.create();candidate.verify()
    protected=json.loads((ART/'protected.json').read_text());verify_hashes(protected)
    return dict(schema='escape-recovery-experiment-v1',candidate=asdict(candidate),candidate_identity=candidate.identity,
        hypothesis='Cumulative positive spike evidence with two-sample agreement yields reliable direction and retained turn/flee/brake/recovery.',
        seed=1100000,python=platform.python_version(),platform=platform.platform(),device='cpu',commit=None,
        software={n:importlib.metadata.version(n) for n in ('torch','numpy','panda3d','pygame')},
        cells=589,anatomical_edges=4139,optimizer_updates=0,learning_rate=None,checkpoint=None,
        max_cases=144,max_seconds=432,max_kernel_steps=864000,validation=cases('validation'),closed_loop=closed_cases(),
        presentation_seconds=1.5,baseline_seconds=.5,closed_seconds=6,policy=POLICY,protected=protected,
        stop='first failed behavioral gate or source/health/measurement/I/O/budget failure; no retries',
        acceptance_sha256=digest(ART/'acceptance.json'))


def presentation(case: dict, pair: ResponsePair, trace, writer, candidate: int, budget, bound: dict, expected: list[dict]) -> dict:
    """Evaluator converts coordinates to RGB; control receives local senses only."""
    pair.reset(); monitor=ThreatMonitor(); scores=WingScores(pair.actual.manifest)
    policy=EscapePolicy(case['seed']); gate=TemporalReadout(pair.actual.manifest); estimate=KinematicEstimate(); estimate.altitude=.1
    row=dict(**case,candidate=candidate,first_strong_seconds=None,trigger_seconds=None,
             side=None,baseline_equal=True)
    az=math.radians(case['azimuth']); el=math.radians(case['elevation'])
    for interval in range(75):
        verify_hashes(bound);budget.reserve_interval()
        eyes=np.zeros((2,24,48,3),dtype=np.float32)
        if interval>=25:
            start,end=((.17,.08),(.12,.06))[case['profile']]
            distance=start+(end-start)*(interval-25)/49
            position=(.3+distance*math.sin(az)*math.cos(el),
                      .3+distance*math.cos(az)*math.cos(el),.1+distance*math.sin(el))
            eyes=encounter_eyes(np.array((.3,.3,.1)),np.zeros(3),SceneSpec(case['seed'],'predator',position))
        observation=FlightObservation(0.,0.,eyes,(0.,0.,0.),(0.,0.,9.81),False,False)
        vector=pair.advance(observation); response=scores(vector); threat=monitor.update(eyes)
        seconds=(interval+1)*.02
        if threat.strong and row['first_strong_seconds'] is None: row['first_strong_seconds']=seconds
        decision=gate.update(pair.actual.last_counts,pair.reference.last_counts,threat.strong,threat.weak_continuous,policy.state!='idle')
        impulse=(WingResponse(left=1. if decision.side=='left' else 0.,right=1. if decision.side=='right' else 0.,strength=1.)
                 if decision.committed else WingResponse())
        policy.command(estimate,observation,impulse,threat)
        if policy.trigger_count and row['trigger_seconds'] is None:
            row['trigger_seconds']=seconds; row['side']=policy.side
        if interval<25:
            row['baseline_equal'] &= bool(np.array_equal(pair.actual.last_counts,pair.reference.last_counts))
        record=dict(**case,candidate=candidate,interval=interval,end_seconds=seconds,
            eyes_sha256=hashlib.sha256(eyes.tobytes()).hexdigest(),
            actual_counts=pair.actual.last_counts.tolist(),reference_counts=pair.reference.last_counts.tolist(),
            wing_vector=vector.tolist(),wing_scores=asdict(response),threat=asdict(threat),
            selected_side=policy.side,escape_state=policy.state,triggers=policy.trigger_count,
            actual_wings=int(pair.actual.last_counts[list(pair.indices)].sum()),
            reference_wings=int(pair.reference.last_counts[list(pair.indices)].sum()),
            finite=pair.finite,max_downstream_hz=pair.max_downstream_hz,
            pathway=asdict(pair.actual.diagnostic),temporal=asdict(decision))
        validate_history(record,expected[interval])
        health(pair)
        write_interval(trace,writer,record)
    row.update(finite=pair.finite,max_downstream_hz=pair.max_downstream_hz)
    row['success']=presentation_success(row)
    return row


def closed_presentation(case: dict, config: CandidateConfig, trace, writer, budget, bound: dict) -> dict:
    """Six seconds; assumed named starts initialize both sides independently."""
    session=RecoverySession(case['seed'],config)
    session.controller.calibrated_start(case['state'])
    # Matched evaluator setup is separate from the controller's fixed calibration.
    if case['state']!='resting':
        session.world.position[2]=.1; session.world.foot_contact=False
        if case['state']=='cruising': session.world.velocity[1]=.06
    else: session.controller.idle.transition('resting'); session.controller.idle.deadline=10.
    az=math.radians(case['azimuth']); distance=.075
    origin=session.world.position.copy()
    predator=(origin[0]+distance*math.sin(az),origin[1]+distance*math.cos(az),max(.021,origin[2]))
    session.spawn('predator',tuple(float(v) for v in predator))
    session.set_intervention(case['intervention'])
    row=dict(**case,first_strong_seconds=None,trigger_seconds=None,side=None,
             baseline_equal=True,weak_reached=False,recovered=False,stable_recovery=False)
    recovery_steps=0;trajectory=[]
    for interval in range(300):
        verify_hashes(bound);budget.reserve_interval()
        pre_position=tuple(session.world.position);pre_orientation=tuple(session.world.orientation)
        snapshot=session.advance();health(session.pair); seconds=(interval+1)*.02
        threat=session.threat.value; escape=session.controller.escape
        if threat.strong and row['first_strong_seconds'] is None: row['first_strong_seconds']=seconds
        if escape.trigger_count and row['trigger_seconds'] is None:
            row['trigger_seconds']=seconds; row['side']=escape.side
        if escape.trigger_count and threat.weak_continuous: row['weak_reached']=True
        if escape.recovered: row['recovered']=True
        stable=(row['recovered'] and escape.state=='idle' and not snapshot.crashed
                and .075<=snapshot.position[2]<=.125 and abs(snapshot.velocity[2])<.02)
        recovery_steps=recovery_steps+1 if stable else 0
        row['stable_recovery']=row['stable_recovery'] or recovery_steps>=15
        record=dict(**case,phase='closed-loop',candidate=-1,interval=interval,
            end_seconds=seconds,actual_counts=session.brain.last_counts.tolist(),
            reference_counts=session.pair.reference.last_counts.tolist(),
            wing_vector=session.pair.last_vector.tolist(),wing_scores=asdict(session.scores(session.pair.last_vector)),
            threat=asdict(threat),selected_side=escape.side,escape_state=escape.state,
            triggers=escape.trigger_count,position=snapshot.position,velocity=snapshot.velocity,
            collided=snapshot.crashed,recovered=escape.recovered,home_guard=escape.home_guard,
            finite=session.pair.finite,max_downstream_hz=session.pair.max_downstream_hz,
            pre_position=pre_position,pre_orientation=pre_orientation,orientation=snapshot.orientation,foot_contact=snapshot.foot_contact,behavior_state=snapshot.behavior_state,
            predator_position=snapshot.object_position,command=asdict(snapshot.command),temporal=asdict(session.readout.last),
            pathway=asdict(session.brain.diagnostic))
        trajectory.append(record);write_interval(trace,writer,record)
    row.update(triggers=escape.trigger_count,collision=session.world.crashed,finite=session.pair.finite,
               max_downstream_hz=session.pair.max_downstream_hz)
    row.update(trajectory_metrics(case,row,trajectory))
    row['success']=closed_success(row)
    actions=Path(trace.name).parent/'actions';actions.mkdir(exist_ok=True)
    session.log.save(actions/f'{case["seed"]}-{case["state"]}-{case["intervention"]}.json')
    return row


def run(directory:Path=RUN) -> dict:
    torch.set_num_threads(1);config=preflight();expected=historical_records();consume(directory)
    status='interrupted';error=None;budget=None;validation=[];closed=[];intervals=0
    try:
        write(directory/'config.json',config)
        candidate=CandidateConfig.from_record(config['candidate'])
        for name,h in candidate.source_sha256:
            p=directory/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(name).read_bytes())
            if digest(p)!=h:raise ValueError('source archive mismatch')
        (directory/'circuit.json').write_bytes((CACHE/'circuit.json').read_bytes())
        budget=ReservationJournal(directory/'progress.json',864000)
        fields=['phase','azimuth','elevation','profile','state','intervention','seed','interval','end_seconds',
                'selected_side','escape_state','triggers','max_downstream_hz']
        pair=ResponsePair(load_manifest(),candidate.response)
        with gzip.open(directory/'traces.jsonl.gz','xt',encoding='utf-8') as trace,(directory/'traces.csv').open('x',newline='',encoding='utf-8') as table:
            writer=csv.DictWriter(table,fieldnames=fields);writer.writeheader()
            for case in config['validation']:
                validation.append(presentation(case,pair,trace,writer,0,budget,dict(candidate.source_sha256),expected[case['seed']]))
                intervals+=75
                print(f'direction {len(validation)}/96 | correct {sum(r["success"] for r in validation)}',flush=True)
            if validation_gate(validation):
                for case in config['closed_loop']:
                    closed.append(closed_presentation(case,candidate,trace,writer,budget,dict(candidate.source_sha256)))
                    intervals+=300
                    print(f'closed {len(closed)}/48 | intact {sum(r["success"] for r in closed if r["intervention"]=="intact")}',flush=True)
            status='closed-direction-failed' if not validation_gate(validation) else 'passed' if closed_gate(closed) else 'closed-physical-failed'
        verify_hashes(config['protected']);candidate.verify()
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
        raise
    finally:
        if budget:budget.close()
        result=dict(status=status,error=error,validation=validation,closed_loop=closed,cases=len(validation)+len(closed),
            recorded_intervals=intervals,reserved_kernel_steps=budget.steps if budget else 0,
            validation_successes=sum(r['success'] for r in validation),
            intact_successes=sum(r['success'] for r in closed if r['intervention']=='intact'),
            silent_triggers=sum(r['triggers'] for r in closed if r['intervention']=='visual-silent'),retry_authorized=False)
        write(directory/'results.json',result)
        outputs={p.relative_to(directory).as_posix():digest(p) for p in directory.rglob('*') if p.is_file()}
        write(directory/'receipt.json',dict(status=status,candidate_identity=config.get('candidate_identity'),outputs=outputs,retry_authorized=False))
        write(directory.parent/'escape-recovery-seal.json',dict(receipt_sha256=digest(directory/'receipt.json'),status=status))
    return result


if __name__=='__main__':
    r=run();print(json.dumps({k:r[k] for k in ('status','cases','reserved_kernel_steps','validation_successes','intact_successes','silent_triggers')},indent=2))
