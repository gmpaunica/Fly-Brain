"""One source-locked experiment; consume once, close at the first failed gate."""
import argparse
from dataclasses import asdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import numpy as np
import torch
from flybrain.flight3d.anatomy import CACHE, digest, load_manifest
from flybrain.flight3d.contracts import FlightObservation
from flybrain.encounters3d.scene import SceneSpec, encounter_eyes
from flybrain.retreat3d.controller import KinematicEstimate
from flybrain.retreat3d.threat import ThreatMonitor
from .control import EscapePolicy, WingScores
from .neural import GAIN_PAIRS, ResponseConfig, ResponsePair, source_hashes
from .protocol import cases, closed_cases, correct_side, select_candidate, validation_gate, closed_gate, history_audit
from .session import TurnFleeSession


def write_json(path: Path, value) -> None:
    """Fresh-only records, finite JSON; refuse silent record replacement."""
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def lock_run(directory: Path) -> dict:
    """A parent-scoped consumed receipt prevents renamed-directory retries."""
    if directory.exists(): raise FileExistsError(directory)
    directory.parent.mkdir(parents=True,exist_ok=True)
    history=history_audit()
    write_json(directory.parent/'turnflee-once.json',dict(directory=str(directory.resolve()),
               consumed=True,retry_authorized=False))
    directory.mkdir()
    manifest=load_manifest()
    configs=[asdict(ResponseConfig.create(manifest,g)) for g in GAIN_PAIRS]
    config=dict(schema='turnflee-experiment-v1',
        hypothesis='Two declared effective gains on retained visual/DN edges produce timely correctly signed steering spikes sufficient for engineered turn-and-flee.',
        max_cases=192,max_seconds=504,max_neural_steps=1008000,
        stop='first failed gate; no extra candidates, threshold changes, tuning or retry',
        gain_pairs=GAIN_PAIRS,candidates=configs,calibration=cases('calibration'),
        validation=cases('validation'),closed_loop=closed_cases(),history_audit=history,
        presentation_seconds=1.5,baseline_seconds=.5,rolling_window_seconds=.2,
        control_seconds=.02,neural_seconds=.001,pending_seconds=.5,
        downstream_guard_hz=240,validation_min=90,azimuth_min=5,
        closed_min=22,state_min=7,silent_triggers_max=0,
        wing_readout='unit signed i1/b3 same-side + b1 opposite-side; six-cell increment norm >=1; seeded bilateral exact tie',
        response_latency='validation/closed-loop: first trigger within 500 ms of first strong-threat onset; calibration: correct gated response during presentation; <=15 degrees front/rear accepts either side',
        flight_policy=dict(turn_degrees=90,speed_m_s=.12,altitude_m=.1,brake_speed_m_s=.02,
                           home_outer_m=.22,home_inner_m=.18),
        closed_start_assumptions='named-state calibration: upright heading zero; rest at leg height, hover at nominal .1m, cruise at nominal .1m and .06m/s forward; no target/body coordinates transferred',
        recovery_rule='weak continuity, speed brake, idle cruise resumed and maintained >=300ms without new trigger/collision; altitude .075..125m',
        seed=1100000,python=platform.python_version(),platform=platform.platform(),device='cpu',
        software={n:importlib.metadata.version(n) for n in ('torch','numpy','panda3d','pygame')},
        network_cells=len(manifest['nodes']),retained_edges=len(manifest['edges']),
        commit=None,optimizer_updates=0,learning_rate=None,checkpoint=None,
        circuit_sha256=digest(CACHE/'circuit.json'),source_sha256=dict(source_hashes()))
    write_json(directory/'config.json',config)
    for name,value in config['source_sha256'].items():
        target=directory/'sources'/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
        if digest(target)!=value: raise ValueError('source archive mismatch')
    write_json(directory/'lock.json',dict(config_sha256=digest(directory/'config.json'),status='consumed-before-onset'))
    return config


def presentation_success(row: dict) -> bool:
    onset=row['first_strong_seconds']; trigger=row['trigger_seconds']
    return (onset is not None and trigger is not None and trigger>=onset
            and (row.get('phase')=='calibration' or trigger-onset<=.5+1e-12)
            and correct_side(row['azimuth'],row['side']) and row['baseline_equal']
            and row['finite'] and row['max_downstream_hz']<=240)


def closed_success(row: dict) -> bool:
    return (presentation_success(row) and not row['collision'] and row['weak_reached']
            and row['recovered'] and row['stable_recovery'])


def write_interval(trace, writer, record: dict) -> None:
    trace.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
    writer.writerow({k:record.get(k) for k in writer.fieldnames})


def presentation(case: dict, pair: ResponsePair, trace, writer, candidate: int) -> dict:
    """Evaluator converts coordinates to RGB; control receives local senses only."""
    pair.reset(); monitor=ThreatMonitor(); scores=WingScores(pair.actual.manifest)
    policy=EscapePolicy(case['seed']); estimate=KinematicEstimate(); estimate.altitude=.1
    row=dict(**case,candidate=candidate,first_strong_seconds=None,trigger_seconds=None,
             side=None,baseline_equal=True)
    az=math.radians(case['azimuth']); el=math.radians(case['elevation'])
    for interval in range(75):
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
        policy.command(estimate,observation,response,threat)
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
            pathway=asdict(pair.actual.diagnostic))
        write_interval(trace,writer,record)
    row.update(finite=pair.finite,max_downstream_hz=pair.max_downstream_hz)
    row['success']=presentation_success(row)
    return row


def closed_presentation(case: dict, config: ResponseConfig, trace, writer) -> dict:
    """Six seconds; assumed named starts initialize both sides independently."""
    session=TurnFleeSession(case['seed'],config)
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
    if case['intervention']=='visual-silent':
        ids={session.brain.ids[i] for i in session.brain.roles['vision']}
        session.brain.silenced=ids; session.pair.reference.silenced=ids
    row=dict(**case,first_strong_seconds=None,trigger_seconds=None,side=None,
             baseline_equal=True,weak_reached=False,recovered=False,stable_recovery=False)
    recovery_steps=0
    for interval in range(300):
        snapshot=session.advance(); seconds=(interval+1)*.02
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
        write_interval(trace,writer,dict(**case,phase='closed-loop',candidate=-1,interval=interval,
            end_seconds=seconds,actual_counts=session.brain.last_counts.tolist(),
            reference_counts=session.pair.reference.last_counts.tolist(),
            wing_vector=session.pair.last_vector.tolist(),wing_scores=asdict(session.scores(session.pair.last_vector)),
            threat=asdict(threat),selected_side=escape.side,escape_state=escape.state,
            triggers=escape.trigger_count,position=snapshot.position,velocity=snapshot.velocity,
            collided=snapshot.crashed,recovered=escape.recovered,home_guard=escape.home_guard,
            finite=session.pair.finite,max_downstream_hz=session.pair.max_downstream_hz))
    row.update(triggers=escape.trigger_count,collision=session.world.crashed,finite=session.pair.finite,
               max_downstream_hz=session.pair.max_downstream_hz)
    row['success']=closed_success(row)
    return row


def run(directory: Path) -> dict:
    """User-authorized capped execution only; never resumes an interrupted run."""
    torch.set_num_threads(1); config=lock_run(directory); manifest=load_manifest()
    candidates=[]; validation=[]; closed=[]; selected=None
    fields=['phase','candidate','azimuth','elevation','profile','seed','interval','end_seconds',
            'actual_wings','reference_wings','selected_side','escape_state','triggers','max_downstream_hz']
    with gzip.open(directory/'traces.jsonl.gz','wt',encoding='utf-8') as trace, \
         (directory/'traces.csv').open('x',newline='',encoding='utf-8') as table:
        writer=csv.DictWriter(table,fieldnames=fields); writer.writeheader()
        for index,gains in enumerate(GAIN_PAIRS):
            pair=ResponsePair(manifest,ResponseConfig.from_record(config['candidates'][index])); rows=[]
            for case in config['calibration']:
                rows.append(presentation(case,pair,trace,writer,index))
                print(f'gain {gains} calibration {len(rows)}/16 | correct {sum(r["success"] for r in rows)}',flush=True)
            candidates.append(dict(gains=gains,eligible=all(r['success'] for r in rows),
                                   successes=sum(r['success'] for r in rows),presentations=rows))
        selected=select_candidate(candidates)
        write_json(directory/'calibration.json',candidates)
        if selected is not None:
            # Immutable calibration evidence before any validation placement.
            receipt=f'{(directory/"calibration.json").as_posix()}#{digest(directory/"calibration.json")}'
            frozen=ResponseConfig.create(manifest,selected,receipt)
            write_json(directory/'response.json',asdict(frozen))
            pair=ResponsePair(manifest,frozen)
            for case in config['validation']:
                validation.append(presentation(case,pair,trace,writer,GAIN_PAIRS.index(selected)))
                print(f'validation {len(validation)}/96 | correct {sum(r["success"] for r in validation)}',flush=True)
            if validation_gate(validation):
                for case in config['closed_loop']:
                    closed.append(closed_presentation(case,frozen,trace,writer))
                    print(f'closed-loop {len(closed)}/48',flush=True)
    status=('closed-calibration-failed' if selected is None else 'closed-validation-failed'
            if not validation_gate(validation) else 'passed' if closed_gate(closed) else 'closed-loop-failed')
    count=48+len(validation)+len(closed); seconds=72+len(validation)*1.5+len(closed)*6
    result=dict(status=status,selected_gains=selected,calibration=candidates,validation=validation,
        closed_loop=closed,cases=count,simulated_seconds=seconds,neural_steps=int(seconds*2000),
        validation_successes=sum(r['success'] for r in validation),
        closed_intact_successes=sum(r['success'] for r in closed if r['intervention']=='intact'),
        silent_triggers=sum(r['triggers'] for r in closed if r['intervention']=='visual-silent'),
        optimizer_updates=0,retreat_delivered=status=='passed',retry_authorized=False)
    if count>192 or seconds>504 or result['neural_steps']>1008000: raise AssertionError('budget exceeded')
    write_json(directory/'results.json',result)
    write_json(directory/'receipt.json',dict(status=status,
        sha256={p.name:digest(p) for p in directory.iterdir() if p.is_file()}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('directory',type=Path)
    result=run(parser.parse_args().directory)
    print(json.dumps({k:result[k] for k in ('status','selected_gains','cases','simulated_seconds','neural_steps')}))
