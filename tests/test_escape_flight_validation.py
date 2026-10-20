"""Prospective exception, exact budget and embodied measurement requirements."""
import copy
import csv
import gzip
import importlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


def api(name):
    assert importlib.util.find_spec('flybrain.escape_flight_validation'), 'adjacent physical validation required'
    return importlib.import_module('flybrain.escape_flight_validation.' + name)


def passing_trajectory():
    """Hand-derived turn, takeoff, separation, brake and 15-tick cruise."""
    rows=[]
    for k in range(25):
        yaw=0. if k<2 else -math.pi/2
        before=0. if k<=2 else -math.pi/2
        rows.append(dict(interval=k,end_seconds=(k+1)*.02,
            pre_position=[.3,.3,.003 if k<2 else .1],pre_orientation=[0.,0.,before],
            position=[.3+(.06 if k>=4 else 0.),.3,.003 if k<2 else .1],
            orientation=[0.,0.,yaw],velocity=[0.,0.,0.],foot_contact=k<2,
            predator_position=[.3,.375,.021],collided=False,
            behavior_state='cruising',escape_state='turning' if k<3 else 'fleeing' if k<6 else 'braking' if k<8 else 'idle',
            recovered=k==8,triggers=1,selected_side='left',
            threat=dict(strong=k==0,weak_continuous=k>=6)))
    return rows


def score(rows):
    return api('metrics').physical_metrics(dict(state='resting'),.02,'left',rows)


def test_actual_movement_passes_hand_derived_trajectory():
    assert score(passing_trajectory())['physical_success']


@pytest.mark.parametrize('failure', ['stationary','wrong-turn','premature-flee','separation','recovery','contact'])
def test_labels_cannot_substitute_for_actual_movement(failure):
    rows=passing_trajectory()
    if failure=='stationary':
        for r in rows:r.update(position=[.3,.3,.003],orientation=[0.,0.,0.],pre_orientation=[0.,0.,0.])
    elif failure=='wrong-turn':
        for r in rows:r['orientation'][2]*=-1;r['pre_orientation'][2]*=-1
    elif failure=='premature-flee':
        rows[1]['escape_state']='fleeing'
    elif failure=='separation':
        for r in rows:r['position']=[.3,.3,.003 if r['interval']<2 else .008]
    elif failure=='recovery':
        for r in rows[8:]:r['velocity'][2]=.021
    elif failure=='contact':
        for r in rows:r['foot_contact']=True
    assert not score(rows)['physical_success']


def test_missing_or_nonfinite_measurements_are_hard_stops():
    rows=passing_trajectory();del rows[2]['pre_orientation']
    with pytest.raises(ValueError,match='measurement'):score(rows)
    rows=passing_trajectory();rows[2]['velocity'][0]=math.nan
    with pytest.raises(ValueError,match='finite'):score(rows)


def test_exact_cases_and_no_directional_protocol():
    cases=api('runner').case_order()
    assert len(cases)==48
    assert cases[:4]==[
        dict(azimuth=0,state='resting',intervention='intact',seed=1110000),
        dict(azimuth=0,state='resting',intervention='visual-silent',seed=1110000),
        dict(azimuth=0,state='hovering',intervention='intact',seed=1110001),
        dict(azimuth=0,state='hovering',intervention='visual-silent',seed=1110001)]
    assert cases[-1]==dict(azimuth=315,state='cruising',intervention='visual-silent',seed=1110023)
    assert 'validation' not in api('runner').protocol()


def test_exact_budget_boundary_and_parent_scoped_no_resume(tmp_path):
    m=api('runner');m.consume(tmp_path/'run')
    with pytest.raises(FileExistsError):m.consume(tmp_path/'renamed')
    budget=m.FlightBudget(tmp_path/'run'/'progress.json')
    for _ in range(14400):budget.reserve_interval()
    assert budget.steps==576000
    with pytest.raises(ValueError,match='budget'):budget.reserve_interval()
    budget.close()
    assert len((tmp_path/'run'/'progress.reservations.jsonl').read_text().splitlines())==14400


def test_source_bound_exception_keeps_original_failed_result():
    m=api('acceptance');record=m.prospective_record()
    candidate=m.verify_exception(record)
    assert candidate.identity=='ea70e22d8879874f14df4299ba797170d7624e0b26c6e0baeddf720d3494dcab'
    assert record['original_status']=='closed-direction-failed'
    assert record['known_failures']==[1100018,1100021]
    assert record['directional_successes']==94
    assert record['physical_receipt_sha256'] is None
    for key,value in [('candidate_identity','forged'),('known_failures',[]),('original_status','passed')]:
        forged=copy.deepcopy(record);forged[key]=value
        with pytest.raises(ValueError):m.verify_exception(forged)


def test_exception_rejects_tampered_saved_file(tmp_path):
    m=api('acceptance');record=m.prospective_record()
    p=tmp_path/'record.json';p.write_text('{}')
    record['bound_files'][str(p)]= '0'*64
    with pytest.raises(ValueError,match='hash'):m.verify_exception(record)


def test_closed_case_reaches_guard_without_exposure(tmp_path):
    m=api('runner');from flybrain.escape_recovery.config import CandidateConfig
    class Stop:
        def reserve_interval(self):raise RuntimeError('pre-neural guard')
    with gzip.open(tmp_path/'trace.gz','wt') as trace:
        writer=csv.DictWriter(io.StringIO(),fieldnames=m.CSV_FIELDS)
        with pytest.raises(RuntimeError,match='pre-neural guard'):
            m.closed_presentation(m.case_order()[0],CandidateConfig.create(),trace,writer,Stop(),{})
    assert not (tmp_path/'actions').exists()


def test_setup_action_storage_and_exact_silent_replay(tmp_path):
    m=api('records');from flybrain.escape_recovery.config import CandidateConfig
    c=CandidateConfig.create();case=api('runner').case_order()[3]
    session=m.setup_case(case,c)
    rows=[]
    for k in range(2):rows.append(m.advance_record(session,case,k))
    action=tmp_path/'actions.json';m.save_actions(session,case,action)
    replay=m.replay_actions(action)
    assert replay.snapshot==session.snapshot and replay.intervention=='visual-silent'
    assert replay.controller.idle.estimate.altitude==session.controller.idle.estimate.altitude
    assert rows[-1]['reference_counts']==replay.pair.reference.last_counts.tolist()
    record=json.loads(action.read_text());record['case']['state']='resting';action.write_text(json.dumps(record))
    with pytest.raises(ValueError):m.replay_actions(action)


def test_json_csv_storage_agree_and_measurement_failure_stops(tmp_path):
    m=api('records');case=api('runner').case_order()[0]
    from flybrain.escape_recovery.config import CandidateConfig
    session=m.setup_case(case,CandidateConfig.create());r=m.advance_record(session,case,0)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=api('runner').CSV_FIELDS);writer.writeheader()
    trace=io.StringIO();m.store_interval(trace,writer,r)
    table=list(csv.DictReader(io.StringIO(stream.getvalue())))
    assert json.loads(trace.getvalue())['position']==r['position']
    assert table[0]['position']==str(r['position'])
    del r['command']
    with pytest.raises(ValueError):m.store_interval(trace,writer,r)


def test_supervisor_seals_child_failure_and_retains_existing_receipt(tmp_path):
    m=api('worker');run=tmp_path/'run';run.mkdir()
    api('runner').FlightBudget(run/'progress.json').close()
    code=m.supervise([sys.executable,'-c','raise SystemExit(7)'],run,tmp_path/'child.log',tmp_path/'heartbeat.json')
    assert code==7
    receipt=(run/'receipt.json').read_bytes()
    assert json.loads(receipt)['status']=='interrupted'
    assert (run.parent/'flight-validation-seal.json').exists()
    m.seal_missing_receipt(run,'later')
    assert (run/'receipt.json').read_bytes()==receipt


def test_new_entrypoint_refuses_missing_activation_without_neural_steps():
    result=subprocess.run([sys.executable,'-m','flybrain.escape_flight_validation','--headless','--steps','0'],capture_output=True,text=True)
    assert result.returncode!=0 and 'not activated' in result.stderr.lower()


def test_saved_temporal_audit_refuses_forged_decision():
    m=api('audit');from flybrain.flight3d.anatomy import load_manifest
    rows=passing_trajectory()
    # Counts have no side evidence: a fabricated trigger cannot be accepted.
    for r in rows:r.update(actual_counts=[0]*589,reference_counts=[0]*589,temporal={})
    with pytest.raises(ValueError):m.audit_temporal(rows,load_manifest())


def test_independent_physical_reconstruction_matches_literal_events():
    result=api('audit').reconstruct_physical(dict(state='resting'),.02,'left',passing_trajectory())
    assert result['physical_success'] and result['turn_seconds']==.06
    assert result['first_flee_seconds']==.08 and result['takeoff_seconds']==.1
    assert result['weak_seconds']==.14 and result['braking_seconds']==.14
    assert result['max_stable_recovery_intervals']==17
    assert result['separation_gain_m']>.02


@pytest.mark.parametrize('condition',['intact','visual-silent'])
def test_saved_replay_compares_every_field_and_refuses_pose_tamper(tmp_path,condition):
    records=api('records');replay=api('replay')
    from flybrain.escape_recovery.config import CandidateConfig
    case=dict(azimuth=0,state='cruising',intervention=condition,seed=1110002)
    session=records.setup_case(case,CandidateConfig.create())
    rows=[records.advance_record(session,case,i) for i in range(2)]
    path=tmp_path/'actions.json';records.save_actions(session,case,path)
    assert replay.verify_case(path,rows)==80
    rows[1]['position'][0]+=.001
    with pytest.raises(ValueError,match='exact replay'):replay.verify_case(path,rows)


def test_activation_refuses_failed_physical_gate_without_writing_marker(tmp_path):
    m=api('activation');run=tmp_path/'run';run.mkdir()
    api('runner').seal(run,dict(status='closed-physical-failed',reserved_kernel_steps=0),None)
    with pytest.raises(ValueError):m.activate(run,tmp_path)
    assert not (tmp_path/'activation.json').exists()


def test_sealing_and_saved_audit_rejects_tamper(tmp_path):
    run=tmp_path/'run';run.mkdir();m=api('runner')
    m.seal(run,dict(status='interrupted',reserved_kernel_steps=0),None)
    assert api('audit').audit(run)['status']=='closed-incomplete'
    (run/'results.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):api('audit').audit(run)


def test_supervisor_start_failure_seals_zero_exposure(tmp_path):
    run=tmp_path/'run';run.mkdir()
    with pytest.raises(OSError):api('worker').supervise(['nonexistent-flybrain-preflight-command'],run,tmp_path/'child.log',tmp_path/'worker.json')
    receipt=json.loads((run/'receipt.json').read_text())
    assert receipt['status']=='interrupted' and receipt['reserved_kernel_steps']==0


def test_interval_durability_failure_poisoned_before_next_exposure(tmp_path,monkeypatch):
    budget=api('runner').FlightBudget(tmp_path/'progress.json')
    import flybrain.escape_diagnostic_journal as journal
    def fail(stream,value):raise OSError('controlled durable I/O failure')
    monkeypatch.setattr(journal,'append_durable',fail)
    with pytest.raises(OSError):budget.reserve_interval()
    assert budget.steps==0 and budget.failed
    with pytest.raises(RuntimeError):budget.reserve_interval()
    budget.close()


def test_temporal_audit_allows_measured_second_trigger():
    from dataclasses import asdict
    import numpy as np
    from flybrain.escape_recovery.readout import TemporalReadout
    from flybrain.flight3d.anatomy import load_manifest
    manifest=load_manifest();gate=TemporalReadout(manifest);zero=np.zeros(589,dtype=np.int64)
    rows=[];triggers=0;selected=None
    for tick in range(5):
        actual=zero.copy()
        if tick==0:actual[gate.indices[1]]=1
        if tick==3:actual[gate.indices[4]]=1
        decision=gate.update(actual,zero,tick in (0,3),tick==2,False)
        if decision.committed:triggers+=1;selected=decision.side
        recovered=tick==2
        rows.append(dict(actual_counts=actual.tolist(),reference_counts=zero.tolist(),end_seconds=(tick+1)*.02,
            threat=dict(strong=tick in (0,3),weak_continuous=tick==2),temporal=json.loads(json.dumps(asdict(decision))),
            recovered=recovered,triggers=triggers,selected_side=selected))
        if recovered:gate.recovered()
    assert api('audit').audit_temporal(rows,manifest)==(.04,'left')


def test_entire_runner_seals_engineering_stop_before_exposure(tmp_path,monkeypatch):
    from dataclasses import asdict
    from flybrain.escape_recovery.config import CandidateConfig
    m=api('runner');candidate=CandidateConfig.create();exception=api('acceptance').prospective_record()
    config=dict(candidate=asdict(candidate),candidate_identity=candidate.identity,source_sha256=m.source_hashes(),
        exception=exception,exception_sha256='engineering-only',acceptance_sha256='engineering-only',
        closed_loop=m.case_order(),protected={})
    monkeypatch.setattr(m,'preflight',lambda:config)
    # Real archives, setup and seal; only the budget refuses the first exposure.
    class StopBudget:
        steps=0
        def __init__(self,path):pass
        def reserve_interval(self):raise RuntimeError('engineering stop before exposure')
        def close(self):pass
    monkeypatch.setattr(m,'FlightBudget',StopBudget)
    # Evidence bindings intentionally absent: source refusal occurs even before
    # the budget. Neither failure is allowed to advance neurons.
    with pytest.raises(ValueError,match='hash mismatch'):m.run(tmp_path/'run')
    result=json.loads((tmp_path/'run'/'results.json').read_text())
    assert result['status']=='failed' and result['reserved_kernel_steps']==0 and result['cases']==0
    assert (tmp_path/'run'/'receipt.json').exists() and (tmp_path/'flight-validation-seal.json').exists()
    with pytest.raises(FileExistsError):m.run(tmp_path/'renamed')


def test_complete_saved_audit_reconstructs_failed_gate_and_refuses_forged_success(tmp_path):
    """Full storage/audit plumbing, 14,400 synthetic records, zero neural calls."""
    from dataclasses import asdict
    from flybrain.escape_recovery.config import CandidateConfig
    from flybrain.flight3d.anatomy import digest
    m=api('runner');candidate=CandidateConfig.create();config=dict(**m.protocol(),candidate=asdict(candidate),candidate_identity=candidate.identity)
    config=json.loads(json.dumps(config));actions=tmp_path/'actions';actions.mkdir()
    temporal=dict(cumulative=[0]*6,left=0,right=0,strength=0.,candidate=None,streak=0,side=None,
        committed=False,first_strong_tick=None,deadline=None,timed_out=False)
    result=dict(status='closed-physical-failed',cases=48,recorded_intervals=14400,reserved_kernel_steps=576000,
        simulated_seconds=288.,intact_successes=0,per_state=dict(resting=0,hovering=0,cruising=0),
        silent_triggers=0,optimizer_updates=0,retry_authorized=False,closed_loop=[])
    # Synthetic reservation file: no fsync/kernel is needed to audit saved data.
    (tmp_path/'progress.json').write_text(json.dumps(dict(limit=576000,journal='reservations.jsonl')))
    with (tmp_path/'reservations.jsonl').open('w') as f:
        for i in range(14400):f.write(json.dumps(dict(reserved_steps=40*(i+1),limit=576000))+'\n')
    with gzip.open(tmp_path/'traces.jsonl.gz','wt') as trace,(tmp_path/'traces.csv').open('w',newline='') as table:
        writer=csv.DictWriter(table,fieldnames=m.CSV_FIELDS);writer.writeheader()
        for case in m.case_order():
            rows=[]
            for i in range(300):
                r=dict(**case,phase='closed-loop',candidate=-1,interval=i,end_seconds=(i+1)*.02,
                    actual_counts=[0]*589,reference_counts=[0]*589,wing_vector=[0]*48,
                    finite=True,max_downstream_hz=0.,temporal=temporal,
                    pre_position=[.3,.3,.003],pre_orientation=[0.,0.,0.],
                    position=[.3,.3,.003],orientation=[0.,0.,0.],velocity=[0.,0.,0.],
                    predator_position=[.3,.375,.021],foot_contact=True,collided=False,
                    recovered=False,triggers=0,selected_side=None,escape_state='idle',behavior_state='resting',
                    threat=dict(strong=False,weak_continuous=False),command=dict(power=0.,pitch=0.,roll=0.,yaw=0.,legs=0.))
                api('records').store_interval(trace,writer,r);rows.append(r)
            result['closed_loop'].append(m.summary(case,rows))
            path=actions/f'{case["seed"]}-{case["state"]}-{case["intervention"]}.json';original=path.with_suffix('.recovery.json')
            original.write_text(json.dumps(dict(seed=case['seed'],intervention=case['intervention'],candidate=config['candidate'],
                candidate_identity=candidate.identity,dt_seconds=.02,steps=[[0.,0.,0.]]*300)))
            path.write_text(json.dumps(dict(case=case,candidate_identity=candidate.identity,recovery_log=original.name,recovery_log_sha256=digest(original))))
    checked=api('audit').audit_completed(tmp_path,config,result)
    assert checked['status']=='audit-passed' and checked['experiment_status']=='closed-physical-failed' and checked['audit_steps']==0
    result['closed_loop'][0]['success']=True
    with pytest.raises(ValueError,match='summary'):api('audit').audit_completed(tmp_path,config,result)


def test_delivery_evidence_refuses_replay_from_another_receipt(tmp_path):
    from flybrain.flight3d.anatomy import digest
    run=tmp_path/'run';run.mkdir();(run/'receipt.json').write_text('{}')
    evidence=dict(status='exact-replay-passed',cases=48,replay_neural_steps=576000,new_scientific_cases=0,
        receipt_sha256=digest(run/'receipt.json'))
    (tmp_path/'replay.json').write_text(json.dumps(evidence))
    (tmp_path/'visual-review.json').write_text(json.dumps(dict(saved_review='reviewed',receipt_sha256=digest(run/'receipt.json'),figures={})))
    api('activation').verify_delivery_evidence(run,tmp_path)
    evidence['receipt_sha256']='forged';(tmp_path/'replay.json').write_text(json.dumps(evidence))
    with pytest.raises(ValueError,match='replay'):api('activation').verify_delivery_evidence(run,tmp_path)
