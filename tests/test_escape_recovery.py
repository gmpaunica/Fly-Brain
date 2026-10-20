"""Temporal decision, trajectory and source/consumption behavior requirements."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api(name='readout'):
    assert importlib.util.find_spec('flybrain.escape_recovery'), 'actual behavior repair required'
    return importlib.import_module('flybrain.escape_recovery.'+name)


def readout():
    m=api(); from flybrain.flight3d.anatomy import load_manifest
    return m.TemporalReadout(load_manifest())


def counts(g,six):
    v=np.zeros(589,dtype=np.int64);v[list(g.indices)]=six;return v


def test_positive_winner_prevents_sole_deficit_and_ties_never_randomize():
    g=readout();z=counts(g,[0]*6)
    for _ in range(3):
        d=g.update(z,counts(g,[0,1,0,0,0,0]),True,False,False)
        assert d.side is None
    g.reset()
    for _ in range(3):
        d=g.update(counts(g,[1,0,0,1,0,0]),z,True,False,False)
        assert d.side is None and d.streak==0


def test_cumulative_evidence_persists_and_needs_two_post_strong_samples():
    g=readout();z=counts(g,[0]*6);a=counts(g,[0,1,0,0,0,0])
    assert g.update(a,z,False,False,False).side is None
    for _ in range(11):g.update(z,z,False,False,False)
    d=g.update(z,z,True,False,False)
    assert d.left==1 and d.streak==1 and d.side is None
    d=g.update(z,z,False,False,False)
    assert d.side=='left' and d.streak==2


def test_sign_change_resets_agreement_and_deadline_never_renews():
    g=readout();z=counts(g,[0]*6)
    assert g.update(counts(g,[0,1,0,0,0,0]),z,True,False,False).side is None
    d=g.update(counts(g,[0,0,0,0,3,0]),z,True,False,False)
    assert d.side is None and d.streak==1
    assert g.update(z,z,False,False,False).side=='right'
    g.reset()
    for _ in range(26):g.update(z,z,True,False,False)
    assert g.update(counts(g,[0,1,0,0,0,0]),z,True,False,False).side is None
    assert g.last.timed_out


def test_weak_rearm_and_reset_clear_evidence():
    g=readout();z=counts(g,[0]*6)
    g.update(counts(g,[0,1,0,0,0,0]),z,True,False,False)
    assert g.update(z,z,False,False,False).side=='left'
    g.update(z,z,False,True,True)
    g.recovered()
    d=g.update(counts(g,[5]*6),z,False,True,False)
    assert not any(d.cumulative) and d.side is None
    g.update(counts(g,[0,0,0,0,1,0]),z,True,False,False)
    assert g.update(z,z,False,False,False).side=='right'
    g.reset();assert g.tick==0 and not g.cumulative.any() and g.last.side is None


def test_rejects_missing_or_noninteger_counts():
    g=readout()
    with pytest.raises(ValueError):g.update(np.zeros(6),np.zeros(6),True,False,False)
    with pytest.raises(ValueError):g.update(np.ones(589)*.1,np.zeros(589),True,False,False)


def test_measured_closed_metrics_reject_stationary_flags_and_wrong_turn():
    m=api('protocol')
    case=dict(state='resting',azimuth=90,intervention='intact')
    row=dict(first_strong_seconds=.02,trigger_seconds=.04,side='left',finite=True,max_downstream_hz=100,baseline_equal=True)
    trajectory=[]
    for k in range(30):
        trajectory.append(dict(end_seconds=(k+1)*.02,position=(.3+.03*(k>=5),.3,.003 if k<2 else .1),orientation=(0.,0.,-np.pi/2 if k>=4 else 0.),
            velocity=(0.,0.,0.),foot_contact=k<2,predator_position=(.3,.375,.021),collided=False,behavior_state='cruising',
            pre_position=(.3,.3,.003),pre_orientation=(0.,0.,0.),
            escape_state='turning' if k<4 else 'fleeing' if k<7 else 'braking' if k<9 else 'idle',
            recovered=k==9,threat=dict(weak_continuous=k>=7),triggers=1))
    # Initial comparison at the actual trigger, later distance rises visibly.
    assert m.trajectory_metrics(case,row,trajectory)['physical_success']
    stationary=[dict(r,position=(.3,.3,.1),orientation=(0.,0.,0.)) for r in trajectory]
    assert not m.trajectory_metrics(case,row,stationary)['physical_success']
    wrong=[dict(r,orientation=(0.,0.,np.pi/2)) for r in trajectory]
    assert not m.trajectory_metrics(case,row,wrong)['physical_success']
    premature=[dict(r,orientation=(0.,0.,0.) if k<=4 else r['orientation']) for k,r in enumerate(trajectory)]
    assert not m.trajectory_metrics(case,row,premature)['measured_turn']


def test_episode_reset_replay_and_candidate_binding(tmp_path):
    m=api('session');cfg=api('config').CandidateConfig.create()
    s=m.RecoverySession(1210000,cfg)
    for _ in range(3):s.advance()
    s.spawn('predator',(.4,.3,.1))
    assert s.readout.tick==0 and not s.readout.cumulative.any()
    for _ in range(2):s.advance()
    expected=s.snapshot
    path=tmp_path/'actions.json';s.log.save(path)
    replay=m.RecoverySession.from_log(path)
    assert replay.snapshot==expected
    record=json.loads(path.read_text());record['candidate_identity']='forged';path.write_text(json.dumps(record))
    with pytest.raises(ValueError):m.RecoverySession.from_log(path)


def test_independent_temporal_audit_rejects_forged_streak():
    m=api('audit');from flybrain.flight3d.anatomy import load_manifest
    manifest=load_manifest();g=readout();zero=counts(g,[0]*6)
    a=counts(g,[0,1,0,0,0,0]);rows=[]
    literal=[dict(cumulative=[0,1,0,0,0,0],left=1,right=0,strength=1.,candidate='left',streak=1,side=None,
                  committed=False,first_strong_tick=1,deadline=26,timed_out=False),
             dict(cumulative=[0,1,0,0,0,0],left=1,right=0,strength=1.,candidate='left',streak=2,side='left',
                  committed=True,first_strong_tick=1,deadline=26,timed_out=False)]
    for k,v in enumerate((a,zero)):
        rows.append(dict(actual_counts=v.tolist(),reference_counts=zero.tolist(),end_seconds=(k+1)*.02,
                         threat=dict(strong=k==0,weak_continuous=False),temporal=literal[k]))
    assert m.temporal_audit(rows,manifest)==(.04,'left')
    rows[1]['temporal']['streak']=1
    with pytest.raises(ValueError):m.temporal_audit(rows,manifest)


def test_candidate_tamper_and_fresh_experiment_lock(tmp_path):
    cfg=api('config').CandidateConfig.create();from dataclasses import asdict
    r=asdict(cfg);r['policy']='{}'
    with pytest.raises(ValueError):api('config').CandidateConfig.from_record(r)
    e=api('experiment');e.consume(tmp_path/'one')
    with pytest.raises(FileExistsError):e.consume(tmp_path/'renamed')


def test_silent_replay_preserves_intervention_and_rejects_unordered_events(tmp_path):
    m=api('session');from dataclasses import asdict
    c=api('config').CandidateConfig.create();s=m.RecoverySession(1210001,c)
    s.spawn('predator',(.4,.3,.1));s.set_intervention('visual-silent')
    for _ in range(2):s.advance()
    p=tmp_path/'silent.json';s.log.save(p)
    restored=m.RecoverySession.from_log(p)
    assert restored.snapshot==s.snapshot and restored.intervention=='visual-silent'
    assert not restored.readout.cumulative.any()
    r=json.loads(p.read_text());r['spawns']=[dict(r['spawns'][0],step=2),dict(r['spawns'][0],step=1)]
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):m.RecoverySession.from_log(p)


def test_presentation_reaches_budget_guard_before_any_neural_step():
    import io,csv
    api('execution');e=api('experiment');from flybrain.flight3d.anatomy import load_manifest
    from flybrain.turnflee.neural import ResponsePair
    pair=ResponsePair(load_manifest(),api('config').CandidateConfig.create().response)
    class StopBudget:
        def reserve_interval(self):raise RuntimeError('before-neural-guard')
    writer=csv.DictWriter(io.StringIO(),fieldnames=['interval'])
    with pytest.raises(RuntimeError,match='before-neural-guard'):
        e.presentation(e.cases('validation')[0],pair,io.StringIO(),writer,0,StopBudget(),{},[])
    assert pair.actual.steps==pair.reference.steps==0
