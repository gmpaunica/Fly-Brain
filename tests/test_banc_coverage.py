"""Training-only coverage collection and original-gate preservation checks."""

import pytest
import torch


def test_coverage_cases_are_fixed_train_only_and_distinct_from_validation():
    from flybrain.training.banc_coverage import coverage_cases
    a=coverage_cases();assert a==coverage_cases() and len(a)==32
    assert all(c.split=='train' and 700000<=c.seed<700032 for c in a)
    assert len({c.seed for c in a})==32
    assert not {c.seed for c in a}.intersection(set(range(620000,620016))|set(range(621000,621032)))


def test_coverage_recorder_labels_student_without_replacing_its_command():
    from flybrain.training.banc_coverage import CoverageRecorder
    from flybrain.neural.banc_memory import MemoryBancReadout,make_memory_controller
    from flybrain.world.sensing import SensoryObservation
    from flybrain.training.readout_experiment import teacher_command
    m=MemoryBancReadout(torch.zeros(16),torch.ones(16));raw=make_memory_controller(m)
    recorder=CoverageRecorder(m,'test-case','student')
    observation=SensoryObservation(.02,.03)
    a=raw.decide(observation);b=recorder.decide(observation)
    assert a.command==b.command and torch.equal(a.activity.spike_counts,b.activity.spike_counts)
    row=recorder.rows[0]
    assert row['target_turn']==float(teacher_command(observation)[1])
    assert row['rates']==recorder.decoder.feature_rates.tolist()


def test_teacher_recording_uses_same_spike_features_but_teacher_body_command():
    from flybrain.training.banc_coverage import CoverageRecorder
    from flybrain.neural.banc_memory import MemoryBancReadout
    from flybrain.world.sensing import SensoryObservation
    from flybrain.training.readout_experiment import teacher_command
    m=MemoryBancReadout(torch.zeros(16),torch.ones(16));r=CoverageRecorder(m,'test-case','teacher')
    o=SensoryObservation(.03,.02);d=r.decide(o);target=teacher_command(o)
    assert [d.command.forward,d.command.turn]==target.tolist()
    assert len(r.rows[0]['rates'])==16
    r.reset();assert r.tick==0 and torch.count_nonzero(r.decoder.feature_rates)==0


def test_aggregate_baseline_is_train_only_mean_and_keeps_negative_labels():
    from flybrain.training.banc_coverage import target_baseline
    y=torch.tensor([[.7,-1.],[.7,-.5],[.7,.5],[.7,1.]],dtype=torch.float64)
    b=target_baseline(y)
    assert b['mean_command']==[.7,0.] and b['constant_mse']==[0.,.625]


def test_original_absolute_fit_gate_cannot_be_relaxed_by_aggregate_variance():
    from flybrain.training.banc_coverage import coverage_gates
    summaries={'trained':{'successes':8,'mean_progress_fraction':1.},'untrained':{'successes':0,'mean_progress_fraction':-.1},'fixed':{'successes':0,'mean_progress_fraction':-1.}}
    assert coverage_gates(.02,.01,.2,summaries,{'checks':True})['passed']
    assert not coverage_gates(.03,.01,.2,summaries,{'checks':True})['passed']
    assert not coverage_gates(.02,.03,.2,summaries,{'checks':True})['passed']


def test_collection_protocol_is_single_use_and_destination_bound(tmp_path):
    from flybrain.training.banc_coverage import prepare_collection,load_collection,collect
    path=prepare_collection(tmp_path/'collection');p,_=load_collection(path)
    assert p['max_episodes']==64 and p['optimizer_steps']==0
    (path.parent/'execution.json').write_text('{}')
    with pytest.raises(FileExistsError):collect(path)
    copied=tmp_path/'copy';copied.mkdir()
    for name in ('protocol.json','protocol.sha256'):(copied/name).write_bytes((path.parent/name).read_bytes())
    with pytest.raises(ValueError):load_collection(copied/'protocol.json')


def test_fit_refuses_partial_receipt_before_reading_or_updating(tmp_path):
    from flybrain.training.banc_coverage import run_training
    path=tmp_path/'protocol.json';path.write_text('{}')
    (tmp_path/'execution.json').write_text('{}')
    with pytest.raises(FileExistsError):run_training(path)
