"""PN collection preserves training trajectories; fitting remains bounded."""

import pytest
import torch


def test_projection_recorder_preserves_frozen_student_commands_and_old_features():
    from flybrain.training.banc_projection import ProjectionRecorder
    from flybrain.training.banc_coverage import CoverageRecorder
    from flybrain.neural.banc_memory import MemoryBancReadout
    from flybrain.world.sensing import SensoryObservation
    m=MemoryBancReadout(torch.zeros(16),torch.ones(16))
    a=ProjectionRecorder(m,'case','student');b=CoverageRecorder(m,'case','student')
    for _ in range(4):
        x=a.decide(SensoryObservation(.023,.017));y=b.decide(SensoryObservation(.023,.017))
        assert x.command==y.command
        assert a.rows[-1]['rates'][:16]==b.rows[-1]['rates']
        assert a.rows[-1]['target_turn']==b.rows[-1]['target_turn']
    assert len(a.rows[-1]['rates'])==20
    a.reset();assert not a.rows and a.tick==0


def test_projection_collection_is_single_use_destination_bound_and_exact_replay(tmp_path):
    from flybrain.training.banc_projection import prepare_collection,load_collection,collect
    path=prepare_collection(tmp_path/'run');p,_=load_collection(path)
    assert p['max_neural_steps']==317280 and p['optimizer_steps']==0
    assert p['expected_sample_count']==15864
    (path.parent/'execution.json').write_text('{}')
    with pytest.raises(FileExistsError):collect(path)
    copied=tmp_path/'copy';copied.mkdir()
    for n in ('protocol.json','protocol.sha256'):(copied/n).write_bytes((path.parent/n).read_bytes())
    with pytest.raises(ValueError):load_collection(copied/'protocol.json')


def test_projection_fit_refuses_consumed_attempt_before_loading(tmp_path):
    from flybrain.training.banc_projection import run_training
    path=tmp_path/'protocol.json';path.write_text('{}');(path.parent/'execution.json').write_text('{}')
    with pytest.raises(FileExistsError):run_training(path)


def test_two_tick_projection_matched_assay_accounts_steps_and_checkpoint_replay(tmp_path):
    from flybrain.training.banc_projection import evaluate_models
    from flybrain.neural.banc_projection import ProjectionReadout,save_checkpoint,load_checkpoint
    from flybrain.training.evaluation import locked_cases,EvaluationConfig
    m=ProjectionReadout(torch.zeros(20),torch.ones(20));path=tmp_path/'m.pt'
    save_checkpoint(path,m,protocol_sha256='a'*64)
    loaded=load_checkpoint(path,protocol_sha256='a'*64,mean=m.mean,scale=m.scale)
    r=evaluate_models(m,loaded,m,locked_cases()['train'][:1],EvaluationConfig(horizon_ticks=2))
    assert r['actual_episodes']==4 and r['actual_world_ticks']==8 and r['actual_neural_steps']==160
    assert r['checkpoint_replay_exact'] and r['effective_weights_unchanged']
