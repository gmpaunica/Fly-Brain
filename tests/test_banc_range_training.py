"""Same historical body policy and labels, separate fixed encoder circuit."""

import pytest
import torch


def test_range_shadow_records_new_features_without_changing_body_policy():
    from flybrain.training.banc_range import RangeRecorder
    from flybrain.training.banc_coverage import CoverageRecorder
    from flybrain.neural.banc_memory import MemoryBancReadout
    from flybrain.world.sensing import SensoryObservation
    m=MemoryBancReadout(torch.zeros(16),torch.ones(16))
    a=RangeRecorder(m,'case','student');b=CoverageRecorder(m,'case','student')
    for _ in range(3):
        x=a.decide(SensoryObservation(.8,1.6));y=b.decide(SensoryObservation(.8,1.6))
        assert x.command==y.command and torch.equal(x.activity.spike_counts,y.activity.spike_counts)
        assert a.rows[-1]['target_turn']==b.rows[-1]['target_turn']
    assert len(a.rows[-1]['rates'])==20
    assert a.network.population.step_count==a.shadow.network.population.step_count==60
    a.reset();assert a.network.population.step_count==a.shadow.network.population.step_count==0 and not a.rows


def test_range_assay_two_ticks_accounts_matched_revised_encoder_and_replay(tmp_path):
    from flybrain.training.banc_range import evaluate_models
    from flybrain.neural.banc_range import save_checkpoint,load_checkpoint
    from flybrain.neural.banc_projection import ProjectionReadout
    from flybrain.training.evaluation import locked_cases,EvaluationConfig
    m=ProjectionReadout(torch.zeros(20),torch.ones(20));path=tmp_path/'r.pt';save_checkpoint(path,m,protocol_sha256='a'*64)
    loaded=load_checkpoint(path,protocol_sha256='a'*64,mean=m.mean,scale=m.scale)
    r=evaluate_models(m,loaded,m,locked_cases()['train'][:1],EvaluationConfig(horizon_ticks=2))
    assert r['actual_episodes']==4 and r['actual_world_ticks']==8 and r['actual_neural_steps']==160
    assert r['checkpoint_replay_exact'] and r['effective_weights_unchanged']


def test_range_collection_and_fit_refuse_consumed_receipt_before_any_steps(tmp_path):
    from flybrain.training.banc_range import collect,run_training
    path=tmp_path/'protocol.json';path.write_text('{}');(path.parent/'execution.json').write_text('{}')
    for action in (collect,run_training):
        with pytest.raises(FileExistsError):action(path)
