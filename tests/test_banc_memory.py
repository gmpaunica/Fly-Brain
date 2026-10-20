"""Numerical and episode-boundary checks for the measured temporal repair."""

import math
import pytest
import torch


def model():
    from flybrain.neural.banc_memory import MemoryBancReadout
    return MemoryBancReadout(torch.zeros(16), torch.ones(16))


def test_dual_timescale_conditioning_retains_each_history_and_matrix_orientation():
    from flybrain.neural.banc_memory import memory_features
    rates = torch.tensor([10., 14., 20., 30., 40., 24., 36., 48.] * 2)
    assert memory_features(rates).tolist() == [12.,22.,33.,44.,4.,4.,6.,8.] * 2
    m=model()
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
        m.hidden.weight[0,12]=.25; m.output.weight[1,0]=.5
    assert m(rates).tolist() == pytest.approx([.5,math.tanh(.5*math.tanh(1.))])
    assert sum(p.numel() for p in m.parameters()) == 306


def test_reconstruction_of_saved_ema_counts_refilters_causally_and_resets_cases():
    from flybrain.neural.banc_memory import refilter_rates
    alpha=-math.expm1(-.02/.1)
    rates=torch.tensor([[alpha*50]*8,[(1-alpha)*alpha*50+alpha*100]*8,[alpha*50]*8],dtype=torch.float64)
    original=rates.clone()
    slow=refilter_rates(rates,['a','a','b'])
    assert torch.equal(rates,original)
    sa=-math.expm1(-.02/.5)
    assert slow[:,0].tolist()==pytest.approx([sa*50,(1-sa)*sa*50+sa*100,sa*50])
    bad=rates.clone();bad[0,0]+=.01
    with pytest.raises(ValueError):refilter_rates(bad,['a','a','b'])


def test_two_filters_observe_identical_actual_spikes_and_reset_replays():
    from flybrain.neural.banc_memory import make_memory_controller
    from flybrain.neural.banc_motor import make_banc_controller
    from flybrain.neural.banc_network import load_verified_banc_config
    from flybrain.world.sensing import SensoryObservation
    learned=make_memory_controller(model());fixed=make_banc_controller(load_verified_banc_config())
    outputs=[]
    for _ in range(3):
        a=learned.decide(SensoryObservation(.01,.02));b=fixed.decide(SensoryObservation(.01,.02))
        assert torch.equal(a.activity.spike_counts,b.activity.spike_counts)
        assert torch.equal(learned.decoder.feature_rates[:8],fixed.decoder.feature_rates)
        outputs.append(a)
    learned.reset()
    assert torch.count_nonzero(learned.decoder.feature_rates)==0
    first=learned.decide(SensoryObservation(.01,.02))
    assert first.command==outputs[0].command
    assert torch.equal(first.activity.spike_counts,outputs[0].activity.spike_counts)


@pytest.mark.parametrize('rates',[torch.zeros(8),torch.zeros((0,16)),torch.full((16,),-1.),torch.full((16,),1001.),torch.full((16,),float('nan'))])
def test_invalid_memory_features_fail_closed(rates):
    with pytest.raises(ValueError):model()(rates)


def test_artificial_readout_fit_changes_only_parameters_and_keeps_normalization():
    from flybrain.training.banc_training import fit_readout
    m=model();a=m.mean.clone();b=m.scale.clone()
    x=torch.arange(64,dtype=torch.float32).reshape(4,16)
    result=fit_readout(m,x,torch.tensor([[.7,-.8],[.7,-.2],[.7,.2],[.7,.8]]),updates=3)
    assert result['failure'] is None and result['parameter_changed_scalars']>0
    assert result['gradient_nonzero_scalars']>0
    assert torch.equal(a,m.mean) and torch.equal(b,m.scale)


def test_memory_checkpoint_exact_output_and_refuses_wrong_filter_or_overwrite(tmp_path):
    from flybrain.neural.banc_memory import save_checkpoint,load_checkpoint
    m=model();path=tmp_path/'model.pt'
    save_checkpoint(path,m,protocol_sha256='a'*64)
    loaded=load_checkpoint(path,protocol_sha256='a'*64,mean=m.mean,scale=m.scale)
    assert torch.equal(m(torch.arange(16).float()),loaded(torch.arange(16).float()))
    with pytest.raises(FileExistsError):save_checkpoint(path,m,protocol_sha256='a'*64)
    payload=torch.load(path,weights_only=True);payload['contract']['filter_tau_seconds']=[.1,.6]
    torch.save(payload,path)
    with pytest.raises(ValueError):load_checkpoint(path,protocol_sha256='a'*64,mean=m.mean,scale=m.scale)


def test_memory_dataset_is_exact_original_training_labels_and_refiltered_counts():
    from flybrain.training.banc_memory import load_training_data
    from flybrain.training.banc_protocol import load_readiness,DEFAULT_READINESS
    original=load_readiness(DEFAULT_READINESS);data=load_training_data()
    assert data.features.shape==(1008,16)
    assert torch.equal(data.features[:,:8],original.features)
    assert torch.equal(data.targets,original.targets)
    assert data.case_counts==original.case_counts


def test_memory_protocol_bound_single_use_and_rejects_resealed_changes(tmp_path):
    import json,hashlib
    from flybrain.training.banc_memory import prepare_protocol,load_protocol,run_training
    path=prepare_protocol(tmp_path/'run');p,_,_=load_protocol(path)
    assert all(c['split']=='train' for c in p['cases'])
    original=path.read_bytes();p['seed']=1011;path.write_text(json.dumps(p))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):load_protocol(path)
    path.write_bytes(original);path.with_suffix('.sha256').write_text(hashlib.sha256(original).hexdigest())
    copied=tmp_path/'copy';copied.mkdir()
    for name in ('protocol.json','protocol.sha256'):(copied/name).write_bytes((path.parent/name).read_bytes())
    with pytest.raises(ValueError):load_protocol(copied/'protocol.json')
    (path.parent/'execution.json').write_text('{}')
    with pytest.raises(FileExistsError):run_training(path)


def test_two_tick_memory_behavior_accounts_three_controls_and_replay(tmp_path):
    from flybrain.training.banc_memory import evaluate_training_models
    from flybrain.training.evaluation import EvaluationConfig,locked_cases
    from flybrain.neural.banc_memory import save_checkpoint,load_checkpoint
    m=model();path=tmp_path/'m.pt';save_checkpoint(path,m,protocol_sha256='a'*64)
    loaded=load_checkpoint(path,protocol_sha256='a'*64,mean=m.mean,scale=m.scale)
    r=evaluate_training_models(m,loaded,m,locked_cases()['train'][:1],EvaluationConfig(horizon_ticks=2))
    assert r['actual_episodes']==4 and r['actual_world_ticks']==8 and r['actual_neural_steps']==160
    assert r['checkpoint_replay_exact'] and r['effective_weights_unchanged']
