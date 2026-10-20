"""Strict matrix, checkpoint and frozen-data checks for one capacity candidate."""

import math
import pytest
import torch


def model():
    from flybrain.neural.banc_capacity import WideMemoryReadout
    return WideMemoryReadout(torch.zeros(16),torch.ones(16))


def test_last_hidden_unit_uses_same_endpoint_only_calculation_and_local_rng():
    state=torch.random.get_rng_state().clone();m=model()
    assert torch.equal(state,torch.random.get_rng_state())
    assert sum(p.numel() for p in m.parameters())==1218
    with torch.no_grad():
        for p in m.parameters():p.zero_()
        m.hidden.weight[63,12]=.25;m.output.weight[1,63]=.5
    rates=torch.tensor([10.,14.,20.,30.,40.,24.,36.,48.]*2)
    assert m(rates).tolist()==pytest.approx([.5,math.tanh(.5*math.tanh(1.))])


def test_wide_checkpoint_exact_output_and_rejects_older_architecture(tmp_path):
    from flybrain.neural.banc_capacity import save_checkpoint,load_checkpoint
    m=model();path=tmp_path/'m.pt';save_checkpoint(path,m,protocol_sha256='a'*64)
    loaded=load_checkpoint(path,protocol_sha256='a'*64,mean=m.mean,scale=m.scale)
    assert torch.equal(m(torch.arange(16).float()),loaded(torch.arange(16).float()))
    payload=torch.load(path,weights_only=True);payload['contract']['architecture']['hidden']=16
    torch.save(payload,path)
    with pytest.raises(ValueError):load_checkpoint(path,protocol_sha256='a'*64,mean=m.mean,scale=m.scale)


def test_capacity_protocol_pins_same_covered_data_and_original_error_gates(tmp_path):
    from flybrain.training.banc_capacity import prepare_protocol,load_protocol,run_training
    path=prepare_protocol(tmp_path/'run');p,_,data=load_protocol(path)
    assert data.features.shape==(15864,16) and p['trainable_parameters']==1218
    assert p['gates']['original_turn_mse_max']==.02225273146561865
    assert p['gates']['aggregate_absolute_turn_mse_max']==.02225273146561865
    (path.parent/'execution.json').write_text('{}')
    with pytest.raises(FileExistsError):run_training(path)


def test_wide_artificial_gradients_do_not_change_normalization():
    from flybrain.training.banc_training import fit_readout
    m=model();a=m.mean.clone();b=m.scale.clone()
    x=torch.arange(64).float().reshape(4,16)
    r=fit_readout(m,x,torch.tensor([[.7,-.8],[.7,-.2],[.7,.2],[.7,.8]]),updates=3)
    assert r['parameter_changed_scalars']>0 and r['gradient_nonzero_scalars']>0
    assert torch.equal(a,m.mean) and torch.equal(b,m.scale)
