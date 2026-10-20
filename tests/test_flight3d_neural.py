"""Causal directed spikes, group-specific motor reads and frozen checkpoints."""

import numpy as np
import torch

from flybrain.flight3d.neural import FlightBrain, FlightReadout, MotorFeatures, save_readout, load_readout


def fixture():
    return {'schema':'flight3d-circuit-v1','nodes':[
        {'id':'1','roles':['vision'],'metadata':{'side':'left','cell_type':'LPLC4'}},
        {'id':'2','roles':['wing_power'],'metadata':{'side':'left'}},
        {'id':'3','roles':['wing_steering'],'metadata':{'side':'right'}},
        {'id':'4','roles':['landing_motor'],'metadata':{'side':'left'}}],
        'edges':[{'pre':'1','post':'2','count':10},{'pre':'2','post':'3','count':10},{'pre':'3','post':'4','count':10}]}


def test_directed_propagation_cuts_and_reset():
    b=FlightBrain(fixture())
    counts=b.advance_rates(np.array([240.,0,0,0]),steps=400)
    assert all(counts>0) and counts[0]>counts[1]>counts[2]>counts[3]
    b.reset(); b.silenced={'2'}
    cut=b.advance_rates(np.array([240.,0,0,0]),steps=400)
    assert cut[0]>0 and not cut[1:].any()
    b.reset()
    assert b.steps==0 and not b.voltage.any() and not b.history.any()


def test_readout_group_separation_zero_motor_drive_and_optimizer(tmp_path):
    f=MotorFeatures(fixture()); sizes=f.sizes
    torch.manual_seed(7)
    model=FlightReadout(sizes)
    rates=torch.ones(8,sum(sizes))*50
    before=model(rates).detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    loss=(model(rates)-.1).square().mean(); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before,model(rates))
    assert not model(torch.zeros(sum(sizes))).any()
    cut=rates.clone(); cut[:,:sizes[0]]=0
    assert not model(cut)[:,0].any()
    assert torch.equal(model(cut)[:,1:],model(rates)[:,1:])
    path=tmp_path/'model.pt'
    save_readout(path,model,'fixture')
    loaded=load_readout(path,'fixture')
    assert torch.equal(model(rates),loaded(rates))
    assert all(not p.requires_grad for p in loaded.parameters())
