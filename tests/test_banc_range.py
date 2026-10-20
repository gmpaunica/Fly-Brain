"""Arena-wide fixed sensory scaffolding; unchanged source circuit."""

import math
import pytest
import torch


def test_range_code_preserves_low_band_and_discriminates_old_clipped_odors():
    from flybrain.neural.banc_range import sensory_rate
    assert sensory_rate(0)==0
    assert sensory_rate(.025)==176*math.sqrt(.025)
    assert sensory_rate(.03)==176*math.sqrt(.03)
    assert 0<sensory_rate(.3)<sensory_rate(.8)<sensory_rate(1.6)<sensory_rate(2.5)==96
    assert sensory_rate(1e308)==96
    for value in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):sensory_rate(value)


def test_range_encoder_side_independence_exact_low_band_neural_replay_and_reset():
    from flybrain.neural.banc_range import RangeNetwork
    from flybrain.neural.banc_projection import ProjectionNetwork
    from flybrain.neural.banc_network import load_verified_banc_config
    from flybrain.data.banc import load_banc_subset
    from flybrain.world.sensing import SensoryObservation
    brains=[cls(load_banc_subset(connectome_gain=1),dynamics_config=load_verified_banc_config()) for cls in (RangeNetwork,ProjectionNetwork)]
    a,b=brains
    assert torch.equal(a.population.weights,b.population.weights)
    for _ in range(5):
        x=a.advance(SensoryObservation(.02,.03));y=b.advance(SensoryObservation(.02,.03))
        assert torch.equal(x.spike_counts,y.spike_counts) and torch.equal(x.snapshot.voltage,y.snapshot.voltage)
    rates=a._sensory_rates(SensoryObservation(.8,1.6))
    assert rates[list(a.topology.indices('sensory_left'))].unique().numel()==1
    assert rates[list(a.topology.indices('sensory_right'))].min()>rates.max()/1.2
    a.reset();x=a.advance(SensoryObservation(.8,1.6));a.reset();y=a.advance(SensoryObservation(.8,1.6))
    assert torch.equal(x.spike_counts,y.spike_counts)


def test_range_checkpoint_cannot_load_as_old_projection_candidate(tmp_path):
    from flybrain.neural.banc_range import save_checkpoint,load_checkpoint
    from flybrain.neural.banc_projection import ProjectionReadout,load_checkpoint as old_load
    m=ProjectionReadout(torch.zeros(20),torch.ones(20));path=tmp_path/'r.pt';save_checkpoint(path,m,protocol_sha256='a'*64)
    loaded=load_checkpoint(path,protocol_sha256='a'*64,mean=m.mean,scale=m.scale)
    assert torch.equal(m(torch.arange(20).float()),loaded(torch.arange(20).float()))
    with pytest.raises(ValueError):old_load(path,protocol_sha256='a'*64,mean=m.mean,scale=m.scale)
