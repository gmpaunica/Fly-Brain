"""Measured upstream spike precision, without a changed spiking circuit."""

import math
import pytest
import torch


def model():
    from flybrain.neural.banc_projection import ProjectionReadout
    return ProjectionReadout(torch.zeros(20),torch.ones(20))


def test_projection_pair_conditioning_and_small_head_orientation():
    from flybrain.neural.banc_projection import projection_features
    x=torch.tensor([10.,14.,20.,30.,40.,24.,36.,48.]*2+[30.,34.,20.,22.])
    assert projection_features(x).tolist()==[12.,22.,33.,44.,4.,4.,6.,8.]*2+[32.,4.,21.,2.]
    m=model()
    with torch.no_grad():
        for p in m.parameters():p.zero_()
        m.hidden.weight[0,17]=.25;m.output.weight[1,0]=.5
    assert m(x).tolist()==pytest.approx([.5,math.tanh(.5*math.tanh(1))])
    assert sum(p.numel() for p in m.parameters())==370


def test_weighted_previous_spikes_have_post_pre_orientation_and_seconds_units():
    from flybrain.neural.banc_projection import incoming_event_rate
    w=torch.tensor([[1.,2.,0.],[0.,1.,2.]])
    spikes=torch.tensor([1.,0.,1.])
    # sums [1,2], budget3 and 20ms -> [16.6667,33.3333] equivalent Hz
    assert incoming_event_rate(w@spikes,.02).tolist()==pytest.approx([1/.06,2/.06])
    with pytest.raises(ValueError):incoming_event_rate(torch.ones(2),0)


def test_projection_observation_preserves_every_spike_voltage_and_endpoint_history():
    from flybrain.neural.banc_projection import make_projection_controller
    from flybrain.neural.banc_memory import make_memory_controller,MemoryBancReadout
    from flybrain.world.sensing import SensoryObservation
    p=make_projection_controller(model());old=make_memory_controller(MemoryBancReadout(torch.zeros(16),torch.ones(16)))
    for i in range(6):
        o=SensoryObservation(.01+i*.002,.023-i*.001)
        a=p.decide(o);b=old.decide(o)
        assert torch.equal(a.activity.spike_counts,b.activity.spike_counts)
        assert torch.equal(a.activity.snapshot.voltage,b.activity.snapshot.voltage)
        assert torch.equal(a.activity.snapshot.spikes,b.activity.snapshot.spikes)
        assert torch.equal(p.decoder.feature_rates[:16],old.decoder.feature_rates)
    assert p.network.population.step_count==120
    assert p.decoder.feature_rates[16:].sum()>0
    copy=p.decoder.feature_rates;p.reset()
    assert not torch.count_nonzero(p.decoder.feature_rates)
    assert copy.sum()>0
    a=p.decide(SensoryObservation(.01,.023));p.reset()
    assert p.decide(SensoryObservation(.01,.023)).command==a.command


def test_projection_signal_reads_actual_cut_weights_and_never_encoder_current():
    from flybrain.neural.banc_projection import make_projection_controller
    from flybrain.world.sensing import SensoryObservation
    c=make_projection_controller(model())
    c.network.population=c.network.population_for_weights(torch.zeros_like(c.network.population.weights))
    for _ in range(4):c.decide(SensoryObservation(.01,.02))
    assert torch.count_nonzero(c.decoder.feature_rates[16:])==0
    assert c.network.population.spikes.sum()>0 or c.network._phases.sum()>0


def test_projection_checkpoint_binds_source_ids_and_signal_semantics(tmp_path):
    from flybrain.neural.banc_projection import save_checkpoint,load_checkpoint
    m=model();path=tmp_path/'p.pt';save_checkpoint(path,m,protocol_sha256='a'*64)
    loaded=load_checkpoint(path,protocol_sha256='a'*64,mean=m.mean,scale=m.scale)
    assert torch.equal(m(torch.arange(20).float()),loaded(torch.arange(20).float()))
    with pytest.raises(FileExistsError):save_checkpoint(path,m,protocol_sha256='a'*64)
    payload=torch.load(path,weights_only=True);payload['contract']['projection_signal']='external odor'
    torch.save(payload,path)
    with pytest.raises(ValueError):load_checkpoint(path,protocol_sha256='a'*64,mean=m.mean,scale=m.scale)


@pytest.mark.parametrize('x',[torch.zeros(16),torch.full((20,),float('nan')),torch.full((20,),-1.),torch.zeros(0,20)])
def test_invalid_projection_feature_arrays_rejected(x):
    with pytest.raises(ValueError):model()(x)
