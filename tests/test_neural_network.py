import math
import random

import pytest
import torch

from flybrain.neural.network import SyntheticNetwork
from flybrain.neural.topology import build_synthetic_topology
from flybrain.world.sensing import SensoryObservation


def indices(brain, group):
    return list(brain.topology.indices(group))


def test_topology_labels_cover_every_neuron_and_edges_stay_directed():
    graph = build_synthetic_topology(seed=7)
    assert len(graph.labels) == len(set(graph.labels)) == 100
    assert len(graph.groups) == 100
    allowed = {
        ('interneuron_left', 'sensory_left'),
        ('interneuron_right', 'sensory_right'),
        ('motor_left', 'interneuron_left'),
        ('motor_right', 'interneuron_right'),
    }
    seen = set()
    for post, pre in graph.weights.nonzero().tolist():
        edge = (graph.groups[post], graph.groups[pre])
        assert edge in allowed
        seen.add(edge)
        assert graph.weights[post, pre] > 0
        assert graph.weights[pre, post] == 0
    assert seen == allowed
    assert not graph.weights.diagonal().any()
    assert len(graph.indices('motor_forward')) == 10


def test_local_topology_seed_repeats_without_global_rng_use():
    torch_before = torch.random.get_rng_state().clone()
    python_before = random.getstate()
    a, b, c = (build_synthetic_topology(seed=s) for s in (7, 7, 8))
    assert torch.equal(a.weights, b.weights)
    assert not torch.equal(a.weights, c.weights)
    assert torch.equal(a.weights != 0, c.weights != 0)
    assert torch.equal(torch_before, torch.random.get_rng_state())
    assert python_before == random.getstate()


def test_encoder_independently_scales_saturates_and_zeroes_other_groups():
    brain = SyntheticNetwork()
    current = brain.encode(SensoryObservation(0.0125, 0.05))
    assert current[indices(brain, 'sensory_left')].tolist() == [1.] * 10
    assert current[indices(brain, 'sensory_right')].tolist() == [2.] * 10
    assert not current[20:].any()
    assert not brain.encode(SensoryObservation(0, 0)).any()
    huge = brain.encode(SensoryObservation(1e308, 1e308))
    assert torch.isfinite(huge).all()
    assert huge[:20].tolist() == [2.] * 20


@pytest.mark.parametrize('left,right,active,silent', [
    (0.025, 0, 'left', 'right'), (0, 0.025, 'right', 'left'),
])
def test_sensory_spikes_propagate_to_same_side_only(left, right, active, silent):
    brain = SyntheticNetwork()
    result = brain.advance(SensoryObservation(left, right), substeps=20)
    for prefix in ('sensory', 'interneuron', 'motor'):
        assert (result.spike_counts[indices(brain, f'{prefix}_{active}')] > 0).all()
        assert not result.spike_counts[indices(brain, f'{prefix}_{silent}')].any()
    assert result.start_step == 0 and result.end_step == 20
    assert result.elapsed_seconds == pytest.approx(0.020)
    # Last-step spike vector alone misses the earlier sensory spike.
    assert result.spike_counts.sum() > result.snapshot.spikes.sum()


def test_zero_sensory_input_only_drives_documented_tonic_forward_group():
    brain = SyntheticNetwork()
    result = brain.advance(SensoryObservation(0, 0), substeps=100)
    forward = indices(brain, 'motor_forward')
    assert (result.spike_counts[forward] == 3).all()
    other = [i for i in range(100) if i not in forward]
    assert not result.spike_counts[other].any()


def test_counts_partition_and_reset_replay_match_every_substep():
    obs = SensoryObservation(0.025, 0.01)
    whole, split = SyntheticNetwork(seed=9), SyntheticNetwork(seed=9)
    weights_before = whole.population.weights
    full = whole.advance(obs, substeps=100)
    parts = [split.advance(obs, substeps=20) for _ in range(5)]
    assert torch.equal(full.spike_counts, sum(p.spike_counts for p in parts))
    assert torch.equal(full.snapshot.voltage, split.snapshot().voltage)
    assert torch.equal(full.snapshot.spikes, split.snapshot().spikes)
    whole.reset()
    assert whole.snapshot().step_count == 0
    assert not whole.snapshot().spikes.any()
    assert not whole.snapshot().voltage.any()
    replay = whole.advance(obs, substeps=100)
    assert torch.equal(full.spike_counts, replay.spike_counts)
    assert torch.equal(full.snapshot.voltage, replay.snapshot.voltage)
    assert torch.equal(weights_before, whole.population.weights)


def test_snapshots_counts_and_topology_weights_cannot_mutate_dynamics():
    a, b = SyntheticNetwork(), SyntheticNetwork()
    obs = SensoryObservation(0.025, 0)
    result = a.advance(obs)
    b.advance(obs)
    result.snapshot.voltage.fill_(99)
    result.snapshot.spikes.fill_(True)
    result.spike_counts.zero_()
    a.topology.weights.fill_(99)
    for _ in range(3):
        a.snapshot()
    assert torch.equal(a.advance(obs).spike_counts, b.advance(obs).spike_counts)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_network_dtype_shape_device_and_no_autograd(dtype):
    brain = SyntheticNetwork(dtype=dtype)
    result = brain.advance(SensoryObservation(0.025, 0.025))
    assert result.spike_counts.shape == (100,)
    assert result.spike_counts.dtype == torch.int64
    assert result.snapshot.spikes.dtype == torch.bool
    assert result.snapshot.voltage.dtype == dtype
    assert result.snapshot.voltage.device.type == 'cpu'
    assert brain.encode(SensoryObservation(0, 0)).dtype == dtype
    assert not result.snapshot.voltage.requires_grad


@pytest.mark.parametrize('substeps', [0, -1, 1.5, True])
def test_invalid_substeps_leave_network_unchanged(substeps):
    brain = SyntheticNetwork()
    with pytest.raises(ValueError):
        brain.advance(SensoryObservation(1, 1), substeps=substeps)
    assert brain.snapshot().step_count == 0


@pytest.mark.parametrize('left,right', [(-1, 0), (0, math.inf), (math.nan, 0)])
def test_encoder_validates_structural_sensory_input(left, right):
    class RawObservation:
        odor_left = left
        odor_right = right
    brain = SyntheticNetwork()
    with pytest.raises(ValueError):
        brain.advance(RawObservation())
    assert brain.snapshot().step_count == 0


def test_unsupported_dtype_rejected():
    with pytest.raises(ValueError):
        SyntheticNetwork(dtype=torch.float16)


def test_cpu_contract_ignores_ambient_default_device():
    # A caller's global allocation context must not silently move this CPU model.
    with torch.device('meta'):
        brain = SyntheticNetwork()
        result = brain.advance(SensoryObservation(0.025, 0))
    assert result.snapshot.voltage.device.type == 'cpu'
    assert result.spike_counts.device.type == 'cpu'
