"""A declared sensory rate code must not smuggle in steering or lose state."""

import pytest
import torch

from flybrain.data.banc import load_banc_subset
from flybrain.neural.banc_network import BancDynamicsConfig, BancNetwork
from flybrain.world.sensing import SensoryObservation


def network():
    return BancNetwork(load_banc_subset(connectome_gain=1), dynamics_config=BancDynamicsConfig(
        3., sensory_mode='phase_rate'))


def test_fractional_rate_schedule_has_less_than_one_spike_count_error_per_orn():
    brain = network()
    obs = SensoryObservation(.01, .04)  # 17.6 Hz / 35.2 Hz, independently.
    activity = brain.advance(obs, substeps=1000)
    for side, expected in (('left', 17.6), ('right', 35.2)):
        afferents = [i for i in brain.topology.indices(f'sensory_{side}') if not brain.topology.weights[i].any()]
        counts = activity.spike_counts[afferents]
        assert torch.all(torch.abs(counts - expected) < 1)
        assert len(torch.unique(counts)) == 2
    assert activity.end_step == 1000


def test_phase_encoder_peeking_is_pure_and_episode_reset_replays_spikes():
    brain = network()
    obs = SensoryObservation(.0625, 0)  # 44 Hz / zero.
    before = brain.encode(obs)
    assert torch.equal(before, brain.encode(obs))
    assert brain.population.step_count == 0
    first = brain.advance(obs, substeps=100)
    brain.reset()
    second = brain.advance(obs, substeps=100)
    assert torch.equal(first.spike_counts, second.spike_counts)
    assert torch.equal(first.snapshot.voltage, second.snapshot.voltage)
    assert not second.spike_counts[list(brain.topology.indices('sensory_right'))].any()
    brain.reset()
    assert not brain.advance(SensoryObservation(0, 0), substeps=100).spike_counts.any()


def test_rate_encoding_is_side_independent_bounded_and_keeps_existing_weights():
    brain = network()
    source = load_banc_subset(connectome_gain=1)
    legacy = BancNetwork(source, dynamics_config=BancDynamicsConfig(3))
    assert torch.equal(brain.topology.weights, legacy.topology.weights)
    with torch.device('meta'):
        extreme = network()
        activity = extreme.advance(SensoryObservation(1e308, 0), substeps=100)
    left = [i for i in extreme.topology.indices('sensory_left') if not extreme.topology.weights[i].any()]
    right = list(extreme.topology.indices('sensory_right'))
    assert set(activity.spike_counts[left].tolist()) <= {9, 10}  # 96 Hz cap.
    assert not activity.spike_counts[right].any()
    assert torch.isfinite(activity.snapshot.voltage).all()


def test_retained_orn_to_orn_connection_can_add_spikes_beyond_external_schedule():
    brain = network()
    activity = brain.advance(SensoryObservation(.0625, 0), substeps=100)
    receiver = brain.topology.index('720575941478318641')
    source = brain.topology.index('720575941577670489')
    assert brain.topology.weights[receiver, source].item() == 3
    assert activity.spike_counts[source].item() in (4, 5)
    assert activity.spike_counts[receiver].item() > 5
