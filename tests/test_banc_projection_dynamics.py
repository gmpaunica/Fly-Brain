"""PN-only reset/parameter changes preserve the rest of the neural contract."""

import math

import pytest
import torch

from flybrain.neural import banc_synapses
from flybrain.data.banc import load_banc_subset
from flybrain.neural.banc_network import BancDynamicsConfig, BancNetwork
from flybrain.training import banc_recovery
from flybrain.world.sensing import SensoryObservation
from flybrain.neural.lif import LIFPopulation


def population():
    assert hasattr(banc_synapses, 'ProjectionChargePopulation'), 'charge-preserving PN model missing'
    weights = torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [1.5, 0., 0.]])
    return banc_synapses.ProjectionChargePopulation(weights, (1,), record_charge=True)


def test_pn_retains_overshoot_while_other_cells_keep_hard_reset():
    brain = population()
    brain.step(torch.tensor([30., 0., 0.]))
    assert brain.step(torch.zeros(3)).tolist() == [False, True, True]
    assert brain.voltage[1].item() == pytest.approx(.4)
    assert brain.voltage[2].item() == 0
    assert brain.charge_losses.reset[1].item() == pytest.approx(1.1)
    assert brain.charge_losses.reset[2].item() == 1.5
    brain.step(torch.zeros(3))
    assert brain.voltage[1].item() == pytest.approx(.4 * math.exp(-.001/.025))


def test_zero_input_pn_tail_decays_below_gate_and_reset_clears_it():
    brain = population()
    brain.step(torch.tensor([30., 0., 0.]))
    brain.step(torch.zeros(3))
    for _ in range(250):
        assert not brain.step(torch.zeros(3)).any()
    assert 0 < brain.voltage[1].item() < 1e-4
    brain.reset()
    assert not brain.voltage.any()
    assert not brain.spikes.any()
    assert brain.step_count == 0


def test_large_pn_jump_uses_one_spike_per_step_and_preserves_residual_charge():
    assert hasattr(banc_synapses, 'ProjectionChargePopulation'), 'charge-preserving PN model missing'
    brain = banc_synapses.ProjectionChargePopulation(torch.tensor([[0., 0.], [3., 0.]]), (1,))
    brain.step(torch.tensor([30., 0.]))
    assert brain.step(torch.zeros(2)).tolist() == [False, True]
    assert brain.voltage[1].item() == pytest.approx(1.9)
    assert brain.step(torch.zeros(2)).tolist() == [False, True]
    assert brain.voltage[1].item() == pytest.approx(1.9*math.exp(-.04)-1.1)


def test_charge_mode_replays_and_preserves_model_through_causal_rebuild():
    config = BancDynamicsConfig(3., sensory_mode='phase_rate', projection_dynamics='charge_preserving')
    brain = BancNetwork(load_banc_subset(connectome_gain=1.), dynamics_config=config)
    first = brain.advance(SensoryObservation(.1, .1), 100)
    brain.reset()
    replay = brain.advance(SensoryObservation(.1, .1), 100)
    assert torch.equal(first.spike_counts, replay.spike_counts)
    assert torch.equal(first.snapshot.voltage, replay.snapshot.voltage)
    assert isinstance(brain.population_for_weights(torch.zeros_like(brain.topology.weights)),
                      banc_synapses.ProjectionChargePopulation)
    protocol = banc_recovery.make_protocol(pn_charge=True)
    assert protocol['candidate_configs'][0]['projection_dynamics'] == 'charge_preserving'
    assert protocol['budget']['maximum_total_neural_substeps'] == 30000


def test_per_neuron_causal_comparison_rejects_compensating_sensory_changes():
    assert hasattr(banc_recovery, 'sensory_counts_match')
    graph = load_banc_subset(connectome_gain=1.)
    intact = {'pulse_per_neuron_counts': [10] * 83}
    changed = {'pulse_per_neuron_counts': [10] * 83}
    left = graph.indices('sensory_left')
    changed['pulse_per_neuron_counts'][left[0]] += 1
    changed['pulse_per_neuron_counts'][left[1]] -= 1
    assert not banc_recovery.sensory_counts_match(graph, intact, changed)
    assert not banc_recovery.sensory_counts_match(graph, intact, changed, allow_sensory_edge_receivers=True)
    changed = {'pulse_per_neuron_counts': [10] * 83}
    changed['pulse_per_neuron_counts'][graph.index('720575941478318641')] += 1
    assert not banc_recovery.sensory_counts_match(graph, intact, changed)
    assert banc_recovery.sensory_counts_match(graph, intact, changed, allow_sensory_edge_receivers=True)


def test_pn_current_fraction_cpu_ownership_and_nonpn_equivalence():
    weights = torch.zeros((3, 3), device='cpu')
    current = torch.tensor([20., 20., 7.], device='cpu')
    torch.set_default_device('meta')
    try:
        brain = banc_synapses.ProjectionChargePopulation(weights, (1,), record_charge=True)
        ordinary = LIFPopulation(weights)
        brain.step(current)
        ordinary.step(current)
        assert brain.voltage.device.type == 'cpu'
        assert brain.voltage[1].item() == pytest.approx(20 * -math.expm1(-.001/.025))
        for i in range(50):
            drive = current if i % 3 else current * 0
            brain.step(drive)
            ordinary.step(drive)
            assert torch.equal(brain.voltage[[0, 2]], ordinary.voltage[[0, 2]])
            assert torch.equal(brain.spikes[[0, 2]], ordinary.spikes[[0, 2]])
        assert brain.charge_losses.leak.device.type == 'cpu'
    finally:
        torch.set_default_device('cpu')
