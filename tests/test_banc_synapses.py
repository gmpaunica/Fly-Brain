"""Numerical tests for area-preserving, PN-only temporal input filtering."""

import importlib.util
import math

import pytest
import torch

from flybrain.neural.lif import LIFPopulation


def population():
    assert importlib.util.find_spec('flybrain.neural.banc_synapses'), 'PN filter missing'
    from flybrain.neural.banc_synapses import ProjectionFilteredPopulation
    return ProjectionFilteredPopulation(torch.tensor([[0., 0.], [1.5, 0.]]), (1,))


def test_one_presynaptic_impulse_is_delayed_spread_and_area_preserved():
    brain = population()
    brain.step(torch.tensor([30., 0.]))
    assert brain.synaptic_trace.tolist() == [0.]
    total = 0.
    for _ in range(100):
        brain.step(torch.zeros(2))
        total += brain.synaptic_trace.item()
    assert total == pytest.approx(1.5, abs=2e-6)
    assert brain.synaptic_trace.item() < 1e-8


def test_filter_starts_with_exact_exponential_fraction_and_reset_clears_tail():
    brain = population()
    brain.step(torch.tensor([30., 0.]))
    brain.step(torch.zeros(2))
    assert brain.synaptic_trace.item() == pytest.approx(1.5*(1-math.exp(-.2)))
    brain.synaptic_trace.fill_(999)
    assert brain.synaptic_trace.item() < 1
    brain.reset()
    assert not brain.synaptic_trace.any()
    assert not brain.voltage.any()
    assert brain.step_count == 0


def test_unfiltered_rows_keep_instantaneous_jumps_and_invalid_input_is_atomic():
    assert importlib.util.find_spec('flybrain.neural.banc_synapses'), 'PN filter missing'
    from flybrain.neural.banc_synapses import ProjectionFilteredPopulation
    weights = torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [1.5, 0., 0.]])
    brain = ProjectionFilteredPopulation(weights, (1,))
    brain.step(torch.tensor([30., 0., 0.]))
    spikes = brain.step(torch.zeros(3))
    assert spikes.tolist() == [False, False, True]
    before = brain.synaptic_trace
    with pytest.raises(ValueError):
        brain.step(torch.full((3,), float('nan')))
    assert torch.equal(before, brain.synaptic_trace)
    assert torch.equal(brain.weights, weights)
