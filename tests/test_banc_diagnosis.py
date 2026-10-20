"""The diagnostic must observe actual transitions without changing them."""

import importlib.util
import math

import pytest
import torch

from flybrain.neural.lif import LIFPopulation


def test_trace_observes_delayed_input_and_reset_overshoot_without_extra_steps():
    assert importlib.util.find_spec('flybrain.training.banc_diagnosis'), 'diagnostic missing'
    from flybrain.training.banc_diagnosis import trace_step
    population = LIFPopulation(torch.tensor([[0., 0.], [1.5, 0.]]), record_charge=True)
    current = torch.tensor([30., 0.])
    first = trace_step(population, current, (1,))[0]
    second = trace_step(population, torch.zeros(2), (1,))[0]
    assert first['synaptic_jump'] == 0
    assert first['spike'] == 0
    assert second['synaptic_jump'] == 1.5
    assert second['pre_reset_voltage'] == 1.5
    assert second['reset_loss'] == 1.5
    assert second['overshoot'] == .5
    assert second['voltage_after'] == 0
    assert second['spike'] == 1
    assert population.step_count == 2


def test_trace_reports_real_leak_and_does_not_change_unobserved_neurons():
    assert importlib.util.find_spec('flybrain.training.banc_diagnosis'), 'diagnostic missing'
    from flybrain.training.banc_diagnosis import trace_step
    weights = torch.zeros((2, 2))
    traced = LIFPopulation(weights, record_charge=True)
    original = LIFPopulation(weights)
    for current in (torch.tensor([1., 2.]), torch.zeros(2)):
        rows = trace_step(traced, current, (0, 1))
        original.step(current)
        assert torch.equal(traced.voltage, original.voltage)
        assert torch.equal(traced.spikes, original.spikes)
    assert rows[0]['leak_loss'] == pytest.approx((1-math.exp(-.05))**2, abs=1e-7)
    assert rows[0]['reset_loss'] == 0
