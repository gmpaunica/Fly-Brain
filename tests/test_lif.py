import math

import pytest
import torch

from flybrain.neural.lif import LIFConfig, LIFPopulation


def half_life_config(**kwargs):
    return LIFConfig(dt=0.001, tau=0.001 / math.log(2), **kwargs)


def population(n=1, **kwargs):
    return LIFPopulation(torch.zeros(n, n, dtype=torch.float64),
                         half_life_config(**kwargs))


def test_passive_voltage_halves_without_current():
    brain = population()
    brain.step(torch.tensor([1.6], dtype=torch.float64))
    assert brain.voltage.tolist() == pytest.approx([0.8])
    brain.step(torch.zeros(1, dtype=torch.float64))
    assert brain.voltage.tolist() == pytest.approx([0.4])
    brain.step(torch.zeros(1, dtype=torch.float64))
    assert brain.voltage.tolist() == pytest.approx([0.2])


def test_current_and_leak_are_relative_to_rest():
    brain = population(rest=2, reset=2, threshold=3)
    brain.step(torch.tensor([1.6], dtype=torch.float64))
    assert brain.voltage.tolist() == pytest.approx([2.8])
    brain.step(torch.zeros(1, dtype=torch.float64))
    assert brain.voltage.tolist() == pytest.approx([2.4])


@pytest.mark.parametrize('current,fires,voltage', [
    (1.999, False, 0.9995), (2.0, True, 0.0), (10.0, True, 0.0),
    (-2.0, False, -1.0),
])
def test_threshold_equality_and_overshoot_reset(current, fires, voltage):
    brain = population()
    spikes = brain.step(torch.tensor([current], dtype=torch.float64))
    assert spikes.tolist() == [fires]
    assert brain.voltage.tolist() == pytest.approx([voltage])


def test_no_refractory_and_configured_reset():
    brain = population(reset=0.25)
    for _ in range(3):
        assert brain.step(torch.tensor([4.0], dtype=torch.float64)).item()
        assert brain.voltage.item() == 0.25
    assert brain.step_count == 3


def test_directed_edge_has_one_step_delay_and_no_reverse():
    # W[B,A]=1, W[A,B]=0. Driving A cannot cause a same-step cascade.
    weights = torch.tensor([[0., 0.], [1., 0.]], dtype=torch.float64)
    brain = LIFPopulation(weights, half_life_config())
    assert brain.step(torch.tensor([2., 0.], dtype=torch.float64)).tolist() == [True, False]
    assert brain.step(torch.zeros(2, dtype=torch.float64)).tolist() == [False, True]
    assert brain.step(torch.zeros(2, dtype=torch.float64)).tolist() == [False, False]
    brain.reset()
    assert brain.step(torch.tensor([0., 2.], dtype=torch.float64)).tolist() == [False, True]
    assert brain.step(torch.zeros(2, dtype=torch.float64)).tolist() == [False, False]


def test_reset_clears_pending_propagation_and_replays():
    brain = LIFPopulation(torch.tensor([[0., 0.], [1., 0.]], dtype=torch.float64),
                          half_life_config())
    drive = torch.tensor([2., 1.], dtype=torch.float64)
    first = brain.step(drive)
    brain.reset()
    assert brain.step_count == 0
    assert not brain.spikes.any()
    assert brain.voltage.tolist() == [0., 0.]
    assert not brain.step(torch.zeros(2, dtype=torch.float64)).any()
    brain.reset()
    assert torch.equal(first, brain.step(drive))


def test_weights_state_and_step_results_do_not_alias():
    weights = torch.zeros(1, 1, dtype=torch.float64)
    brain = LIFPopulation(weights, half_life_config())
    weights.fill_(10)
    result = brain.step(torch.tensor([2.], dtype=torch.float64))
    result.zero_()
    brain.voltage.fill_(99)
    brain.weights.fill_(99)
    assert brain.spikes.item()
    assert brain.voltage.item() == 0
    assert not brain.step(torch.zeros(1, dtype=torch.float64)).item()


@pytest.mark.parametrize('kwargs', [
    {'dt': 0}, {'dt': -1}, {'dt': math.nan}, {'tau': 0}, {'tau': math.inf},
    {'threshold': 0}, {'reset': 1}, {'rest': 1}, {'rest': math.nan},
])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LIFConfig(**kwargs)


@pytest.mark.parametrize('weights', [
    torch.zeros(0, 0), torch.zeros(2, 3), torch.zeros(2),
    torch.zeros(1, 1, dtype=torch.int64), torch.zeros(1, 1, dtype=torch.float16),
    torch.tensor([[math.nan]]), torch.tensor([[math.inf]]),
])
def test_reject_invalid_weights(weights):
    with pytest.raises(ValueError):
        LIFPopulation(weights)


@pytest.mark.parametrize('current', [
    torch.zeros(2, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64),
    torch.zeros(1, dtype=torch.float32), torch.tensor([math.nan], dtype=torch.float64),
    torch.tensor([math.inf], dtype=torch.float64),
])
def test_bad_current_does_not_advance_state(current):
    brain = population()
    with pytest.raises(ValueError):
        brain.step(current)
    assert brain.step_count == 0
    assert brain.voltage.item() == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_tensor_contract(dtype):
    brain = LIFPopulation(torch.zeros(3, 3, dtype=dtype))
    result = brain.step(torch.zeros(3, dtype=dtype))
    assert result.shape == (3,)
    assert result.dtype == torch.bool
    assert brain.voltage.dtype == dtype
    assert brain.voltage.device.type == 'cpu'
    assert not brain.voltage.requires_grad


def test_inhibitory_jump_prevents_otherwise_suprathreshold_drive():
    brain = LIFPopulation(torch.tensor([[0., 0.], [-1., 0.]], dtype=torch.float64),
                          half_life_config())
    assert brain.step(torch.tensor([2., 1.5], dtype=torch.float64)).tolist() == [True, False]
    assert brain.step(torch.tensor([0., 1.5], dtype=torch.float64)).tolist() == [False, False]
    assert brain.voltage[1].item() == pytest.approx(0.125)


def test_non_cpu_current_rejected_before_state_changes():
    brain = population()
    with pytest.raises(ValueError):
        brain.step(torch.zeros(1, dtype=torch.float64, device='meta'))
    assert brain.step_count == 0


def test_transition_overflow_leaves_previous_state_intact():
    huge = torch.finfo(torch.float32).max
    brain = LIFPopulation(torch.tensor([[0., 0.], [huge, huge]]), half_life_config())
    brain.step(torch.tensor([2., 2.]))
    before = brain.voltage
    with pytest.raises(ValueError):
        brain.step(torch.zeros(2))
    assert brain.step_count == 1
    assert torch.equal(before, brain.voltage)
    assert brain.spikes.tolist() == [True, True]


def test_dtype_conversion_cannot_collapse_reset_onto_threshold():
    with pytest.raises(ValueError):
        LIFPopulation(torch.zeros(1, 1), LIFConfig(reset=1 - 1e-12))
