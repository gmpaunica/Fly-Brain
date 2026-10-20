"""Numerical contracts for the small learned motor readout."""

import math

import pytest
import torch

from flybrain.neural.network import NeuralActivity, NeuralSnapshot
from flybrain.neural.readout import (
    GROUP_NAMES, LearnedMotorDecoder, SpikeFeatureFilter, TrainableMotorReadout,
)
from flybrain.neural.topology import build_synthetic_topology
from flybrain.world.fly import MotorCommand


def activity_with_group_counts(**totals: int) -> NeuralActivity:
    topology = build_synthetic_topology()
    counts = torch.zeros(100, dtype=torch.int64)
    for group, total in totals.items():
        counts[list(topology.indices(group))[0]] = total
    snapshot = NeuralSnapshot(20, torch.zeros(100), torch.zeros(100, dtype=torch.bool))
    return NeuralActivity(0, 20, 0.02, counts, snapshot)


def test_filter_uses_group_mean_and_resets_ema():
    topology = build_synthetic_topology()
    feature_filter = SpikeFeatureFilter(topology, alpha=0.5)
    activity = activity_with_group_counts(sensory_left=10, motor_right=10)

    first = feature_filter.update(activity)
    second = feature_filter.update(activity)
    assert first.shape == (7,)
    assert first[GROUP_NAMES.index('sensory_left')].item() == pytest.approx(0.5)
    assert first[GROUP_NAMES.index('motor_right')].item() == pytest.approx(0.5)
    assert second[GROUP_NAMES.index('sensory_left')].item() == pytest.approx(0.75)
    assert first[GROUP_NAMES.index('motor_forward')].item() == 0
    feature_filter.reset()
    assert torch.equal(feature_filter.update(activity), first)


def test_gradient_updates_every_readout_scalar_without_changing_topology():
    topology = build_synthetic_topology()
    before = topology.weights
    model = TrainableMotorReadout()
    inputs = torch.arange(1, 8, dtype=torch.float32).unsqueeze(0) / 10
    target = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    previous = {name: value.detach().clone() for name, value in model.named_parameters()}

    loss = torch.nn.functional.mse_loss(model(inputs), target)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) == parameter.numel()
    optimizer.step()

    for name, parameter in model.named_parameters():
        assert torch.count_nonzero(parameter.detach() - previous[name]) == parameter.numel()
    assert torch.equal(topology.weights, before)


def test_decoder_bounds_commands_and_restarts_without_model_change():
    topology = build_synthetic_topology()
    model = TrainableMotorReadout()
    with torch.no_grad():
        model.linear.bias.copy_(torch.tensor([30.0, -30.0]))
    decoder = LearnedMotorDecoder(topology, model)
    activity = activity_with_group_counts(motor_forward=10)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    command = decoder.decode(activity)
    assert isinstance(command, MotorCommand)
    assert 0 <= command.forward <= 1
    assert -1 <= command.turn <= 1
    assert math.isfinite(command.forward) and math.isfinite(command.turn)
    decoder.reset()
    assert decoder.decode(activity) == command
    assert all(torch.equal(value, before[name]) for name, value in model.named_parameters())
