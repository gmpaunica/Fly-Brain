"""Numerical contracts for the one bounded M6 readout fit."""

import pytest
import torch

from flybrain.neural.readout import TrainableMotorReadout
from flybrain.training.m6_recovery import fit_conditioned_readout, fold_standardization


def test_fold_standardization_preserves_both_bounded_outputs():
    model = TrainableMotorReadout(9)
    with torch.no_grad():
        model.linear.weight[0, :2] = torch.tensor([0.4, -0.2])
        model.linear.weight[1, :2] = torch.tensor([-0.3, 0.7])
        model.linear.bias[:] = torch.tensor([0.1, -0.4])
    mean = torch.tensor([2.0, -1.0] + [0.0] * 7)
    scale = torch.tensor([0.5, 4.0] + [1.0] * 7)
    raw = torch.tensor([[2.5, 3.0] + [0.0] * 7,
                        [1.5, -5.0] + [0.0] * 7])

    folded = fold_standardization(model, mean, scale)

    assert torch.allclose(folded(raw), model((raw - mean) / scale), atol=1e-7)
    assert folded.linear.weight[1, 0].item() == pytest.approx(-0.6)
    assert folded.linear.weight[1, 1].item() == pytest.approx(0.175)
    assert folded.linear.bias[1].item() == pytest.approx(0.975)


def test_conditioned_fit_has_finite_turn_gradient_and_budget_guard():
    features = torch.tensor([[i / 10 + j / 100 for j in range(9)]
                             for i in range(6)], dtype=torch.float32)
    targets = torch.tensor([[0.7, turn] for turn in
                            (-0.8, -0.4, -0.1, 0.1, 0.4, 0.8)])

    outcome = fit_conditioned_readout(features, targets, steps=5)

    assert len(outcome.curve) == 6
    assert outcome.gradient_nonzero_scalars > 0
    assert outcome.parameter_changed_scalars > 0
    assert outcome.curve[-1]['turn_mse'] < outcome.curve[0]['turn_mse']
    assert torch.isfinite(outcome.model(features)).all()
    with pytest.raises(ValueError, match='500'):
        fit_conditioned_readout(features, targets, steps=501)
