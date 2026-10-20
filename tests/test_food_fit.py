"""One fixed readout objective updates parameters using real PyTorch gradients."""

import torch

from flybrain.neural.food_readout import FoodReadout
from flybrain.training.food_fit import fit_readout


def test_fit_has_finite_nonzero_gradients_and_changes_only_readout():
    features = torch.zeros((64, 22), dtype=torch.float32)
    features[:32, :20] = .5
    features[32:, 20:] = 60
    targets = torch.tensor([[.6, .3, 0.]] * 32 + [[0., 0., 1.]] * 32)
    model = FoodReadout(torch.zeros(22), torch.ones(22), seed=1010)
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    record = fit_readout(model, features, targets, updates=10)
    assert record['optimizer_steps'] == 10
    assert record['gradient_nonzero_scalars'] > 0
    assert record['parameter_changed_scalars'] > 0
    assert record['failure'] is None
    assert any(not torch.equal(old, new) for old, new in zip(initial, model.parameters()))
