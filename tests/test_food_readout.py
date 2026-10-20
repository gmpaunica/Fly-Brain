"""Neural history, learned stop command and checkpoint behavior."""

from pathlib import Path

import torch

from flybrain.neural.food_contact import FoodContactNetwork
from flybrain.neural.food_readout import (FoodDecoder, FoodReadout, load_food_checkpoint,
                                          save_food_checkpoint)
from flybrain.world.food import FoodObservation
from flybrain.world.fly import MotorCommand


def test_contact_relay_history_is_neural_and_resets():
    brain = FoodContactNetwork()
    model = FoodReadout(torch.zeros(22), torch.ones(22))
    decoder = FoodDecoder(brain.topology, model)
    quiet = decoder.decode(brain.advance(FoodObservation(.02, .02, False)))
    assert decoder.feature_rates.shape == (22,)
    assert decoder.feature_rates[-2:].tolist() == [0, 0]
    assert quiet.forward >= 0
    decoder.decode(brain.advance(FoodObservation(.02, .02, True)))
    assert (decoder.feature_rates[-2:] > 0).all()
    assert decoder.feature_rates[-2] > decoder.feature_rates[-1]
    brain.reset()
    decoder.reset()
    assert not decoder.feature_rates.any()


def test_positive_learned_stop_is_exact_zero_and_checkpoint_reloads(tmp_path: Path):
    model = FoodReadout(torch.zeros(22), torch.ones(22))
    assert sum(parameter.numel() for parameter in model.parameters()) == 419
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.copy_(torch.tensor([0., 0., 1.]))
    brain = FoodContactNetwork()
    decoder = FoodDecoder(brain.topology, model)
    command = decoder.decode(brain.advance(FoodObservation(.02, .02, True)))
    assert command == MotorCommand()
    path = tmp_path / 'candidate.pt'
    save_food_checkpoint(path, model, protocol_sha256='a' * 64)
    loaded = load_food_checkpoint(path, protocol_sha256='a' * 64,
                                  mean=torch.zeros(22), scale=torch.ones(22))
    x = torch.zeros((2, 22))
    assert torch.equal(model(x), loaded(x))
