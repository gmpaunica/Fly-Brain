"""Numerical and neural-boundary checks for the endpoint-only learned decoder."""

import math

import pytest
import torch

from flybrain.world.sensing import SensoryObservation


def make_model():
    from flybrain.neural.banc_readout import BancReadout
    return BancReadout(torch.full((8,), 20.), torch.full((8,), 10.), seed=1010)


def test_readout_standardizes_rates_and_orients_destination_source_matrices():
    model = make_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.hidden.weight[0, 0] = 1.
        model.output.weight[1, 0] = .5
    result = model(torch.full((8,), 30.))
    assert result.tolist() == pytest.approx([.5, math.tanh(.5 * math.tanh(1.))])
    assert model(torch.full((3, 8), 30.)).shape == (3, 2)
    assert sum(p.numel() for p in model.parameters()) == 178


def test_initialization_is_reproducible_and_does_not_consume_global_rng():
    before = torch.random.get_rng_state().clone()
    first, second = make_model(), make_model()
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))


@pytest.mark.parametrize('rates', [
    torch.zeros(7), torch.zeros(9), torch.zeros((0, 8)),
    torch.full((8,), float('nan')), torch.full((8,), -1.),
    torch.full((8,), 1001.), torch.ones(8, dtype=torch.int64),
])
def test_readout_rejects_invalid_endpoint_rates(rates):
    with pytest.raises(ValueError):
        make_model()(rates)


def test_learned_decoder_uses_identical_endpoint_filter_and_reset():
    from flybrain.neural.banc_readout import make_readout_controller
    from flybrain.neural.banc_motor import make_banc_controller
    from flybrain.neural.banc_network import load_verified_banc_config
    fixed = make_banc_controller(load_verified_banc_config())
    model = make_model()
    learned = make_readout_controller(model)
    graph = learned.network.population.weights.clone()
    observations = [SensoryObservation(.01, .015)] * 4
    for observation in observations:
        original = fixed.decide(observation)
        decision = learned.decide(observation)
        assert torch.equal(original.activity.spike_counts, decision.activity.spike_counts)
        assert torch.equal(fixed.decoder.feature_rates, learned.decoder.feature_rates)
        expected = model(fixed.decoder.feature_rates.float()).tolist()
        assert [decision.command.forward, decision.command.turn] == expected
    learned.reset()
    assert torch.count_nonzero(learned.decoder.feature_rates) == 0
    fixed.reset()
    first = learned.decide(observations[0])
    assert first.activity.end_step == 20
    assert torch.equal(graph, learned.network.population.weights)
    assert torch.equal(fixed.decide(observations[0]).activity.spike_counts,
                       first.activity.spike_counts)


def test_checkpoint_replays_and_refuses_overwrite(tmp_path):
    from flybrain.neural.banc_readout import save_checkpoint, load_checkpoint
    path = tmp_path / 'model.pt'
    model = make_model()
    save_checkpoint(path, model, protocol_sha256='a' * 64)
    reloaded = load_checkpoint(path, protocol_sha256='a' * 64,
                               mean=model.mean, scale=model.scale)
    rates = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    assert torch.equal(model(rates), reloaded(rates))
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, protocol_sha256='a' * 64)
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        load_checkpoint(path, protocol_sha256='b' * 64,
                        mean=model.mean, scale=model.scale)


@pytest.mark.parametrize('key', [
    'schema', 'subset_sha256', 'dynamics_config', 'effective_weights_sha256',
    'feature_names', 'filter_tau_seconds', 'architecture',
])
def test_checkpoint_rejects_wrong_feature_graph_or_dynamics_contract(tmp_path, key):
    from flybrain.neural.banc_readout import save_checkpoint, load_checkpoint
    model = make_model()
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, protocol_sha256='a' * 64)
    payload = torch.load(path, weights_only=True)
    payload['contract'][key] = 'wrong'
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, protocol_sha256='a' * 64,
                        mean=model.mean, scale=model.scale)


def test_checkpoint_rejects_changed_normalization_and_nonfinite_parameters(tmp_path):
    from flybrain.neural.banc_readout import save_checkpoint, load_checkpoint
    model = make_model()
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, protocol_sha256='a' * 64)
    payload = torch.load(path, weights_only=True)
    payload['state_dict']['mean'][0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, protocol_sha256='a' * 64,
                        mean=model.mean, scale=model.scale)
    payload['state_dict']['mean'] = model.mean.clone()
    payload['state_dict']['hidden.weight'][0, 0] = float('nan')
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, protocol_sha256='a' * 64,
                        mean=model.mean, scale=model.scale)
