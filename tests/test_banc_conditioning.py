"""Behavior and integrity checks for the measured endpoint conditioning repair."""

import math
from pathlib import Path

import pytest
import torch


def model():
    from flybrain.neural.banc_conditioning import ConditionedBancReadout
    return ConditionedBancReadout(torch.zeros(8), torch.ones(8), seed=1010)


def test_pair_transform_preserves_mean_and_right_minus_left_information():
    from flybrain.neural.banc_conditioning import contrast_features
    rates = torch.tensor([10., 14., 20., 30., 40., 24., 36., 48.])
    expected = torch.tensor([12., 22., 33., 44., 4., 4., 6., 8.])
    assert torch.equal(contrast_features(rates), expected)
    assert torch.equal(contrast_features(rates.repeat(2, 1)), expected.repeat(2, 1))


def test_contrast_scale_restores_small_directional_signal_before_matrix_product():
    from flybrain.neural.banc_conditioning import ConditionedBancReadout
    mean = torch.tensor([142.] * 4 + [0.] * 4)
    scale = torch.tensor([72.] * 4 + [4.] * 4)
    m = ConditionedBancReadout(mean, scale)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.hidden.weight[0, 4] = 1
        m.output.weight[1, 0] = .5
    rates = torch.tensor([140., 144., 140., 140., 140., 144., 144., 144.])
    assert m(rates).tolist() == pytest.approx([.5, math.tanh(.5 * math.tanh(1))])
    assert sum(p.numel() for p in m.parameters()) == 178


def test_signed_train_only_normalization_and_local_initialization():
    from flybrain.neural.banc_conditioning import ConditionedBancReadout, normalization
    rates = torch.tensor([[20., 10., 20., 20., 20., 10., 10., 10.],
                          [30., 10., 30., 30., 30., 10., 10., 10.]])
    mean, scale = normalization(rates)
    assert mean.tolist() == [17.5] * 4 + [-15.] * 4
    assert scale.tolist() == [2.5] * 4 + [5.] * 4
    state = torch.random.get_rng_state().clone()
    a, b = ConditionedBancReadout(mean, scale), ConditionedBancReadout(mean, scale)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(a(rates), b(rates))


@pytest.mark.parametrize('rates', [torch.zeros(7), torch.zeros((0, 8)),
    torch.ones(8, dtype=torch.int64), torch.full((8,), -1.),
    torch.full((8,), 1001.), torch.full((8,), float('nan'))])
def test_invalid_endpoint_rates_fail_closed(rates):
    with pytest.raises(ValueError):
        model()(rates)


def test_artificial_fit_has_gradients_changes_and_fixed_buffers():
    from flybrain.training.banc_training import fit_readout
    m = model(); mean, scale = m.mean.clone(), m.scale.clone()
    x = torch.tensor([[1., 2., 1., 1., 1., 2., 2., 2.],
                      [2., 1., 2., 2., 2., 1., 1., 1.]])
    result = fit_readout(m, x, torch.tensor([[.7, .8], [.7, -.8]]), updates=3)
    assert result['failure'] is None and result['optimizer_steps'] == 3
    assert result['gradient_nonzero_scalars'] > 0 and result['parameter_changed_scalars'] > 0
    assert torch.equal(mean, m.mean) and torch.equal(scale, m.scale)


def test_checkpoint_exact_output_and_exclusive_save(tmp_path):
    from flybrain.neural.banc_conditioning import save_checkpoint, load_checkpoint
    m = model(); path = tmp_path / 'model.pt'
    save_checkpoint(path, m, protocol_sha256='a' * 64)
    loaded = load_checkpoint(path, protocol_sha256='a' * 64, mean=m.mean, scale=m.scale)
    assert torch.equal(m(torch.arange(8).float()), loaded(torch.arange(8).float()))
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(path, m, protocol_sha256='a' * 64)
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        load_checkpoint(path, protocol_sha256='b' * 64, mean=m.mean, scale=m.scale)


@pytest.mark.parametrize('key', ['conditioning', 'feature_names', 'effective_weights_sha256'])
def test_checkpoint_rejects_wrong_transform_features_or_graph(tmp_path, key):
    from flybrain.neural.banc_conditioning import save_checkpoint, load_checkpoint
    m = model(); path = tmp_path / 'model.pt'
    save_checkpoint(path, m, protocol_sha256='a' * 64)
    payload = torch.load(path, weights_only=True)
    payload['contract'][key] = 'wrong'
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, protocol_sha256='a' * 64, mean=m.mean, scale=m.scale)


def test_conditioned_decoder_has_identical_spikes_filter_reset_and_two_tick_replay(tmp_path):
    from flybrain.neural.banc_conditioning import save_checkpoint, load_checkpoint
    from flybrain.neural.banc_readout import make_readout_controller
    from flybrain.neural.banc_motor import make_banc_controller
    from flybrain.neural.banc_network import load_verified_banc_config
    from flybrain.training.banc_training import evaluate_training_models
    from flybrain.training.evaluation import EvaluationConfig, locked_cases
    from flybrain.world.sensing import SensoryObservation
    m = model(); learned = make_readout_controller(m)
    fixed = make_banc_controller(load_verified_banc_config())
    for _ in range(3):
        a, b = learned.decide(SensoryObservation(.01, .02)), fixed.decide(SensoryObservation(.01, .02))
        assert torch.equal(a.activity.spike_counts, b.activity.spike_counts)
        assert torch.equal(learned.decoder.feature_rates, fixed.decoder.feature_rates)
    learned.reset()
    assert torch.count_nonzero(learned.decoder.feature_rates) == 0
    path = tmp_path / 'model.pt'; save_checkpoint(path, m, protocol_sha256='a' * 64)
    loaded = load_checkpoint(path, protocol_sha256='a' * 64, mean=m.mean, scale=m.scale)
    result = evaluate_training_models(m, loaded, m, locked_cases()['train'][:1], EvaluationConfig(horizon_ticks=2))
    assert result['checkpoint_replay_exact'] and result['effective_weights_unchanged']
    assert result['actual_world_ticks'] == 8 and result['actual_neural_steps'] == 160


def test_recovery_protocol_is_train_only_bound_and_single_use(tmp_path):
    from flybrain.training.banc_conditioning import prepare_protocol, load_protocol, run_training
    path = prepare_protocol(tmp_path / 'run')
    p, _, data = load_protocol(path)
    assert p['optimizer_step_ceiling'] == 500 and p['sample_count'] == 1008
    assert data.features.shape == (1008, 8)
    assert all(c['split'] == 'train' for c in p['cases'])
    copied = tmp_path / 'copy'; copied.mkdir()
    for name in ('protocol.json', 'protocol.sha256'):
        (copied / name).write_bytes((path.parent / name).read_bytes())
    with pytest.raises(ValueError, match='destination'):
        load_protocol(copied / 'protocol.json')
    (path.parent / 'execution.json').write_text('{}')
    with pytest.raises(FileExistsError):
        run_training(path)


def test_modified_or_resealed_candidate_is_rejected(tmp_path):
    import json, hashlib
    from flybrain.training.banc_conditioning import prepare_protocol, load_protocol
    path = prepare_protocol(tmp_path / 'run')
    p = json.loads(path.read_text()); p['seed'] = 1011
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        load_protocol(path)
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        load_protocol(path)
