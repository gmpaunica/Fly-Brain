"""Train-only protocol integrity and artificial-data optimizer checks."""

import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / 'experiments/fb09-banc-readiness'


def prepare(tmp_path):
    from flybrain.training.banc_protocol import prepare_protocol
    return prepare_protocol(tmp_path / 'fb10', readiness_dir=READINESS)


def test_preparation_binds_readiness_and_normalization_without_executing(tmp_path):
    from flybrain.training.banc_protocol import load_protocol, load_readiness
    path = prepare(tmp_path)
    protocol, digest, dataset = load_protocol(path)
    assert dataset.features.shape == (1008, 8)
    assert dataset.targets.shape == (1008, 2)
    assert len(digest) == 64
    assert protocol['optimizer_step_ceiling'] == 500
    assert protocol['max_world_ticks'] == 15000
    assert all(case['split'] == 'train' for case in protocol['cases'])
    assert not (path.parent / 'execution.json').exists()
    assert not list(path.parent.glob('*.pt'))
    assert not (path.parent / 'results.json').exists()
    expected_mean = dataset.features.double().mean(dim=0).float()
    expected_scale = dataset.features.double().std(dim=0, correction=0).clamp_min(1).float()
    assert torch.equal(expected_mean, torch.tensor(protocol['normalization']['mean']))
    assert torch.equal(expected_scale, torch.tensor(protocol['normalization']['scale']))
    assert torch.equal(dataset.features, load_readiness(READINESS).features)


def test_preparation_preserves_partial_protocol_and_stale_bytes_stop_execution(tmp_path):
    from flybrain.training.banc_training import run_training
    path = prepare(tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(tmp_path)
    assert path.read_bytes() == original
    path.write_bytes(original + b' ')
    with pytest.raises(ValueError, match='hash'):
        run_training(path)
    assert not (path.parent / 'execution.json').exists()


def test_prepared_protocol_is_single_use_even_after_partial_execution(tmp_path):
    from flybrain.training.banc_training import run_training
    path = prepare(tmp_path)
    receipt = path.parent / 'execution.json'
    receipt.write_text('{"status":"started"}\n', encoding='utf-8')
    with pytest.raises(FileExistsError):
        run_training(path)
    assert receipt.read_text(encoding='utf-8') == '{"status":"started"}\n'


def test_copying_protocol_cannot_reopen_the_same_attempt_at_another_destination(tmp_path):
    from flybrain.training.banc_protocol import load_protocol
    path = prepare(tmp_path)
    copied = tmp_path / 'copy'
    copied.mkdir()
    for name in ('protocol.json', 'protocol.sha256'):
        (copied / name).write_bytes((path.parent / name).read_bytes())
    with pytest.raises(ValueError, match='destination'):
        load_protocol(copied / 'protocol.json')


@pytest.mark.parametrize('key,value', [
    ('seed', 7), ('optimizer_step_ceiling', 501),
    ('normalization', {'mean': [0.] * 8, 'scale': [1.] * 8}),
])
def test_resealed_protocol_still_rejects_different_candidate_or_normalization(tmp_path, key, value):
    from flybrain.training.banc_protocol import load_protocol, sha256
    path = prepare(tmp_path)
    protocol = json.loads(path.read_text(encoding='utf-8'))
    protocol[key] = value
    path.write_text(json.dumps(protocol), encoding='utf-8')
    path.with_suffix('.sha256').write_text(sha256(path), encoding='utf-8')
    with pytest.raises(ValueError):
        load_protocol(path)


def test_readiness_loader_rejects_corrupted_samples(tmp_path):
    from flybrain.training.banc_protocol import load_readiness
    for name in ('samples.csv', 'results.json', 'protocol.json'):
        (tmp_path / name).write_bytes((READINESS / name).read_bytes())
    with (tmp_path / 'samples.csv').open('a', encoding='utf-8') as stream:
        stream.write('bad row\n')
    with pytest.raises(ValueError, match='hash'):
        load_readiness(tmp_path)


def test_readiness_loader_rejects_historical_validation_labels(tmp_path):
    import hashlib
    from flybrain.training.banc_protocol import load_readiness
    for name in ('samples.csv', 'results.json', 'protocol.json'):
        (tmp_path / name).write_bytes((READINESS / name).read_bytes())
    samples = tmp_path / 'samples.csv'
    samples.write_bytes(samples.read_bytes().replace(b'train-0', b'validation-100'))
    result = json.loads((tmp_path / 'results.json').read_text(encoding='utf-8'))
    result['samples_sha256'] = hashlib.sha256(samples.read_bytes()).hexdigest()
    (tmp_path / 'results.json').write_text(json.dumps(result), encoding='utf-8')
    with pytest.raises(ValueError, match='train'):
        load_readiness(tmp_path)


def test_artificial_fit_changes_parameters_with_gradients_and_frozen_normalization():
    from flybrain.neural.banc_readout import BancReadout
    from flybrain.training.banc_training import fit_readout
    features = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    targets = torch.tensor([[.7, -.8], [.7, -.2], [.7, .2], [.7, .8]])
    model = BancReadout(features.mean(0), features.std(0, correction=0), seed=1010)
    mean, scale = model.mean.clone(), model.scale.clone()
    result = fit_readout(model, features, targets, updates=3)
    assert result['optimizer_steps'] == 3
    assert result['gradient_nonzero_scalars'] > 0
    assert result['parameter_changed_scalars'] > 0
    assert len(result['curve']) == 4
    assert [row['update'] for row in result['curve']] == [0, 1, 2, 3]
    assert result['curve'][-1]['turn_mse'] < result['curve'][0]['turn_mse']
    assert torch.equal(mean, model.mean) and torch.equal(scale, model.scale)


def test_nonfinite_fit_stops_before_any_update():
    from flybrain.neural.banc_readout import BancReadout
    from flybrain.training.banc_training import fit_readout
    model = BancReadout(torch.zeros(8), torch.ones(8), seed=1010)
    with torch.no_grad():
        model.hidden.weight.fill_(float('nan'))
    result = fit_readout(model, torch.ones((2, 8)), torch.zeros((2, 2)), updates=3)
    assert result['optimizer_steps'] == 0
    assert result['failure'] is not None
    assert result['curve'] == []


def test_fit_requires_a_sample_matrix_rather_than_broadcasting_one_feature_vector():
    from flybrain.neural.banc_readout import BancReadout
    from flybrain.training.banc_training import fit_readout
    model = BancReadout(torch.zeros(8), torch.ones(8), seed=1010)
    with pytest.raises(ValueError):
        fit_readout(model, torch.ones(8), torch.zeros((8, 2)), updates=1)


def test_behavior_rejects_cases_that_are_only_named_train():
    from dataclasses import replace
    from flybrain.neural.banc_readout import BancReadout
    from flybrain.training.banc_training import evaluate_training_models
    from flybrain.training.evaluation import EvaluationConfig, locked_cases
    model = BancReadout(torch.zeros(8), torch.ones(8), seed=1010)
    mislabeled = replace(locked_cases()['validation'][0], split='train')
    with pytest.raises(ValueError):
        evaluate_training_models(model, model, model, [mislabeled], EvaluationConfig(horizon_ticks=1))


@pytest.mark.parametrize('updates', [0, 501, True, 2.5])
def test_fit_refuses_updates_outside_budget(updates):
    from flybrain.neural.banc_readout import BancReadout
    from flybrain.training.banc_training import fit_readout
    model = BancReadout(torch.zeros(8), torch.ones(8), seed=1010)
    with pytest.raises(ValueError):
        fit_readout(model, torch.ones((2, 8)), torch.zeros((2, 2)), updates=updates)


def test_turn_gate_cannot_pass_from_forward_improvement_or_tied_arrivals():
    from flybrain.training.banc_training import training_gates
    summaries = {
        'trained': {'successes': 7, 'mean_progress_fraction': .8},
        'untrained': {'successes': 0, 'mean_progress_fraction': .1},
        'fixed': {'successes': 0, 'mean_progress_fraction': -.2},
    }
    assert training_gates(.02, .089, summaries, {'checks': True})['passed']
    assert not training_gates(.03, .089, summaries, {'checks': True})['passed']
    summaries['untrained']['successes'] = 7
    assert not training_gates(.02, .089, summaries, {'checks': True})['passed']


def test_real_two_tick_evaluation_reloads_replays_and_accounts_all_steps(tmp_path):
    from flybrain.neural.banc_readout import BancReadout, save_checkpoint, load_checkpoint
    from flybrain.training.banc_training import evaluate_training_models
    from flybrain.training.evaluation import EvaluationConfig, locked_cases
    model = BancReadout(torch.zeros(8), torch.ones(8), seed=1010)
    path = tmp_path / 'tiny.pt'
    save_checkpoint(path, model, protocol_sha256='a' * 64)
    reloaded = load_checkpoint(path, protocol_sha256='a' * 64,
                               mean=model.mean, scale=model.scale)
    result = evaluate_training_models(model, reloaded, model,
              locked_cases()['train'][:1], EvaluationConfig(horizon_ticks=2))
    assert result['actual_world_ticks'] == 8  # three controls plus one replay
    assert result['actual_neural_steps'] == 160
    assert result['checkpoint_replay_exact']
    assert result['effective_weights_unchanged']
    assert set(result['episodes']) == {'trained', 'untrained', 'fixed'}
