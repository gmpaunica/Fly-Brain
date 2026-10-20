"""The first bounded training experiment must remain reproducible and separate."""

import csv
import json
import math

import pytest
import torch

from flybrain.neural.readout import TrainableMotorReadout
from flybrain.training.evaluation import (
    EvaluationConfig, evaluate_cases, locked_cases, run_episode, write_run,
)
from flybrain.training.readout_experiment import (
    collect_training_samples, learned_controller_factory, load_checkpoint,
    run_experiment, save_checkpoint, teacher_command, train_readout,
)
from flybrain.world.sensing import SensoryObservation


def test_teacher_uses_only_bilateral_concentrations_and_turn_sign():
    right = teacher_command(SensoryObservation(0.02, 0.03))
    left = teacher_command(SensoryObservation(0.03, 0.02))
    absent = teacher_command(SensoryObservation(0, 0))

    assert right.tolist() == pytest.approx([0.7, 1.0])
    assert left.tolist() == pytest.approx([0.7, -1.0])
    assert absent.tolist() == pytest.approx([0.7, 0])


def test_collection_uses_locked_train_only_and_has_real_tick_features():
    config = EvaluationConfig(horizon_ticks=3)
    cases = locked_cases()
    dataset = collect_training_samples(cases['train'][:2], config, sample_ticks=3)

    assert dataset.features.shape == (6, 7)
    assert dataset.targets.shape == (6, 2)
    assert dataset.case_counts == {'train-0': 3, 'train-1': 3}
    assert torch.isfinite(dataset.features).all()
    assert torch.isfinite(dataset.targets).all()
    with pytest.raises(ValueError, match='locked train'):
        collect_training_samples(cases['final'][:1], config, sample_ticks=1)


def test_locked_train_starts_have_aliased_spike_features_but_distinct_targets():
    config = EvaluationConfig(horizon_ticks=1)
    dataset = collect_training_samples(locked_cases()['train'], config, sample_ticks=1)

    assert torch.unique(dataset.features, dim=0).shape[0] == 1
    assert torch.unique(dataset.targets[:, 1]).numel() > 1


def test_full_batch_training_records_loss_gradients_and_parameter_updates():
    features = torch.tensor([[0.2] * 7, [0.8] * 7], dtype=torch.float32)
    targets = torch.tensor([[0.7, 0.3], [0.7, -0.3]], dtype=torch.float32)

    outcome = train_readout(features, targets, steps=4, learning_rate=0.05)

    assert len(outcome.curve) == 5
    assert [point['step'] for point in outcome.curve] == list(range(5))
    assert outcome.curve[-1]['loss'] < outcome.curve[0]['loss']
    assert outcome.gradient_nonzero_scalars > 0
    assert outcome.parameter_changed_scalars > 0
    assert math.isfinite(outcome.gradient_min_abs)
    assert math.isfinite(outcome.parameter_change_min_abs)


def test_checkpoint_reloads_identical_embodied_trajectory(tmp_path):
    outcome = train_readout(
        torch.tensor([[0.5] * 7], dtype=torch.float32),
        torch.tensor([[0.7, 0.4]], dtype=torch.float32),
        steps=3, learning_rate=0.05,
    )
    path = tmp_path / 'readout.pt'
    save_checkpoint(path, outcome.model, network_seed=7, alpha=0.5)
    loaded = load_checkpoint(path, network_seed=7)
    case = locked_cases()['validation'][0]
    config = EvaluationConfig(horizon_ticks=4)

    original = run_episode(case, config, learned_controller_factory(outcome.model, 7))
    replay = run_episode(case, config, learned_controller_factory(loaded, 7))
    assert replay == original
    with pytest.raises(ValueError, match='topology'):
        load_checkpoint(path, network_seed=8)


def test_original_checkpoint_rejects_nine_feature_model(tmp_path):
    with pytest.raises(ValueError, match='seven'):
        save_checkpoint(tmp_path / 'wrong.pt', TrainableMotorReadout(9),
                        network_seed=7, alpha=0.5)


def test_short_experiment_records_train_curve_and_matched_held_out_rows(tmp_path):
    config = EvaluationConfig(horizon_ticks=2)
    cases = locked_cases()
    baseline_dir = tmp_path / 'baseline'
    baseline, _ = write_run(
        baseline_dir, config, cases,
        evaluate_cases(cases['validation'] + cases['final'], config),
    )
    bad_baseline = tmp_path / 'bad-baseline.json'
    tampered = json.loads(baseline.read_text(encoding='utf-8'))
    tampered['results'][0]['case_id'] = 'train-0'
    bad_baseline.write_text(json.dumps(tampered), encoding='utf-8')
    with pytest.raises(ValueError, match='baseline protocol'):
        run_experiment(tmp_path / 'rejected', bad_baseline, config=config,
                       sample_ticks=1, steps=2)
    output_dir = tmp_path / 'trained'

    result_path = run_experiment(
        output_dir, baseline, config=config, sample_ticks=1,
        steps=2, learning_rate=0.05,
    )
    payload = json.loads(result_path.read_text(encoding='utf-8'))
    with (output_dir / 'curve.csv').open(newline='', encoding='utf-8') as file:
        curve = list(csv.DictReader(file))
    with (output_dir / 'episodes.csv').open(newline='', encoding='utf-8') as file:
        episodes = list(csv.DictReader(file))

    assert payload['training_steps'] == 2
    assert payload['train_sample_count'] == 8
    assert payload['gradient_nonzero_scalars'] > 0
    assert payload['parameter_changed_scalars'] > 0
    assert payload['topology_unchanged'] is True
    assert payload['reload_replay_exact'] is True
    assert payload['first_tick_unique_feature_vectors'] == 1
    assert payload['first_tick_unique_teacher_turns'] > 1
    assert [row['step'] for row in curve] == ['0', '1', '2']
    assert len(episodes) == 12
    assert {row['split'] for row in episodes} == {'validation', 'final'}
    assert [row['case_id'] for row in episodes] == [
        case.case_id for case in cases['validation'] + cases['final']
    ]
    assert (output_dir / 'curve.svg').read_text(encoding='utf-8').startswith('<svg')
    assert (output_dir / 'readout.pt').is_file()
