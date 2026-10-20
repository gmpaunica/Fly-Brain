"""Matched seven/nine-feature learning and validation gate for FB-07H."""

import csv
import json

import pytest
import torch

from flybrain.neural.network import NeuralActivity, NeuralSnapshot
from flybrain.neural.readout import (
    HybridFeatureFilter, TrainableMotorReadout,
)
from flybrain.neural.topology import build_synthetic_topology
from flybrain.training.evaluation import (
    EvaluationConfig, evaluate_cases, locked_cases, run_episode, write_run,
)
from flybrain.training.fb07h_training import (
    candidate_controller_factory, collect_paired_samples, load_matched_checkpoint,
    run_experiment, save_matched_checkpoint, validation_gate,
)
from flybrain.training.readout_experiment import train_readout


def test_hybrid_filter_appends_post_interval_voltages_and_resets():
    topology = build_synthetic_topology()
    counts = torch.zeros(100, dtype=torch.int64)
    counts[topology.indices('sensory_left')[0]] = 10
    voltage = torch.zeros(100)
    voltage[list(topology.indices('sensory_left'))] = 0.2
    voltage[list(topology.indices('sensory_right'))] = 0.4
    activity = NeuralActivity(0, 20, 0.02, counts,
                              NeuralSnapshot(20, voltage, torch.zeros(100, dtype=torch.bool)))
    feature_filter = HybridFeatureFilter(topology, alpha=0.5)

    first = feature_filter.update(activity)
    assert first.shape == (9,)
    assert first.tolist() == pytest.approx([0.5, 0, 0, 0, 0, 0, 0, 0.2, 0.4])
    assert feature_filter.update(activity)[0].item() == pytest.approx(0.75)
    feature_filter.reset()
    assert torch.equal(feature_filter.update(activity), first)
    assert not first.requires_grad


def test_nine_input_readout_has_gradients_and_bounded_command():
    features = torch.tensor([[0.2] * 7 + [0.1, 0.4],
                             [0.8] * 7 + [0.3, 0.2]], dtype=torch.float32)
    targets = torch.tensor([[0.7, 0.3], [0.7, -0.2]], dtype=torch.float32)
    outcome = train_readout(features, targets, steps=4, learning_rate=0.05)

    assert outcome.model.feature_count == 9
    assert sum(p.numel() for p in outcome.model.parameters()) == 20
    assert outcome.gradient_nonzero_scalars == 20
    assert outcome.parameter_changed_scalars == 20
    assert outcome.curve[-1]['loss'] < outcome.curve[0]['loss']
    assert outcome.model(features).shape == (2, 2)
    assert ((outcome.model(features)[:, 0] >= 0) &
            (outcome.model(features)[:, 0] <= 1)).all()
    with pytest.raises(ValueError, match='nine'):
        outcome.model(torch.zeros(7))


def test_paired_collection_uses_same_train_ticks_and_rejects_final():
    config = EvaluationConfig(horizon_ticks=3)
    seven, nine = collect_paired_samples(locked_cases()['train'][:2], config,
                                         sample_ticks=3)

    assert seven.features.shape == (6, 7)
    assert nine.features.shape == (6, 9)
    assert torch.equal(seven.features, nine.features[:, :7])
    assert torch.equal(seven.targets, nine.targets)
    assert seven.case_counts == nine.case_counts == {'train-0': 3, 'train-1': 3}
    assert torch.isfinite(nine.features).all()
    with pytest.raises(ValueError, match='locked train'):
        collect_paired_samples(locked_cases()['final'][:1], config, sample_ticks=1)


def test_checkpoint_rejects_feature_mismatch_and_replays_embodied_case(tmp_path):
    model = TrainableMotorReadout(feature_count=9)
    with torch.no_grad():
        model.linear.weight[1, 7] = 1.0
        model.linear.weight[1, 8] = -1.0
    path = tmp_path / 'nine.pt'
    save_matched_checkpoint(path, model, network_seed=7)
    loaded = load_matched_checkpoint(path, network_seed=7, feature_count=9)
    case = locked_cases()['validation'][0]
    config = EvaluationConfig(horizon_ticks=4)

    assert run_episode(case, config, candidate_controller_factory(model, 7)) == (
        run_episode(case, config, candidate_controller_factory(loaded, 7)))
    with pytest.raises(ValueError, match='feature'):
        load_matched_checkpoint(path, network_seed=7, feature_count=7)
    with pytest.raises(ValueError, match='topology'):
        load_matched_checkpoint(path, network_seed=8, feature_count=9)


def test_gate_requires_nine_arrivals_above_both_controls_and_progress():
    fixed = {'successes': 0, 'mean_progress_fraction': -0.2}
    seven = {'successes': 1, 'mean_progress_fraction': 0.1}
    nine = {'successes': 2, 'mean_progress_fraction': 0.2}

    assert validation_gate(nine, seven, fixed)['passed'] is True
    assert validation_gate({**nine, 'successes': 1}, seven, fixed)['passed'] is False
    assert validation_gate({**nine, 'mean_progress_fraction': -0.3},
                           seven, fixed)['passed'] is False


def test_short_run_records_matched_validation_and_keeps_final_sealed(tmp_path):
    config = EvaluationConfig(horizon_ticks=2)
    cases = locked_cases()
    baseline, _ = write_run(
        tmp_path / 'baseline', config, cases,
        evaluate_cases(cases['validation'] + cases['final'], config),
    )
    path = run_experiment(tmp_path / 'trained', baseline, config=config,
                          sample_ticks=2, steps=2)
    result = json.loads(path.read_text(encoding='utf-8'))
    with (tmp_path / 'trained' / 'validation.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))

    assert result['experiment'] == 'fb07h_matched_voltage_readout'
    assert result['train_sample_count'] == 16
    assert result['models']['seven']['feature_count'] == 7
    assert result['models']['nine']['feature_count'] == 9
    assert result['models']['nine']['trainable_scalar_count'] == 20
    assert result['models']['nine']['reload_replay_exact'] is True
    assert result['topology_unchanged'] is True
    assert [row['case_id'] for row in rows] == [case.case_id for case in cases['validation']]
    assert not result['final_cases_evaluated']
    assert 'final' not in result['results']
    assert not (tmp_path / 'trained' / 'final.csv').exists()
