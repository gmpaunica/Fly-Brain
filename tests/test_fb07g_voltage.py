"""FB-07G: post-interval voltage information on locked sensory replays."""

import csv
import json

import pytest
import torch

from flybrain.neural.network import SyntheticNetwork
from flybrain.training.evaluation import EvaluationConfig, locked_cases
from flybrain.training.fb07f_collision import StageTrace
from flybrain.training.fb07g_voltage import (
    feature_metrics, run_diagnostic, sensory_voltage_means, trace_cases,
)
from flybrain.world.sensing import SensoryObservation


def test_voltage_means_use_post_interval_bilateral_state_and_reset():
    network = SyntheticNetwork(encoder_mode='rational', encoder_scale=0.005)
    observation = SensoryObservation(0.007, 0.012)
    activity = network.advance(observation)
    voltage = sensory_voltage_means(activity, network.topology)
    left = network.topology.indices('sensory_left')
    right = network.topology.indices('sensory_right')

    assert voltage.shape == (2,)
    assert voltage.tolist() == pytest.approx([
        activity.snapshot.voltage[list(left)].mean().item(),
        activity.snapshot.voltage[list(right)].mean().item(),
    ])
    assert not voltage.requires_grad
    network.reset()
    assert sensory_voltage_means(network.advance(observation),
                                 network.topology).tolist() == pytest.approx(voltage.tolist())


def test_hybrid_features_separate_equal_spikes_and_lower_turn_floor():
    traces = [
        StageTrace((0.01, 0.02), -1.0, (1.0, 2.0), (0,), (0,),
                   (0.0,), (0.0,) * 7, (0.2, 0.3)),
        StageTrace((0.02, 0.01), 1.0, (2.0, 1.0), (0,), (0,),
                   (0.0,), (0.0,) * 7, (0.4, 0.3)),
    ]

    result = feature_metrics(traces)

    assert result['ema']['aliased_ticks'] == 2
    assert result['ema']['turn_mse_floor'] == 1.0
    assert result['hybrid']['aliased_ticks'] == 0
    assert result['hybrid']['turn_mse_floor'] == 0.0
    assert result['conflicting_ema_pairs'] == 1
    assert result['pairs_resolved_by_voltage'] == 1


def test_short_replay_is_reset_deterministic_and_excludes_final():
    config = EvaluationConfig(horizon_ticks=3)
    cases = locked_cases()['train'][:2]
    result = trace_cases(cases, config, ticks=3)

    assert result == trace_cases(cases, config, ticks=3)
    assert [row['case_id'] for row in result['cases']] == ['train-0', 'train-1']
    assert result['features']['hybrid']['ticks'] == 6
    assert result['feature_size'] == 9
    assert result['topology_unchanged'] is True
    with pytest.raises(ValueError, match='locked train or validation'):
        trace_cases(locked_cases()['final'][:1], config, ticks=1)


def test_artifact_records_twelve_cases_and_no_learning_or_final(tmp_path):
    path = run_diagnostic(tmp_path, config=EvaluationConfig(horizon_ticks=2),
                          train_ticks=2, validation_ticks=2)
    result = json.loads(path.read_text(encoding='utf-8'))
    with (tmp_path / 'cases.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))

    assert result['experiment'] == 'fb07g_voltage_features'
    assert len(rows) == 12
    assert result['probe']['train']['features']['hybrid']['ticks'] == 16
    assert result['probe']['validation']['features']['hybrid']['ticks'] == 8
    assert result['feature_size'] == 9
    assert result['topology_unchanged'] is True
    assert result['training_run'] is False
    assert result['final_cases_evaluated'] is False
    assert 'final' not in result['cases']
