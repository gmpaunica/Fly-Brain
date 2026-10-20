"""FB-07E: a fixed weak-signal encoder probe, without training."""

import csv
from copy import deepcopy
import json
import math

import pytest
import torch

from flybrain.neural.network import SyntheticNetwork
from flybrain.training.evaluation import EvaluationConfig, locked_cases
from flybrain.training.fb07d_diagnosis import feature_aliasing
from flybrain.training.fb07e_probe import (
    controlled_weak_probe, decide_gate, probe_cases, run_diagnostic,
)
from flybrain.world.sensing import SensoryObservation


def test_rational_current_is_zero_finite_monotone_and_keeps_default_unchanged():
    candidate = SyntheticNetwork(seed=7, encoder_mode='rational', encoder_scale=0.005)
    original = SyntheticNetwork(seed=7)

    assert torch.equal(candidate.topology.weights, original.topology.weights)
    assert candidate.encode(SensoryObservation(0, 0))[0].item() == 0
    assert candidate.encode(SensoryObservation(0.005, 0.01))[0].item() == pytest.approx(1.25)
    assert candidate.encode(SensoryObservation(0.005, 0.01))[10].item() == pytest.approx(5 / 3)
    assert original.encode(SensoryObservation(0.005, 0.01))[0].item() == pytest.approx(0.4)
    currents = [candidate.encode(SensoryObservation(value, value))[0].item()
                for value in (0, 0.005, 0.01, 0.1, 1e300)]
    assert currents == sorted(currents)
    assert all(math.isfinite(value) and 0 <= value <= 2.5 for value in currents)
    with pytest.raises(ValueError, match='mode'):
        SyntheticNetwork(encoder_mode='unknown')


def test_rational_first_tick_distinguishes_reversed_weak_antenna_readings():
    left = SensoryObservation(0.007, 0.012)
    right = SensoryObservation(0.012, 0.007)

    def sensory_totals(mode, scale, observation):
        network = SyntheticNetwork(encoder_mode=mode, encoder_scale=scale)
        counts = network.advance(observation).spike_counts
        return (counts[:10].sum().item(), counts[10:20].sum().item())

    assert sensory_totals('linear', 0.025, left) == (0, 0)
    assert sensory_totals('linear', 0.05, left) == (0, 0)
    assert sensory_totals('rational', 0.005, left) == (0, 10)
    assert sensory_totals('rational', 0.005, right) == (10, 0)


def test_controlled_weak_probe_records_same_inputs_for_all_encoders():
    probe = controlled_weak_probe(network_seed=7)

    assert probe['observations'] == [[0.007, 0.012], [0.012, 0.007]]
    assert probe['original']['sensory_spike_totals'] == [[0, 0], [0, 0]]
    assert probe['wider']['sensory_spike_totals'] == [[0, 0], [0, 0]]
    assert probe['candidate']['sensory_spike_totals'] == [[0, 10], [10, 0]]


def test_alias_metric_omits_hard_saturation_for_smooth_encoder():
    observations = [SensoryObservation(0.01, 0.02)]
    features = torch.zeros((1, 7))

    result = feature_aliasing(observations, features, encoder_scale=None)

    assert result['ticks'] == 1
    assert 'both_saturated_ticks' not in result


def test_probe_replays_same_locked_observations_and_resets_per_case():
    config = EvaluationConfig(horizon_ticks=3)
    cases = locked_cases()['train'][:2]
    result = probe_cases(cases, config, ticks=3)
    repeated = probe_cases(cases, config, ticks=3)

    assert result == repeated
    assert [row['case_id'] for row in result['cases']] == ['train-0', 'train-1']
    assert all(row['ticks'] == 3 for row in result['cases'])
    assert {result[name]['ticks'] for name in ('original', 'wider', 'candidate')} == {6}
    assert len({result[name]['unique_raw_pairs']
                for name in ('original', 'wider', 'candidate')}) == 1
    assert result['topology_unchanged'] is True
    with pytest.raises(ValueError, match='locked train or validation'):
        probe_cases(locked_cases()['final'][:1], config, ticks=1)


def test_gate_requires_alias_floor_and_weak_activity_on_both_splits():
    good = {
        split: {
            'original': {'aliased_ticks': 8, 'turn_mse_floor': 0.2,
                         'weak_spiking_readings': 2},
            'wider': {'aliased_ticks': 6, 'turn_mse_floor': 0.1,
                      'weak_spiking_readings': 1},
            'candidate': {'aliased_ticks': 5, 'turn_mse_floor': 0.09,
                          'weak_spiking_readings': 3},
        } for split in ('train', 'validation')
    }
    assert decide_gate(good)['passed'] is True
    for metric, bad_value, flag in (
        ('aliased_ticks', 7, 'alias_improved'),
        ('turn_mse_floor', 0.2, 'turn_floor_not_worse'),
        ('weak_spiking_readings', 2, 'weak_activity_improved'),
    ):
        failed = deepcopy(good)
        failed['validation']['candidate'][metric] = bad_value
        assert decide_gate(failed)['passed'] is False
        assert decide_gate(failed)['validation'][flag] is False


def test_short_diagnostic_records_only_train_and_validation_without_learning(tmp_path):
    config = EvaluationConfig(horizon_ticks=3)
    result_path = run_diagnostic(tmp_path, config=config, train_ticks=2,
                                 validation_ticks=3)
    result = json.loads(result_path.read_text(encoding='utf-8'))
    with (tmp_path / 'cases.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))

    assert result['experiment'] == 'fb07e_weak_signal_probe'
    assert result['probe']['train']['candidate']['ticks'] == 16
    assert result['probe']['validation']['candidate']['ticks'] == 12
    assert len(rows) == 36
    assert {row['encoder'] for row in rows} == {'original', 'wider', 'candidate'}
    assert {row['split'] for row in rows} == {'train', 'validation'}
    assert result['final_cases_evaluated'] is False
    assert result['training_run'] is False
    assert 'final' not in result['cases']
    assert result['controlled_weak_probe']['candidate']['sensory_spike_totals'] == (
        [[0, 10], [10, 0]]
    )
