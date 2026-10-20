"""FB-07F: localize exact readout-feature collisions without training."""

import csv
import json

import pytest

from flybrain.training.evaluation import EvaluationConfig, locked_cases
from flybrain.training.fb07f_collision import (
    StageTrace, pair_attribution, run_diagnostic, stage_metrics, trace_cases,
)


def trace(raw, turn, current, sensory, full, groups, ema):
    return StageTrace(raw, turn, current, sensory, full, groups, ema)


def test_stage_metrics_count_distinct_inputs_and_turn_floor():
    traces = [
        trace((0.01, 0.02), 1.0, (1.0, 1.0), (0, 0), (0, 0),
              (0.0, 0.0), (0.0, 0.0)),
        trace((0.02, 0.01), -1.0, (1.0, 1.0), (0, 0), (0, 0),
              (0.0, 0.0), (0.0, 0.0)),
        trace((0.03, 0.03), 0.0, (2.0, 2.0), (1, 1), (1, 1),
              (1.0, 1.0), (1.0, 1.0)),
    ]

    metrics = stage_metrics(traces)

    assert metrics['current']['unique_vectors'] == 2
    assert metrics['current']['aliased_ticks'] == 2
    assert metrics['current']['turn_mse_floor'] == pytest.approx(2 / 3)
    assert metrics['ema']['unique_vectors'] == 2
    assert metrics['ema']['aliased_ticks'] == 2


def test_final_conflict_pairs_report_latest_distinct_stage():
    pairs = [
        # Input values collapse even in encoded current.
        ((1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0),
         (0.0, 0.0), (0.0, 0.0)),
        # Encoded currents differ, spike counts do not.
        ((1, 2), (2, 1), (0, 0), (0, 0), (0, 0), (0, 0),
         (0.0, 0.0), (0.0, 0.0)),
        # Sensory neuron patterns differ, seven means coincide.
        ((1, 2), (2, 1), (1, 0), (0, 1), (1, 0), (0, 1),
         (0.5, 0.0), (0.5, 0.0)),
        # Only downstream neuron patterns differ before group means.
        ((1, 2), (2, 1), (0, 0), (0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
         (0.0, 0.5), (0.0, 0.5)),
        # Distinct group means reach identical EMA features.
        ((1, 2), (2, 1), (0, 0), (0, 0), (0, 0), (0, 0),
         (0.0, 0.5), (0.5, 0.0)),
    ]
    traces = []
    for marker, (ca, cb, sa, sb, fa, fb, ga, gb) in enumerate(pairs):
        traces.extend((
            trace((0.01 + marker, 0.02), 1.0, ca, sa, fa, ga,
                  (float(marker),)),
            trace((0.02 + marker, 0.01), -1.0, cb, sb, fb, gb,
                  (float(marker),)),
        ))

    result = pair_attribution(traces)

    assert result == {
        'conflicting_pairs': 5,
        'raw_to_current': 1,
        'current_to_spikes': 1,
        'sensory_spikes_to_group_means': 1,
        'other_spikes_to_group_means': 1,
        'group_means_to_ema': 1,
    }


def test_short_replay_resets_cases_and_keeps_final_sealed():
    config = EvaluationConfig(horizon_ticks=3)
    cases = locked_cases()['train'][:2]
    probe = trace_cases(cases, config, ticks=3)

    assert probe == trace_cases(cases, config, ticks=3)
    assert [row['case_id'] for row in probe['cases']] == ['train-0', 'train-1']
    assert {probe[name]['stages']['ema']['ticks']
            for name in ('original', 'wider', 'candidate')} == {6}
    assert probe['topology_unchanged'] is True
    with pytest.raises(ValueError, match='locked train or validation'):
        trace_cases(locked_cases()['final'][:1], config, ticks=1)


def test_short_artifact_records_stage_rows_without_learning_or_final(tmp_path):
    config = EvaluationConfig(horizon_ticks=2)
    path = run_diagnostic(tmp_path, config=config, train_ticks=2,
                          validation_ticks=2)
    result = json.loads(path.read_text(encoding='utf-8'))
    with (tmp_path / 'cases.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))

    assert result['experiment'] == 'fb07f_collision_stages'
    assert len(rows) == 36
    assert result['probe']['train']['candidate']['stages']['ema']['ticks'] == 16
    assert result['probe']['validation']['candidate']['stages']['ema']['ticks'] == 8
    assert result['topology_unchanged'] is True
    assert result['training_run'] is False
    assert result['final_cases_evaluated'] is False
    assert 'final' not in result['cases']
