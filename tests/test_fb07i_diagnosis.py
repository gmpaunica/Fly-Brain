"""FB-07I scoring-only tick traces for failed embodied navigation."""

import csv
import json
import math

import pytest

from flybrain.neural.readout import TrainableMotorReadout
from flybrain.training.evaluation import (
    EvaluationConfig, evaluate_cases, fixed_controller, locked_cases,
    run_episode, write_run,
)
from flybrain.training.fb07h_training import run_experiment
from flybrain.training.fb07i_diagnosis import (
    bearing_error, run_diagnostic, summarize_ticks, trace_cases, trace_episode,
)


def test_bearing_error_uses_clockwise_screen_coordinates():
    assert bearing_error(0, 0, 0, 0, 1) == pytest.approx(math.pi / 2)
    assert bearing_error(0, 0, 0, 0, -1) == pytest.approx(-math.pi / 2)
    assert bearing_error(0, 0, 3 * math.pi / 2, 1, 0) == pytest.approx(math.pi / 2)


def test_tick_summary_counts_alignment_and_wall_segments():
    ticks = [
        {'bearing_error': 0.5, 'teacher_turn': 1.0, 'command_turn': -1.0,
         'distance_change': 2.0, 'collision': False, 'command_forward': 0.7},
        {'bearing_error': -0.5, 'teacher_turn': -1.0, 'command_turn': -1.0,
         'distance_change': -1.0, 'collision': True, 'command_forward': 0.7},
        {'bearing_error': 0.05, 'teacher_turn': 0.0, 'command_turn': 0.0,
         'distance_change': -2.0, 'collision': True, 'command_forward': 0.7},
    ]

    result = summarize_ticks(ticks)

    assert result['teacher_alignment'] == {'matching': 2, 'eligible': 2}
    assert result['command_alignment'] == {'matching': 1, 'eligible': 2}
    assert result['first_wall_tick'] == 2
    assert result['wall_ticks'] == 2
    assert result['distance_change_before_wall'] == 2.0
    assert result['distance_change_from_wall'] == -3.0
    assert result['approach_ticks'] == 1
    assert result['retreat_ticks'] == 2


def test_short_trace_replays_evaluator_and_keeps_controller_sensory_only():
    case = locked_cases()['validation'][0]
    config = EvaluationConfig(horizon_ticks=3)
    rows, endpoint = trace_episode(case, config,
                                   lambda: fixed_controller(config.network_seed),
                                   ticks=3)
    scored = run_episode(case, config)

    assert len(rows) == scored.ticks
    assert endpoint['final_x'] == scored.final_x
    assert endpoint['final_y'] == scored.final_y
    assert endpoint['outcome'] == scored.outcome
    assert rows[0]['tick'] == 1
    assert rows[0]['distance_change'] == pytest.approx(
        rows[0]['pre_distance'] - rows[0]['post_distance'])
    assert set(rows[0]['observation']) == {'odor_left', 'odor_right'}


def test_short_multi_case_trace_resets_and_excludes_final():
    config = EvaluationConfig(horizon_ticks=2)
    models = {'seven': TrainableMotorReadout(7),
              'nine': TrainableMotorReadout(9)}
    cases = locked_cases()['train'][:2]
    first = trace_cases(cases, config, ticks=2, models=models)

    assert first == trace_cases(cases, config, ticks=2, models=models)
    assert len(first['cases']) == 6
    assert len(first['ticks']) == 12
    assert {row['controller'] for row in first['cases']} == {'fixed', 'seven', 'nine'}
    with pytest.raises(ValueError, match='locked train or validation'):
        trace_cases(locked_cases()['final'][:1], config, ticks=1, models=models)


def test_short_artifact_has_only_train_validation_and_matches_fb07h(tmp_path):
    config = EvaluationConfig(horizon_ticks=2)
    cases = locked_cases()
    baseline, _ = write_run(tmp_path / 'baseline', config, cases,
                            evaluate_cases(cases['validation'] + cases['final'], config))
    trained = run_experiment(tmp_path / 'trained', baseline, config=config,
                             sample_ticks=2, steps=2)
    path = run_diagnostic(tmp_path / 'diagnosis', trained,
                          config=config, train_ticks=2, validation_ticks=2)
    result = json.loads(path.read_text(encoding='utf-8'))
    with (tmp_path / 'diagnosis' / 'ticks.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))

    assert result['experiment'] == 'fb07i_embodied_failure_diagnosis'
    assert len(rows) == 72
    assert len(result['cases']['train']) == 24
    assert len(result['cases']['validation']) == 12
    assert result['validation_replay_exact'] is True
    assert result['training_run'] is False
    assert result['final_cases_evaluated'] is False
    assert 'final' not in result['cases']
