"""Behavioral contracts for renderer-free, fixed-controller evaluation."""

import csv
import json
import math

import pytest

from flybrain.training.evaluation import (
    EpisodeCase,
    EvaluationConfig,
    evaluate_cases,
    locked_cases,
    run_episode,
    summarize,
    write_run,
)


def test_locked_splits_have_distinct_reproducible_initial_conditions():
    first = locked_cases()
    second = locked_cases()

    assert first == second
    assert {split: len(cases) for split, cases in first.items()} == {
        'train': 8, 'validation': 4, 'final': 8,
    }
    assert [case.seed for case in first['train']] == list(range(8))
    assert [case.seed for case in first['validation']] == list(range(100, 104))
    assert [case.seed for case in first['final']] == list(range(200, 208))
    poses = {(case.x, case.y, case.heading) for cases in first.values()
             for case in cases}
    assert len(poses) == 20
    for cases in first.values():
        for case in cases:
            distance = math.hypot(case.x - 600, case.y - 300)
            assert 60 <= distance <= 180
            assert 0 <= case.heading < math.tau
            assert 6 <= case.x <= 794
            assert 6 <= case.y <= 594


def test_start_inside_success_radius_terminates_without_neural_step():
    case = EpisodeCase('initial', 'validation', 100, 600, 300, 0)
    result = run_episode(case, EvaluationConfig(horizon_ticks=3))

    assert result.outcome == 'success'
    assert result.ticks == 0
    assert result.time_to_food_seconds == 0
    assert result.path_length == 0
    assert result.spike_count == 0
    assert result.collision_count == 0


def test_last_allowed_tick_can_succeed_and_uses_post_movement_pose():
    case = EpisodeCase('near', 'validation', 101, 579.7, 300, 0)
    result = run_episode(case, EvaluationConfig(horizon_ticks=2))

    assert result.outcome == 'success'
    assert result.ticks == 2
    assert result.time_to_food_seconds == pytest.approx(0.04)
    assert result.final_distance < 20
    assert result.progress_fraction == 1
    assert result.path_length > 0
    assert result.path_efficiency is not None
    assert 0 < result.path_efficiency <= 1


def test_timeout_counts_wall_contact_and_keeps_failed_time_null():
    case = EpisodeCase('wall', 'final', 200, 794, 300, 0)
    result = run_episode(case, EvaluationConfig(horizon_ticks=2))

    assert result.outcome == 'timeout'
    assert result.ticks == 2
    assert result.elapsed_seconds == pytest.approx(0.04)
    assert result.time_to_food_seconds is None
    assert result.path_efficiency is None
    assert result.collision_count == 1
    assert result.path_length == 0
    assert result.final_distance == pytest.approx(194)
    assert result.progress_fraction == pytest.approx(0)


def test_repeated_fixed_evaluation_matches_and_summary_includes_failures():
    config = EvaluationConfig(horizon_ticks=3)
    cases = locked_cases()['validation'][:2]
    first = evaluate_cases(cases, config)
    second = evaluate_cases(cases, config)

    assert first == second
    report = summarize(first)
    assert report['episodes'] == 2
    assert report['successes'] + report['timeouts'] == 2
    assert report['success_rate'] == report['successes'] / 2
    assert report['mean_progress_fraction'] == pytest.approx(
        sum(result.progress_fraction for result in first) / 2
    )
    if report['successes'] == 0:
        assert report['mean_success_time_seconds'] is None
        assert report['mean_success_path_efficiency'] is None


def test_written_run_contains_locked_cases_metadata_and_per_episode_csv(tmp_path):
    config = EvaluationConfig(horizon_ticks=1)
    cases = locked_cases()
    measured = evaluate_cases(cases['validation'][:1], config)
    json_path, csv_path = write_run(tmp_path, config, cases, measured)

    payload = json.loads(json_path.read_text(encoding='utf-8'))
    with csv_path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))

    assert payload['experiment'] == 'fb06_fixed_baseline'
    assert payload['controller'] == 'fixed_synthetic_100'
    assert payload['training_steps'] == 0
    assert payload['learning_rate'] is None
    assert payload['checkpoint'] is None
    assert payload['software']['device'] == 'cpu'
    assert payload['config']['horizon_ticks'] == 1
    assert len(payload['cases']['train']) == 8
    assert len(payload['cases']['final']) == 8
    assert payload['results'][0]['case_id'] == measured[0].case_id
    assert rows[0]['case_id'] == measured[0].case_id
    assert rows[0]['outcome'] == measured[0].outcome
