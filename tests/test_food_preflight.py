"""Recorded full-circle starts and embodied teacher scoring."""

import math

from flybrain.training.food_preflight import generate_cases, run_teacher_case


def test_case_splits_are_disjoint_balanced_and_inside_world():
    cases = generate_cases()
    assert {split: len(group) for split, group in cases.items()} == {
        'train': 40, 'validation': 16, 'final': 64}
    assert len({case.seed for group in cases.values() for case in group}) == 120
    for split, group in cases.items():
        sectors = [int(case.heading / (math.tau / 8)) for case in group]
        expected = 5 if split == 'train' else (2 if split == 'validation' else 8)
        assert all(sectors.count(sector) == expected for sector in range(8))
        for case in group:
            assert 60 <= math.hypot(case.x - 600, case.y - 300) <= 220.000001
            assert 6 <= case.x <= 794 and 6 <= case.y <= 594


def test_teacher_embodied_toward_and_away_cases_stop_at_food():
    cases = generate_cases()['train']
    for index in (32, 36):
        result = run_teacher_case(cases[index])
        assert result['first_contact_seconds'] is not None
        assert result['first_contact_seconds'] <= 17
        assert result['approach_speed'] <= 30
        assert result['stationary_dwell_seconds'] >= 2
        assert result['departures'] == 0
