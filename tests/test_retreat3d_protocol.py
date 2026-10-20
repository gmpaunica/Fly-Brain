"""Locked case accounting and fail-closed scientific receipts without rerunning."""
import importlib
import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('flybrain.retreat3d.experiment') is not None
    return importlib.import_module('flybrain.retreat3d.experiment')


def test_locked_cases_and_budget():
    m = api(); calibration = m.cases('calibration'); validation = m.cases('validation')
    assert len(calibration) == 48 and len(validation) == 96
    assert sorted({c['azimuth'] for c in calibration}) == list(range(0, 360, 45))
    assert sorted({c['azimuth'] for c in validation}) == [11.25+22.5*i for i in range(16)]
    assert {c['elevation'] for c in validation} == {-20, 0, 20}
    assert len(m.closed_cases()) == 48
    assert m.MAX_SECONDS == 504 and m.MAX_NEURAL_STEPS == 1008000


def test_validation_gate_global_and_each_azimuth():
    m = api(); rows = [dict(c, success=True) for c in m.cases('validation')]
    assert m.validation_gate(rows)
    for r in rows[:2]: r['success'] = False
    assert not m.validation_gate(rows)
    assert not m.validation_gate(rows[:-1])


def test_case_requires_usable_response_and_no_wrong_confident_windows():
    m = api()
    from flybrain.retreat3d.neural import FrozenDecoder
    d = FrozenDecoder(np.eye(8, 48), tuple(str(i) for i in range(48)))
    assert m.score_windows([np.eye(8, 48)[0]], d, 11.25)['success']
    assert not m.score_windows([np.zeros(48)], d, 11.25)['success']
    assert not m.score_windows([np.eye(8, 48)[0], np.eye(8, 48)[4]], d, 11.25)['success']
