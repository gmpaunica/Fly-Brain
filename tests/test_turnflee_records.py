"""Lock ownership and saved-record scoring, without scientific reruns."""
import importlib
import json
import numpy as np
import pytest
from flybrain.flight3d.anatomy import load_manifest
from flybrain.turnflee.neural import ResponseConfig


def api():
    assert importlib.util.find_spec('flybrain.turnflee.experiment'), 'one-shot recorded experiment required'
    return importlib.import_module('flybrain.turnflee.experiment')


def test_lock_refuses_renamed_retry_before_any_presentation(tmp_path):
    m=api(); directory=tmp_path/'run'
    m.lock_run(directory)
    config=json.loads((directory/'config.json').read_text())
    assert config['history_audit']['calibration_historical']
    assert config['max_cases']==192 and config['max_neural_steps']==1008000
    assert (directory/'sources/src/flybrain/turnflee/neural.py').exists()
    with pytest.raises(FileExistsError): m.lock_run(tmp_path/'renamed')
    assert not (tmp_path/'renamed').exists()


def test_presentation_scoring_rejects_wrong_or_late_neural_response():
    m=api()
    good=dict(azimuth=90,first_strong_seconds=.62,trigger_seconds=.9,side='left',
              baseline_equal=True,finite=True,max_downstream_hz=240)
    assert m.presentation_success(good)
    for changes in (dict(side='right'),dict(trigger_seconds=1.14),dict(baseline_equal=False),
                    dict(max_downstream_hz=245),dict(finite=False),dict(first_strong_seconds=None),dict(trigger_seconds=None)):
        assert not m.presentation_success(dict(good,**changes))


def test_calibration_eligibility_does_not_add_validation_latency_gate():
    m=api()
    row=dict(phase='calibration',azimuth=90,first_strong_seconds=.6,trigger_seconds=1.2,
             side='left',baseline_equal=True,finite=True,max_downstream_hz=240)
    assert m.presentation_success(row)
    assert not m.presentation_success(dict(row,phase='validation'))


def test_closed_success_requires_weak_recovery_and_no_collision():
    m=api()
    row=dict(first_strong_seconds=.1,trigger_seconds=.2,side='left',azimuth=90,
             collision=False,weak_reached=True,recovered=True,stable_recovery=True,
             finite=True,max_downstream_hz=200,baseline_equal=True)
    assert m.closed_success(row)
    for field in ('weak_reached','recovered','stable_recovery','finite'):
        assert not m.closed_success(dict(row,**{field:False}))
    assert not m.closed_success(dict(row,collision=True))
