"""Recorded experiments must survive accidental repeated commands."""

import importlib

import pytest

from flybrain.neural.readout import TrainableMotorReadout
from flybrain.training.evaluation import EvaluationConfig, locked_cases, write_run
from flybrain.training.fb07h_training import save_matched_checkpoint
from flybrain.training.readout_experiment import save_checkpoint


def test_baseline_writer_refuses_to_replace_existing_result_pair(tmp_path):
    config = EvaluationConfig(horizon_ticks=1)
    paths = write_run(tmp_path, config, locked_cases(), ())
    original = {path: path.read_bytes() for path in paths}
    with pytest.raises(FileExistsError, match='unused|empty'):
        write_run(tmp_path, config, locked_cases(), ())
    assert {path: path.read_bytes() for path in paths} == original


@pytest.mark.parametrize('module,function,extra', [
    ('readout_experiment', 'run_experiment', 1),
    ('fb07d_diagnosis', 'run_diagnosis', 0),
    ('fb07e_probe', 'run_diagnostic', 0),
    ('fb07f_collision', 'run_diagnostic', 0),
    ('fb07g_voltage', 'run_diagnostic', 0),
    ('fb07h_training', 'run_experiment', 1),
    ('fb07i_diagnosis', 'run_diagnostic', 1),
    ('fb07j_gain', 'run_intervention', 2),
])
def test_runner_rejects_existing_records_before_loading_or_computing(
    tmp_path, module, function, extra,
):
    sentinel = tmp_path / 'record.txt'
    sentinel.write_bytes(b'preserve this run')
    runner = getattr(importlib.import_module(f'flybrain.training.{module}'), function)
    missing_inputs = [tmp_path / 'missing.json'] * extra
    # An occupied destination must be refused before input/config processing.
    with pytest.raises(FileExistsError, match='unused|empty'):
        runner(tmp_path, *missing_inputs, config=object())
    assert sentinel.read_bytes() == b'preserve this run'
    assert list(tmp_path.iterdir()) == [sentinel]


@pytest.mark.parametrize('matched', [False, True])
def test_checkpoint_save_never_truncates_existing_file(tmp_path, matched):
    path = tmp_path / 'weights.pt'
    path.write_bytes(b'original checkpoint')
    model = TrainableMotorReadout(9 if matched else 7)
    with pytest.raises(FileExistsError):
        if matched:
            save_matched_checkpoint(path, model, network_seed=7)
        else:
            save_checkpoint(path, model, network_seed=7, alpha=0.5)
    assert path.read_bytes() == b'original checkpoint'
