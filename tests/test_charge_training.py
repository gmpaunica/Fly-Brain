"""Train-only collection and distinct charge-checkpoint semantics."""

import csv
import json

import pytest
import torch

from flybrain.neural.readout import TrainableMotorReadout
from flybrain.training.evaluation import EvaluationConfig, locked_cases, run_episode
from flybrain.training.fb07h_training import collect_paired_samples, save_matched_checkpoint
from flybrain.training.charge_repair import (
    charge_controller_factory, collect_charge_samples, load_charge_checkpoint,
    run_repair, save_charge_checkpoint,
)


def test_charge_collection_preserves_teacher_and_spike_rows_and_rejects_nontrain():
    cases = locked_cases()['train'][:2]
    config = EvaluationConfig(horizon_ticks=3)
    previous = collect_paired_samples(cases, config, sample_ticks=3)[1]
    charge = collect_charge_samples(cases, config, sample_ticks=3)
    assert charge.features.shape == (6, 9)
    assert torch.equal(charge.features[:, :7], previous.features[:, :7])
    assert torch.equal(charge.targets, previous.targets)
    assert charge.case_counts == previous.case_counts
    assert torch.isfinite(charge.features).all()
    with pytest.raises(ValueError, match='train'):
        collect_charge_samples(locked_cases()['validation'][:1], config, sample_ticks=1)


def test_charge_checkpoint_replays_and_rejects_old_nine_feature_meaning(tmp_path):
    model = TrainableMotorReadout(9)
    with torch.no_grad():
        model.linear.weight[1, 7:] = torch.tensor([-4.0, 4.0])
    path = tmp_path / 'charge.pt'
    save_charge_checkpoint(path, model, network_seed=7)
    loaded = load_charge_checkpoint(path, network_seed=7)
    case = locked_cases()['train'][0]
    config = EvaluationConfig(horizon_ticks=4)
    assert run_episode(case, config, charge_controller_factory(model, 7)) == (
        run_episode(case, config, charge_controller_factory(loaded, 7)))
    old = tmp_path / 'voltage.pt'
    save_matched_checkpoint(old, model, network_seed=7)
    with pytest.raises(ValueError, match='contract'):
        load_charge_checkpoint(old, network_seed=7)
    with pytest.raises(ValueError, match='contract'):
        load_charge_checkpoint(path, network_seed=8)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_charge_checkpoint(path, model, network_seed=7)
    assert path.read_bytes() == original


def test_short_repair_records_matched_train_controls_and_preserves_run(tmp_path):
    output = tmp_path / 'repair'
    path = run_repair(output, config=EvaluationConfig(horizon_ticks=2),
                      sample_ticks=2, steps=2)
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['optimizer_steps'] == 2
    assert result['sample_count'] == 16
    assert set(result['summaries']) == {'trained', 'untrained', 'fixed'}
    assert all(row['episodes'] == 8 for row in result['summaries'].values())
    assert all(case['split'] == 'train' for case in result['cases'])
    assert result['checks']['topology_unchanged']
    assert result['checks']['checkpoint_replay_exact']
    assert result['validation_cases_evaluated'] is False
    assert result['final_cases_evaluated'] is False
    assert result['gate']['passed'] is False
    with (output / 'curve.csv').open(newline='', encoding='utf-8') as file:
        assert len(list(csv.DictReader(file))) == 3
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        run_repair(output)
    assert path.read_bytes() == before
