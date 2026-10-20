"""Frozen evaluation integrity, paired statistics and final-set gating."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch

from flybrain.neural.readout import TrainableMotorReadout
from flybrain.training.charge_repair import save_charge_checkpoint
from flybrain.training.evaluation import EvaluationConfig
from flybrain.training.charge_evaluation import (
    assess_split, paired_arrival_test, run_evaluation, verify_protocol,
)


def test_paired_probability_uses_discordant_cases_and_handles_ties():
    result = paired_arrival_test([True] * 10, [False] * 10)
    assert result['trained_only'] == 10
    assert result['control_only'] == 0
    assert result['one_sided_p'] == pytest.approx(1 / 1024)
    mixed = paired_arrival_test([True] * 8 + [False] * 2,
                                [False] * 8 + [True] * 2)
    assert mixed['one_sided_p'] == pytest.approx(56 / 1024)
    assert paired_arrival_test([True, False], [True, False])['one_sided_p'] == 1
    with pytest.raises(ValueError):
        paired_arrival_test([True], [False, True])


def test_validation_and_final_use_the_declared_arrival_and_paired_gates():
    validation = {'trained': [True] * 13 + [False] * 3,
                  'untrained': [False] * 16, 'fixed': [False] * 16}
    assert assess_split('validation', validation)['passed']
    assert not assess_split('validation', {**validation,
        'trained': [True] * 12 + [False] * 4})['passed']
    assert not assess_split('validation', {**validation,
        'fixed': validation['trained']})['passed']
    final = {'trained': [True] * 26 + [False] * 6,
             'untrained': [False] * 32, 'fixed': [False] * 32}
    assert assess_split('final', final)['passed']
    # 26 vs 18, but 14 wins / 6 losses yields p > .025: margin alone fails.
    assert not assess_split('final', {**final,
        'fixed': [True] * 12 + [False] * 14 + [True] * 6})['passed']
    with pytest.raises(ValueError):
        assess_split('final', validation)


def _toy_protocol(tmp_path, *, near_food=False):
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    saved = {}
    for name in ('trained', 'untrained'):
        path = checkpoints / f'{name}.pt'
        model = TrainableMotorReadout(9)
        if near_food and name == 'trained':
            with torch.no_grad():
                model.linear.bias[0] = 4.0
        save_charge_checkpoint(path, model, network_seed=7)
        saved[name] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    source = tmp_path / 'source.txt'
    source.write_text('test fixture', encoding='utf-8')
    cases = {}
    for split, count, start in [('validation', 16, 900000), ('final', 32, 910000)]:
        cases[split] = [dict(case_id=f'{split}-{i}', split=split, seed=start+i,
                             x=578.5 if near_food else 400.0,
                             y=300.0, heading=0.0) for i in range(count)]
    payload = {
        'protocol': 'm6_charge_fresh_matched_evaluation', 'network_seed': 7,
        'config': asdict(EvaluationConfig(horizon_ticks=1)), 'cases': cases,
        'candidate_checkpoints': saved, 'checkpoint_directory': 'checkpoints',
        'frozen_source_sha256': {'source.txt': hashlib.sha256(source.read_bytes()).hexdigest()},
        'budget': {'training_updates': 0, 'validation_episodes': 48,
                   'final_episodes_if_validation_passes': 96, 'maximum_world_ticks': 86400,
                   'horizon_ticks_per_episode': 1},
    }
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_integrity_checks_reject_changed_source_checkpoint_and_protocol(tmp_path):
    protocol, digest = _toy_protocol(tmp_path)
    verify_protocol(protocol, expected_sha256=digest, project_root=tmp_path)
    source = tmp_path / 'source.txt'
    source.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='source'):
        verify_protocol(protocol, expected_sha256=digest, project_root=tmp_path)
    source.write_text('test fixture', encoding='utf-8')
    (tmp_path / 'checkpoints/trained.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        verify_protocol(protocol, expected_sha256=digest, project_root=tmp_path)
    with pytest.raises(ValueError, match='protocol'):
        verify_protocol(protocol, expected_sha256='wrong', project_root=tmp_path)


def test_failed_validation_never_runs_final_and_protocol_cannot_be_reused(tmp_path):
    protocol, digest = _toy_protocol(tmp_path)
    output = tmp_path / 'evaluation'
    path = run_evaluation(protocol, output, expected_sha256=digest, project_root=tmp_path)
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['optimizer_steps'] == 0
    assert result['gates']['validation']['passed'] is False
    assert result['final_cases_evaluated'] is False
    assert set(result['results']) == {'validation'}
    assert not (output / 'final.csv').exists()
    rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
    assert len(rows) == 48
    assert {row['result']['split'] for row in rows} == {'validation'}
    with pytest.raises(FileExistsError):
        run_evaluation(protocol, tmp_path / 'second-output', expected_sha256=digest,
                       project_root=tmp_path)
    copied = tmp_path / 'copied-protocol.json'
    copied.write_bytes(protocol.read_bytes())
    with pytest.raises(FileExistsError):
        run_evaluation(copied, tmp_path / 'third-output', expected_sha256=digest,
                       project_root=tmp_path)


def test_successful_validation_opens_one_matched_final_run(tmp_path):
    # In one real world tick, speed ~.982 crosses the radius; speed .5 does not.
    protocol, digest = _toy_protocol(tmp_path, near_food=True)
    output = tmp_path / 'evaluation'
    path = run_evaluation(protocol, output, expected_sha256=digest, project_root=tmp_path)
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['gates']['validation']['passed']
    assert result['gates']['final']['successes'] == {'trained': 32, 'untrained': 0, 'fixed': 0}
    assert result['m6_gate_passed']
    assert result['final_cases_evaluated']
    assert result['actual_episodes'] == result['actual_world_ticks'] == 144
    assert result['frozen_parameters_unchanged']
    assert (output / 'final.csv').exists()
