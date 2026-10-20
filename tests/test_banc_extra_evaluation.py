"""Fresh evaluation of the one separately passing extra-training checkpoint."""

import importlib
import json

import pytest
import torch


def runner():
    return importlib.import_module('flybrain.training.banc_extra_evaluation')


def test_passing_training_evidence_is_recomputed_without_steps():
    module = runner()
    p, digest, models, result = module.training_evidence()
    assert result['training_gate_passed'] is True
    assert result['gates']['aggregate_absolute_turn_fit'] is True
    assert result['per_output_mse'][1] <= p['gates']['aggregate_absolute_turn_mse_max']
    assert result['historical_500_update_prefix_exact'] is True
    assert result['optimizer_steps'] == 750
    assert all(not parameter.requires_grad for model in models.values()
               for parameter in model.parameters())
    assert all(not torch.any(parameter.grad) for model in models.values()
               for parameter in model.parameters() if parameter.grad is not None)
    assert len(digest) == 64


def isolated_preparation(monkeypatch, tmp_path, module):
    """Exercise pre-assay preparation without reopening the real consumed receipt."""
    from flybrain.training.banc_frozen_evaluation import historical_seeds
    monkeypatch.setattr(module, 'CENTRAL_RECEIPT', tmp_path / 'unused-receipt.json')
    monkeypatch.setattr(module, 'historical_seeds',
                        lambda *, exclude=None: historical_seeds(exclude=module.DEFAULT_OUTPUT))


def test_fresh_cases_are_disjoint_and_preparation_takes_zero_steps(tmp_path, monkeypatch):
    module = runner()
    cases = module.fresh_cases()
    assert [len(cases[name]) for name in ('validation', 'final')] == [16, 32]
    assert {case.seed for group in cases.values() for case in group} <= module.historical_seeds()
    isolated_preparation(monkeypatch, tmp_path, module)
    path = module.prepare(tmp_path / 'assay')
    p, digest = module.load_protocol(path)
    assert p['budget']['optimizer_steps'] == 0
    assert p['budget']['maximum_episodes'] == 144
    assert p['gates']['validation_arrivals_min'] == 13
    assert p['gates']['final_arrivals_min'] == 26
    assert p['training_gate_passed'] is True
    assert digest == path.with_suffix('.sha256').read_text().strip()
    assert not (path.parent / 'execution.json').exists()


def test_changed_protocol_and_consumed_attempt_fail_before_episodes(tmp_path, monkeypatch):
    module = runner()
    isolated_preparation(monkeypatch, tmp_path, module)
    path = module.prepare(tmp_path / 'assay')
    definition = json.loads(path.read_text())
    definition['gates']['final_arrivals_min'] = 1
    path.write_text(json.dumps(definition))
    with pytest.raises(ValueError, match='hash'):
        module.load_protocol(path)
    assert not (path.parent / 'execution.json').exists()
    receipt = tmp_path / 'receipt.json'
    receipt.write_text('consumed')
    monkeypatch.setattr(module, 'CENTRAL_RECEIPT', receipt)
    with pytest.raises(FileExistsError, match='consumed'):
        module.run(path)


def test_training_failure_blocks_protocol_even_if_record_is_resealed(monkeypatch, tmp_path):
    module = runner()
    isolated_preparation(monkeypatch, tmp_path, module)
    original = module.training_evidence
    def failed():
        p, digest, models, result = original()
        return p, digest, models, {**result, 'training_gate_passed': False}
    monkeypatch.setattr(module, 'training_evidence', failed)
    with pytest.raises(ValueError, match='passing training gate'):
        module.prepare(tmp_path / 'assay')
    assert not (tmp_path / 'assay').exists()
