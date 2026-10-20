"""One capped extension of the original FB-10 precision fit."""

import importlib
import json

import pytest
import torch


def runner():
    return importlib.import_module('flybrain.training.banc_extra_training')


def test_fit_budget_is_one_finite_750_update_candidate():
    module = runner()
    from flybrain.neural.banc_projection import ProjectionReadout
    from flybrain.training.banc_training import fit_readout

    model = ProjectionReadout(torch.zeros(20), torch.ones(20), seed=1010)
    x = torch.zeros((4, 20), dtype=torch.float64)
    y = torch.tensor([[.7, -.4]] * 4, dtype=torch.float64)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fit = module.fit(model, x, y, updates=3)
    original = ProjectionReadout(torch.zeros(20), torch.ones(20), seed=1010)
    old_fit = fit_readout(original, x, y, updates=3)
    assert fit == old_fit
    assert all(torch.equal(value, original.state_dict()[key])
               for key, value in model.state_dict().items())
    assert fit['optimizer_steps'] == 3
    assert len(fit['curve']) == 4
    assert fit['failure'] is None
    assert fit['gradient_nonzero_scalars'] > 0
    assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError):
        module.fit(model, x, y, updates=751)
    with pytest.raises(ValueError):
        module.fit(model, x, y.clone().fill_(float('nan')), updates=1)
    with torch.no_grad():
        model.hidden.weight[0, 0] = float('nan')
    failed = module.fit(model, x, y, updates=1)
    assert failed['failure'] == 'nonfinite readout parameters'
    assert failed['optimizer_steps'] == 0
    assert failed['curve'] == []


def test_preparation_declares_gate_budget_and_preserves_history(tmp_path):
    module = runner()
    original = (module.PROJECT_ROOT / 'experiments/fb10-projection-readout/results.json').read_bytes()
    path = module.prepare(tmp_path / 'new-fit')
    p, digest = module.load_protocol(path)
    assert p['optimizer_step_ceiling'] == 750
    assert p['learning_rate'] == .01
    assert p['seed'] == 1010
    assert p['train_only'] is True
    assert p['gates']['aggregate_absolute_turn_mse_max'] == pytest.approx(.02225273146561865)
    assert p['budget'] == {'optimizer_updates': 750, 'episodes': 25,
                           'world_ticks': 15000, 'neural_steps': 300000}
    assert p['candidate_sha256']['trained.pt'] == module.OLD_CANDIDATE['trained.pt']
    assert digest == path.with_suffix('.sha256').read_text().strip()
    assert original == (module.PROJECT_ROOT / 'experiments/fb10-projection-readout/results.json').read_bytes()
    assert not (path.parent / 'execution.json').exists()
    with pytest.raises(FileExistsError):
        module.prepare(path.parent)


def test_changed_protocol_or_preserved_source_rejected_before_training(tmp_path):
    module = runner()
    path = module.prepare(tmp_path / 'new-fit')
    p = json.loads(path.read_text())
    p['learning_rate'] = .02
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match='protocol'):
        module.load_protocol(path)
    assert not (path.parent / 'execution.json').exists()


def test_consumed_attempt_refuses_retry_without_loading(tmp_path):
    module = runner()
    out = tmp_path / 'old'
    out.mkdir()
    marker = out / 'execution.json'
    marker.write_text('preserve')
    with pytest.raises(FileExistsError, match='consumed'):
        module.run(out / 'missing-protocol.json')
    assert marker.read_text() == 'preserve'


def test_failed_training_gate_stops_new_fresh_evaluation():
    module = runner()
    assert module.may_evaluate_fresh({'training_gate_passed': False}) is False
    assert module.may_evaluate_fresh({'training_gate_passed': True}) is True
    with pytest.raises(ValueError):
        module.may_evaluate_fresh({'training_gate_passed': 1})
