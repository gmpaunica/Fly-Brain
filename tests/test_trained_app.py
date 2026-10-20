"""Frozen app integration must preserve measured behavior and historical bytes."""

import importlib
import shutil

import pygame
import pytest
import torch


def loader():
    return importlib.import_module('flybrain.neural.frozen_banc')


def test_saved_head_is_frozen_and_load_does_not_step_or_consume_rng():
    module = loader()
    before = torch.random.get_rng_state().clone()
    controller = module.load_frozen_controller()
    assert torch.equal(before, torch.random.get_rng_state())
    assert controller.network.snapshot().step_count == 0
    model = controller.decoder.model
    assert not model.training
    assert sum(p.numel() for p in model.parameters()) == 370
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert torch.count_nonzero(controller.decoder.feature_rates) == 0


@pytest.mark.parametrize('relative', [
    'experiments/fb10-projection-readout/trained.pt',
    'experiments/fb10-projection-readout/protocol.json',
    'experiments/fb10-projection-readout/results.json',
    'experiments/fb10-projection-evaluation/results.json',
    'experiments/fb10-projection-evaluation/protocol.json',
    'src/flybrain/neural/banc_projection.py',
])
def test_changed_checkpoint_evidence_or_source_is_rejected(tmp_path, relative):
    module = loader()
    shutil.copytree(module.PROJECT_ROOT / 'src', tmp_path / 'src',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in module.PINNED_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(module.PROJECT_ROOT / name, target)
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs|missing') as exc:
        module.load_frozen_controller(project_root=tmp_path)
    assert str(target) in str(exc.value)


def test_actual_head_reset_timing_motor_gate_and_render_independence():
    app = importlib.import_module('flybrain.trained_app')
    loop = app.build_simulation(7)
    reference = app.build_simulation(7)
    model = loop.controller.decoder.model
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    def signature(frame):
        return (frame.body, frame.command, frame.pre_movement_observation,
                frame.activity.spike_counts.tolist(), frame.activity.end_step)
    pygame.font.init()
    try:
        surface = pygame.Surface((1200, 600))
        world, brain = app.build_views(surface, loop)
        first = []
        for tick in range(30):
            frame = loop.step()
            first.append(signature(frame))
            assert frame.activity.end_step == (tick + 1) * 20
            brain.observe(frame)
            rates = loop.controller.decoder.feature_rates.clone()
            for _ in range(2):
                world.draw(loop.world)
                brain.draw()
            assert torch.equal(rates, loop.controller.decoder.feature_rates)
            assert signature(reference.step()) == first[-1]
        loop.advance_snapshots(.009)
        loop.reset(7)
        brain.reset()
        assert loop.advance_snapshots(.011) == ()
        assert not any(brain.model.activity_levels)
        assert torch.count_nonzero(loop.controller.decoder.feature_rates) == 0
        assert [signature(loop.step()) for _ in range(30)] == first
        loop.reset(7)
        loop.controller.set_motor_enabled(False)
        stopped = [loop.step() for _ in range(10)]
        assert stopped[0].body == stopped[-1].body
        assert all(f.command.forward == f.command.turn == 0 for f in stopped)
        assert stopped[-1].activity.end_step == 200
        assert any(f.activity.spike_counts.sum() > 0 for f in stopped)
        assert all(torch.equal(v, weights[k]) for k, v in model.state_dict().items())
    finally:
        pygame.quit()


@pytest.mark.parametrize('mode', ['banc-trained', 'banc', 'manual'])
def test_cli_exact_tick_smoke(mode, capsys):
    app = importlib.import_module('flybrain.trained_app')
    assert app.main(['--control', mode, '--seed', '7', '--headless', '--steps', '3']) == 0
    assert f'mode={mode} ticks=3' in capsys.readouterr().out
    assert not pygame.get_init()


def test_cli_missing_model_fails_before_pygame(monkeypatch, capsys):
    app = importlib.import_module('flybrain.trained_app')
    def missing(**kwargs):
        raise FileNotFoundError('missing trained checkpoint')
    monkeypatch.setattr(loader(), 'load_frozen_controller', missing)
    with pytest.raises(SystemExit) as exc:
        app.main(['--headless', '--steps', '1'])
    assert exc.value.code == 2
    assert 'Frozen BANC controller is unavailable' in capsys.readouterr().err
    assert not pygame.get_init()


def test_cli_reset_and_escape_events_clear_world_brain_and_neural_time(monkeypatch):
    app = importlib.import_module('flybrain.trained_app')
    loop = app.build_simulation(7)
    monkeypatch.setattr(app, 'build_simulation', lambda seed: loop)
    events = iter(([], [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)],
                   [], [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)]))
    monkeypatch.setattr(pygame.event, 'get', lambda: next(events))
    frames = []
    original = app.build_views
    def observed_views(surface, simulation):
        world, brain = original(surface, simulation)
        draw = brain.draw
        def observed_draw():
            frames.append((loop.world.tick_count, brain.model.tick_id,
                           loop.controller.network.snapshot().step_count))
            draw()
        monkeypatch.setattr(brain, 'draw', observed_draw)
        return world, brain
    monkeypatch.setattr(app, 'build_views', observed_views)
    assert app.main(['--headless', '--steps', '10', '--seed', '7']) == 0
    assert frames == [(1, 1, 20), (0, 0, 0), (1, 1, 20)]
    assert not pygame.get_init()
