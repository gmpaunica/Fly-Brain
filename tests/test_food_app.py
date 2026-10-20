"""Accepted food UI loads the frozen result and keeps old reference modes."""

import importlib
import json
import shutil

import pygame
import pytest
import torch

from flybrain.foraging import ForagingSimulation
from flybrain.neural.food_readout import FoodReadout, make_food_controller
from flybrain.simulation import FixedStepSimulation
from flybrain.world.arena import Arena
from flybrain.world.odor import OdorSource


def test_accepted_loader_freezes_exact_food_checkpoint_without_rng_use():
    module = importlib.import_module('flybrain.neural.accepted_food')
    before = torch.random.get_rng_state().clone()
    controller = module.load_accepted_controller()
    assert torch.equal(before, torch.random.get_rng_state())
    assert controller.network.snapshot().step_count == 0
    assert len(controller.network.topology.root_ids) == 85
    model = controller.decoder.model
    saved = torch.load(module.PROJECT_ROOT / 'experiments/food-stop-fit-2026-09-28/trained.pt',
                       weights_only=True, map_location='cpu')
    assert all(torch.equal(value, saved['state_dict'][key])
               for key, value in model.state_dict().items())
    assert sum(parameter.numel() for parameter in model.parameters()) == 419
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in model.parameters())


@pytest.mark.parametrize('relative', [
    'experiments/food-stop-recovery-2026-09-29/results.json',
    'experiments/food-stop-recovery-2026-09-29/episodes.jsonl',
    'src/flybrain/neural/food_readout.py',
])
def test_accepted_loader_rejects_changed_result_journal_or_source(tmp_path, relative):
    module = importlib.import_module('flybrain.neural.accepted_food')
    source = module.PROJECT_ROOT
    names = set(module.PINNED_FILES) | {
        'experiments/food-stop-recovery-2026-09-29/results.json'}
    for protocol_name in (
        'experiments/food-stop-fit-2026-09-28/protocol.json',
        'experiments/food-stop-fresh-2026-09-28/protocol.json',
        'experiments/food-stop-recovery-2026-09-29/protocol.json',
    ):
        protocol = json.loads((source / protocol_name).read_text(encoding='utf-8'))
        names.update(protocol['source_sha256'])
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    target = tmp_path / relative
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs|missing'):
        module.load_accepted_controller(project_root=tmp_path)


def test_candidate_world_and_brain_panels_read_activity_without_stepping():
    app = importlib.import_module('flybrain.food_app')
    arena = Arena()
    controller = make_food_controller(FoodReadout(torch.zeros(22), torch.ones(22)))
    loop = ForagingSimulation(FixedStepSimulation(
        arena, arena.spawn(7), odor_source=OdorSource(600, 300)), controller)
    pygame.font.init()
    try:
        world, brain = app.build_views(pygame.Surface((1200, 600)), loop)
        for tick in range(5):
            frame = loop.step()
            brain.observe(frame)
            assert frame.activity.end_step == (tick + 1) * 20
            before = controller.network.snapshot()
            world.draw(loop.world)
            brain.draw()
            brain.draw()
            after = controller.network.snapshot()
            assert before.step_count == after.step_count
            assert torch.equal(before.voltage, after.voltage)
        loop.reset(7)
        brain.reset()
        assert brain.model.tick_id == 0
    finally:
        pygame.font.quit()


def test_food_mode_and_historical_reference_both_run(capsys):
    app = importlib.import_module('flybrain.food_app')
    assert app.main(['--seed', '7', '--headless', '--steps', '2']) == 0
    assert 'mode=food-trained ticks=2 checkpoint=approach-stop' in capsys.readouterr().out
    assert not pygame.get_init()
    assert app.main(['--control', 'banc-trained', '--seed', '7',
                     '--headless', '--steps', '2']) == 0
    assert 'mode=banc-trained ticks=2 checkpoint=750-updates' in capsys.readouterr().out
    assert not pygame.get_init()


def test_default_launcher_selects_food_mode_after_live_review(capsys):
    launcher = importlib.import_module('flybrain.__main__')
    assert launcher.main(['--seed', '7', '--headless', '--steps', '2']) == 0
    assert 'mode=food-trained ticks=2 checkpoint=approach-stop' in capsys.readouterr().out


def test_accepted_food_reset_timing_motor_gate_and_render_independence():
    app = importlib.import_module('flybrain.food_app')
    loop = app.build_simulation(7)
    reference = app.build_simulation(7)
    model = loop.controller.decoder.model
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    def signature(frame):
        return (frame.body, frame.command, frame.pre_movement_observation,
                frame.activity.spike_counts.tolist(), frame.activity.end_step)
    pygame.font.init()
    try:
        world, brain = app.build_views(pygame.Surface((1200, 600)), loop)
        first = []
        for tick in range(30):
            frame = loop.step()
            first.append(signature(frame))
            assert frame.activity.end_step == (tick + 1) * 20
            brain.observe(frame)
            before = loop.controller.network.snapshot()
            world.draw(loop.world)
            brain.draw()
            brain.draw()
            after = loop.controller.network.snapshot()
            assert before.step_count == after.step_count
            assert torch.equal(before.voltage, after.voltage)
            assert signature(reference.step()) == first[-1]
        loop.reset(7)
        brain.reset()
        assert brain.model.tick_id == 0
        assert [signature(loop.step()) for _ in range(30)] == first
        loop.reset(7)
        loop.controller.set_motor_enabled(False)
        still = [loop.step() for _ in range(10)]
        assert still[0].body == still[-1].body
        assert all(frame.command.forward == frame.command.turn == 0 for frame in still)
        assert still[-1].activity.end_step == 200
        assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())
    finally:
        pygame.font.quit()


def test_food_app_random_trial_and_exact_replay_keys(monkeypatch, capsys):
    from flybrain.behavior import EpisodeSequence
    app = importlib.import_module('flybrain.food_app')
    loop = app.build_simulation(7)
    resets = []
    original_reset = loop.reset
    def record_reset(seed):
        resets.append(seed)
        original_reset(seed)
    monkeypatch.setattr(loop, 'reset', record_reset)
    monkeypatch.setattr(app, 'build_simulation', lambda seed: loop)
    events = iter(([], [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)],
                   [], [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE)],
                   [], [pygame.event.Event(pygame.QUIT)]))
    monkeypatch.setattr(pygame.event, 'get', lambda: next(events))
    assert app.main(['--seed', '7', '--headless', '--steps', '10']) == 0
    expected = EpisodeSequence(7).next_seed()
    assert resets == [expected, expected]
    assert f'Trial 2: seed={expected}' in capsys.readouterr().out
    assert not pygame.get_init()
