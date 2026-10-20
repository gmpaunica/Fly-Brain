"""BANC view and app integration use the same simulation clock."""

import pygame
import torch
import pytest

from flybrain.app import build_views, display_size
from flybrain.closed_loop import BodyState, ClosedLoopSnapshot
from flybrain.data.banc import load_banc_subset
from flybrain.neural.banc_network import BancNetwork
from flybrain.world.fly import MotorCommand
from flybrain.world.sensing import SensoryObservation


def test_charge_banc_closed_loop_reset_motor_gate_and_render_independence():
    from dataclasses import asdict
    from flybrain.closed_loop import ClosedLoopSimulation
    from flybrain.neural.banc_motor import make_banc_controller
    from flybrain.neural.banc_network import BancDynamicsConfig
    from flybrain.simulation import FixedStepSimulation
    from flybrain.world.arena import Arena
    from flybrain.world.odor import OdorSource

    config = BancDynamicsConfig(3., sensory_mode='phase_rate', projection_dynamics='charge_preserving')
    def make_loop():
        arena = Arena()
        return ClosedLoopSimulation(FixedStepSimulation(arena, arena.spawn(7),
            odor_source=OdorSource(600, 300)), make_banc_controller(config))
    def signature(frame):
        return (frame.body, frame.command, frame.pre_movement_observation,
                frame.activity.spike_counts.tolist(), frame.activity.end_step)

    loop = make_loop()
    reference = make_loop()
    pygame.font.init()
    try:
        _, view = build_views(pygame.Surface((1200, 600)), (800, 600), 'banc', loop.controller.network.topology)
        first = []
        for i in range(30):
            tick = loop.step()
            first.append(signature(tick))
            assert tick.activity.end_step == (i + 1) * 20
            view.observe(tick)
            view.draw()
            view.draw()
            assert signature(reference.step()) == signature(tick)
        loop.reset(7)
        assert [signature(loop.step()) for _ in range(30)] == first
        loop.reset(7)
        initial = asdict(loop.world.fly)
        loop.controller.set_motor_enabled(False)
        stopped = [loop.step() for _ in range(30)]
        assert asdict(loop.world.fly) == initial
        assert all(t.command == MotorCommand() for t in stopped)
        assert any(t.activity.spike_counts.sum() > 0 for t in stopped)
        assert stopped[-1].activity.end_step == 600
        loop.controller.set_motor_enabled(True)
        assert loop.step().command.forward > 0
        assert asdict(loop.world.fly) != initial
    finally:
        pygame.quit()


def test_banc_cli_refuses_unverified_dynamics(monkeypatch, capsys):
    from flybrain.app import main
    import flybrain.neural.banc_network as banc
    def missing():
        raise FileNotFoundError('no accepted dynamics')
    monkeypatch.setattr(banc, 'load_verified_banc_config', missing)
    with pytest.raises(SystemExit) as exc:
        main(['--headless', '--steps', '1', '--control', 'banc'])
    assert exc.value.code == 2
    assert 'BANC dynamics are not verified' in capsys.readouterr().err


def test_banc_view_covers_all_roles_and_drawing_never_advances_neural_state():
    pygame.font.init()
    try:
        graph = load_banc_subset(connectome_gain=1.0)
        brain = BancNetwork(graph)
        assert display_size((800, 600), 'banc') == (1200, 600)
        surface = pygame.Surface((1200, 600))
        world_view, view = build_views(surface, (800, 600), 'banc', graph)
        obs = SensoryObservation(0.025, 0)
        activity = brain.advance(obs)
        frame = ClosedLoopSnapshot(1, obs, activity, MotorCommand(), False,
                                   BodyState(400, 300, 0, 6, 0, 0))
        view.observe(frame)
        before = brain.snapshot()
        levels = view.model.activity_levels
        view.draw()
        pixels = pygame.image.tobytes(view.surface, 'RGB')
        view.draw()
        assert pygame.image.tobytes(view.surface, 'RGB') == pixels
        assert view.model.activity_levels == levels
        assert torch.equal(brain.snapshot().voltage, before.voltage)
        assert brain.snapshot().step_count == before.step_count
        assert len(view.positions) == len(set(view.positions)) == 83
        assert all(0 < x < 400 and 95 < y < 450 for x, y in view.positions)
        assert set(graph.groups) <= set(view.group_colors)
        assert world_view.control_mode == 'banc'
        view.reset()
        assert not any(view.model.activity_levels)
    finally:
        pygame.quit()
