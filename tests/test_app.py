import os
import subprocess
import sys
from dataclasses import asdict

import pytest


def cli(*args):
    return subprocess.run([sys.executable,'-m','flybrain.app',*args], capture_output=True, text=True, timeout=20)


def test_headless_cli():
    result = cli('--headless','--steps','5','--seed','7')
    assert result.returncode == 0, result.stderr
    assert 'ticks=5' in result.stdout


def test_neural_headless_cli_runs_exact_ticks():
    result = cli('--headless','--steps','120','--seed','7','--control','neural')
    assert result.returncode == 0, result.stderr
    assert 'mode=neural ticks=120' in result.stdout


def test_importing_manual_app_does_not_import_torch():
    code = "import sys; import flybrain.app; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith('False')


@pytest.mark.parametrize('args', [
    ('--headless',), ('--steps','0'), ('--steps','-1'), ('--steps','1.5'),
    ('--control','automatic'),
])
def test_invalid_cli(args):
    assert cli(*args).returncode == 2


def test_manual_key_mapping():
    import pygame
    from flybrain.app import command_from_keys
    assert command_from_keys(set()).forward == 0
    for key in [pygame.K_w,pygame.K_UP]:
        assert command_from_keys({key}).forward == 1
    for key in [pygame.K_a,pygame.K_LEFT]:
        assert command_from_keys({key}).turn == -1
    for key in [pygame.K_d,pygame.K_RIGHT]:
        assert command_from_keys({key}).turn == 1
    assert command_from_keys({pygame.K_a,pygame.K_RIGHT}).turn == 0


def test_neural_mode_builds_side_by_side_world_and_brain_views():
    import pygame
    from flybrain.app import build_views, display_size
    from flybrain.neural.network import SyntheticNetwork

    pygame.font.init()
    try:
        network = SyntheticNetwork(seed=7)
        assert display_size((800, 600), 'manual') == (800, 600)
        assert display_size((800, 600), 'neural') == (1200, 600)
        surface = pygame.Surface((1200, 600))

        world_view, brain_view = build_views(
            surface,
            (800, 600),
            'neural',
            network.topology,
        )

        assert world_view.surface.get_size() == (800, 600)
        assert brain_view.surface.get_size() == (400, 600)
        assert brain_view.topology.labels == network.topology.labels
    finally:
        pygame.quit()


def test_headless_exact_ticks_environment_cleanup_and_readonly_draw(monkeypatch):
    import pygame
    from flybrain.app import main
    from flybrain.simulation import FixedStepSimulation
    from flybrain.world.arena import Arena
    from flybrain.visualization.world_view import WorldView
    monkeypatch.setenv('SDL_VIDEODRIVER','original-video')
    monkeypatch.delenv('SDL_AUDIODRIVER',raising=False)
    assert main(['--headless','--steps','5']) == 0
    assert os.environ['SDL_VIDEODRIVER'] == 'original-video'
    assert 'SDL_AUDIODRIVER' not in os.environ
    assert not pygame.get_init()
    pygame.font.init()
    try:
        arena = Arena()
        sim = FixedStepSimulation(arena,arena.spawn(7))
        before = asdict(sim.fly)
        surface = pygame.Surface((800,600))
        view = WorldView(surface)
        for _ in range(3):
            view.draw(sim)
        assert asdict(sim.fly) == before
        assert sim.tick_count == 0
        assert surface.get_at((400,300))[:3] != surface.get_at((300,300))[:3]
    finally:
        pygame.quit()
