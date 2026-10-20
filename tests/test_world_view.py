import random
from dataclasses import asdict

import pygame
import pytest

from flybrain.simulation import FixedStepSimulation
from flybrain.visualization.world_view import WorldView
from flybrain.world.arena import Arena
from flybrain.world.fly import Fly, MotorCommand
from flybrain.world.odor import OdorSource


@pytest.mark.parametrize('control_mode', ['manual', 'neural'])
def test_source_antennae_and_readings_render_without_changing_episode(control_mode):
    pygame.font.init()
    try:
        source = OdorSource(600,300)
        sim = FixedStepSimulation(Arena(),Fly(400,300),odor_source=source)
        surface = pygame.Surface((800,600))
        view = WorldView(surface, control_mode=control_mode)
        initial_pose = asdict(sim.fly)
        initial_observation = sim.observation
        initial_rng = random.getstate()
        for _ in range(3):
            view.draw(sim)
        assert asdict(sim.fly) == initial_pose
        assert sim.observation == initial_observation
        assert sim.tick_count == 0
        assert random.getstate() == initial_rng
        background = surface.get_at((300,300))[:3]
        for point in [(600,300),(410,295),(410,305)]:
            assert surface.get_at(point)[:3] != background
        # Sensor HUD occupies a separate line below the existing pose HUD.
        assert any(surface.get_at((x,y))[:3] != background
                   for x in range(14,700) for y in range(110,126))
        reference = FixedStepSimulation(Arena(),Fly(400,300),odor_source=source)
        for _ in range(20):
            sim.step(MotorCommand(1,0.25))
            view.draw(sim)
            view.draw(sim)
            reference.step(MotorCommand(1,0.25))
        assert asdict(sim.fly) == asdict(reference.fly)
        assert sim.observation == reference.observation
    finally:
        pygame.quit()


def test_control_modes_render_distinct_hud_labels_and_unknown_mode_is_rejected():
    pygame.font.init()
    try:
        simulation = FixedStepSimulation(Arena(), Fly(400,300))
        manual_surface = pygame.Surface((800,600))
        neural_surface = pygame.Surface((800,600))
        WorldView(manual_surface, control_mode='manual').draw(simulation)
        WorldView(neural_surface, control_mode='neural').draw(simulation)

        assert pygame.image.tobytes(manual_surface, 'RGB') != pygame.image.tobytes(
            neural_surface, 'RGB'
        )
        with pytest.raises(ValueError):
            WorldView(pygame.Surface((800,600)), control_mode='automatic')
    finally:
        pygame.quit()
