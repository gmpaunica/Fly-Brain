"""Draw world state without advancing physics or consuming random numbers."""

import math
import pygame

from flybrain.simulation import FixedStepSimulation


class WorldView:
    """Arena, body heading and a clearly labeled read-only control HUD."""

    def __init__(self, surface: pygame.Surface,
                 control_mode: str = 'manual') -> None:
        if control_mode not in ('manual', 'neural', 'banc'):
            raise ValueError("control_mode must be 'manual', 'neural' or 'banc'")
        self.surface = surface
        self.control_mode = control_mode
        self.font = pygame.font.Font(None, 24)

    def draw(self, simulation: FixedStepSimulation) -> None:
        """Read and render the current pose; never change simulation state."""
        surface = self.surface
        surface.fill((18, 24, 32))
        pygame.draw.rect(surface, (90, 110, 130), surface.get_rect(), 2)
        source = simulation.odor_source
        if source is not None:
            source_center = (round(source.x), round(source.y))
            pygame.draw.circle(surface, (85, 190, 120), source_center, 10)
            surface.blit(self.font.render('Food odor source', True, (140, 220, 170)),
                         (source_center[0] + 14, source_center[1] - 8))
        fly = simulation.fly
        center = (round(fly.x), round(fly.y))
        for point, color in zip(fly.antenna_positions(),
                                ((80, 190, 250), (250, 125, 175))):
            antenna = (round(point[0]), round(point[1]))
            pygame.draw.line(surface, color, center, antenna, 1)
            pygame.draw.circle(surface, color, antenna, 3)
        pygame.draw.circle(surface, (240, 190, 70), center, round(fly.radius))
        tip = (round(fly.x + 16 * math.cos(fly.heading)),
               round(fly.y + 16 * math.sin(fly.heading)))
        pygame.draw.line(surface, (255, 245, 210), center, tip, 2)
        if self.control_mode == 'manual':
            lines = [
                'FlyBrain | MANUAL DEBUG',
                'W / Up: forward   A / Left: turn left   D / Right: turn right',
                'R: reset   Esc / close: quit   Opposing turns cancel',
            ]
        elif self.control_mode == 'banc':
            lines = [
                'FlyBrain | BANC v888 SUBSET',
                '83 sourced cells | assumed dynamics | untrained motor readout',
                'R: reset   Esc / close: quit   Movement keys disabled',
            ]
        else:
            lines = [
                'FlyBrain | NEURAL CLOSED LOOP',
                'Sensory-only spiking controller active (no learning)',
                'R: reset   Esc / close: quit   Movement keys disabled',
            ]
        lines.extend([
            f'x={fly.x:.1f}  y={fly.y:.1f}  heading={fly.heading:.3f} rad  ticks={simulation.tick_count}',
            f'Odor (current pose): L={simulation.observation.odor_left:.4e}  R={simulation.observation.odor_right:.4e}',
        ])
        for index, text in enumerate(lines):
            surface.blit(self.font.render(text, True, (220, 230, 240)), (14, 12 + index * 24))
