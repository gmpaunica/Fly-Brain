"""Interactive frozen BANC readout; preserves the protocol-bound original app."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pygame

from . import app as original_app
from .simulation import FixedStepSimulation
from .visualization.world_view import WorldView
from .world.arena import Arena
from .world.odor import OdorSource

if TYPE_CHECKING:
    from .closed_loop import ClosedLoopSimulation
    from .visualization.brain_view import BrainView


class TrainedWorldView(WorldView):
    """Reuse world geometry, replacing the fixed-mode HUD with trained scope."""

    def draw(self, simulation: FixedStepSimulation) -> None:
        super().draw(simulation)
        pygame.draw.rect(self.surface, (18, 24, 32), (3, 3, self.surface.get_width() - 6, 80))
        for index, text in enumerate((
            'FlyBrain | BANC SUBSET | FROZEN TRAINED READOUT',
            '83 sourced cells | assumed dynamics | PN-input + endpoint readout',
            'R: reset   Esc / close: quit   Movement keys disabled',
        )):
            self.surface.blit(self.font.render(text, True, (220, 230, 240)), (14, 12 + index * 24))


def build_simulation(seed: int) -> ClosedLoopSimulation:
    """Build the accepted sensory-only controller in the original 20 ms world."""
    from .closed_loop import ClosedLoopSimulation
    from .neural.frozen_banc import load_frozen_controller

    controller = load_frozen_controller()
    arena = Arena()
    world = FixedStepSimulation(arena, arena.spawn(seed),
                                odor_source=OdorSource(.75 * arena.width, .5 * arena.height))
    return ClosedLoopSimulation(world, controller)


def build_views(surface: pygame.Surface, loop: ClosedLoopSimulation) -> tuple[WorldView, BrainView]:
    """Create read-only world/brain panels with explicit frozen-readout labels."""
    from .visualization.brain_view import BrainView

    class FrozenBrainView(BrainView):
        def draw(self) -> None:
            super().draw()
            y = self.surface.get_height() - 47
            pygame.draw.rect(self.surface, (13, 18, 29), (3, y, self.surface.get_width() - 6, 44))
            for offset, text in enumerate(('Frozen learned readout | 370 parameters',
                                           'PN inputs + endpoints | no online training')):
                self.surface.blit(self.small_font.render(text, True, (175, 190, 210)),
                                  (14, y + offset * 17))

    width, height = int(loop.world.arena.width), int(loop.world.arena.height)
    return (TrainedWorldView(surface.subsurface((0, 0, width, height)), 'banc'),
            FrozenBrainView(surface.subsurface((width, 0, original_app.BRAIN_PANEL_WIDTH, height)),
                            loop.controller.network.topology, kind='banc'))


def main(argv: list[str] | None = None) -> int:
    """Run frozen inference or delegate existing modes unchanged to the old app."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--steps', type=original_app.positive_steps, help='stop after N physics ticks')
    parser.add_argument('--control', choices=('banc-trained', 'banc', 'neural', 'manual'),
                        default='banc-trained')
    args = parser.parse_args(argv)
    if args.headless and args.steps is None:
        parser.error('--headless requires --steps')
    if args.control != 'banc-trained':
        forwarded = ['--control', args.control, '--seed', str(args.seed)]
        if args.headless:
            forwarded.append('--headless')
        if args.steps is not None:
            forwarded.extend(('--steps', str(args.steps)))
        return original_app.main(forwarded)
    try:
        loop = build_simulation(args.seed)
    except ModuleNotFoundError as exc:
        if exc.name == 'torch':
            parser.error('trained control requires the optional neural dependencies')
        raise
    except (OSError, ValueError) as exc:
        parser.error(f'Frozen BANC controller is unavailable: {exc}')

    with original_app.headless_drivers(args.headless):
        try:
            pygame.init()
            world_size = (int(loop.world.arena.width), int(loop.world.arena.height))
            surface = pygame.display.set_mode(original_app.display_size(world_size, 'banc'))
            pygame.display.set_caption('FlyBrain | BANC FROZEN TRAINED READOUT')
            world, brain = build_views(surface, loop)
            clock = pygame.time.Clock()
            running = True
            while running:
                elapsed = loop.world.dt if args.headless else min(clock.tick(60) / 1000., .25)
                reset = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        loop.reset(args.seed)
                        brain.reset()
                        reset = True
                if not running:
                    break
                if not reset:
                    if args.steps is not None:
                        elapsed = min(elapsed, (args.steps - loop.world.tick_count) * loop.world.dt)
                    for frame in loop.advance_snapshots(elapsed):
                        brain.observe(frame)
                world.draw(loop.world)
                brain.draw()
                pygame.display.flip()
                if args.steps is not None and loop.world.tick_count >= args.steps:
                    running = False
            if args.headless:
                print(f'Headless smoke complete: mode=banc-trained ticks={loop.world.tick_count}')
            return 0
        finally:
            pygame.quit()


if __name__ == '__main__':
    raise SystemExit(main())
