"""Inspect arrival-trained and fixed BANC behavior across replayable random trials."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pygame

from . import app as original_app
from .behavior import EpisodeSequence
from .simulation import FixedStepSimulation
from .trained_app import TrainedWorldView
from .world.arena import Arena
from .world.odor import OdorSource

if TYPE_CHECKING:
    from .foraging import ForagingSimulation
    from .visualization.brain_view import BrainView


class VerifiedWorldView(TrainedWorldView):
    """Expose trial identity and measured behavior without changing the policy."""

    def __init__(self, surface: pygame.Surface, loop: ForagingSimulation,
                 trained: bool = True) -> None:
        super().__init__(surface, 'banc')
        self.loop = loop
        self.trained = trained
        self.episodes: EpisodeSequence | None = None

    def draw(self, simulation: FixedStepSimulation) -> None:
        super().draw(simulation)
        pygame.draw.rect(self.surface, (18, 24, 32), (3, 3, self.surface.get_width() - 6, 80))
        lines = [
            ('FlyBrain | BANC | FROZEN READOUT | 750 updates' if self.trained
             else 'FlyBrain | BANC SUBSET | UNTRAINED'),
            ('Learned task: food arrival only | stopping has not been learned' if self.trained
             else 'Fixed motor decoder | no learned stopping'),
            'R: new random trial   Backspace: replay trial   Esc / close: quit',
        ]
        for index, text in enumerate(lines):
            self.surface.blit(self.font.render(text, True, (220, 230, 240)), (14, 12 + index * 24))
        visit = self.loop.visit
        trial = (f'Trial {self.episodes.episode_number} | seed {self.episodes.current_seed}'
                 if self.episodes is not None else 'Behavior observation')
        y = self.surface.get_height() - 79
        pygame.draw.rect(self.surface, (18, 24, 32), (3, y, self.surface.get_width() - 6, 76))
        for index, text in enumerate((
            trial,
            f'Speed: {visit.commanded_speed:.1f} commanded / {visit.displacement_speed:.1f} actual units/s'
            f' | food contact: {"yes" if visit.contact else "no"}',
            f'Stationary contact: {visit.hold_seconds:.2f} / 2.00 s'
            f' | {"SETTLED" if visit.settled else "not settled"}',
        )):
            self.surface.blit(self.font.render(text, True, (220, 230, 240)), (14, y + 7 + index * 22))


def build_simulation(seed: int) -> ForagingSimulation:
    """Observe the arrival-trained controller without changing its commands."""
    from .foraging import ForagingSimulation
    from .neural.verified_banc import load_frozen_controller

    controller = load_frozen_controller()
    arena = Arena()
    world = FixedStepSimulation(arena, arena.spawn(seed),
                                odor_source=OdorSource(.75 * arena.width, .5 * arena.height))
    return ForagingSimulation(world, controller)


def build_fixed_simulation(seed: int) -> ForagingSimulation:
    """Expose the fixed comparator under identical trial and sensor semantics."""
    from .foraging import ForagingSimulation
    from .neural.banc_motor import make_banc_controller
    from .neural.banc_network import load_verified_banc_config

    arena = Arena()
    world = FixedStepSimulation(arena, arena.spawn(seed),
                                odor_source=OdorSource(.75 * arena.width, .5 * arena.height))
    return ForagingSimulation(world, make_banc_controller(load_verified_banc_config()))


def build_views(surface: pygame.Surface, loop: ForagingSimulation) -> tuple[VerifiedWorldView, BrainView]:
    """Create read-only world/brain panels with explicit frozen-readout labels."""
    from .trained_app import build_views as historical_views
    from .neural.banc_projection import ProjectionDecoder
    from .visualization.brain_view import BrainView

    width, height = int(loop.world.arena.width), int(loop.world.arena.height)
    trained = isinstance(loop.controller.decoder, ProjectionDecoder)
    if trained:
        _, brain = historical_views(surface, loop)
    else:
        brain = BrainView(surface.subsurface((width, 0, original_app.BRAIN_PANEL_WIDTH, height)),
                          loop.controller.network.topology, kind='banc')
    return VerifiedWorldView(surface.subsurface((0, 0, width, height)), loop, trained), brain


def main(argv: list[str] | None = None) -> int:
    """Run BANC trials; delegate the historical manual/synthetic debug modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help='replayable trial sequence; omitted uses a fresh seed')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--steps', type=original_app.positive_steps, help='stop after N physics ticks')
    parser.add_argument('--control', choices=('banc-trained', 'banc', 'neural', 'manual'),
                        default='banc-trained')
    args = parser.parse_args(argv)
    if args.headless and args.steps is None:
        parser.error('--headless requires --steps')
    try:
        episodes = EpisodeSequence(args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if args.control in ('manual', 'neural'):
        forwarded = ['--control', args.control, '--seed', str(episodes.current_seed)]
        if args.headless:
            forwarded.append('--headless')
        if args.steps is not None:
            forwarded.extend(('--steps', str(args.steps)))
        return original_app.main(forwarded)
    try:
        loop = (build_simulation(episodes.current_seed) if args.control == 'banc-trained'
                else build_fixed_simulation(episodes.current_seed))
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
            pygame.display.set_caption(
                'FlyBrain | BANC FROZEN READOUT | 750 updates | arrival only'
                if args.control == 'banc-trained' else 'FlyBrain | BANC SUBSET | UNTRAINED')
            world, brain = build_views(surface, loop)
            world.episodes = episodes
            print(f'Trial 1: seed={episodes.current_seed} heading={loop.world.fly.heading:.6f} rad', flush=True)
            clock = pygame.time.Clock()
            running = True
            while running:
                elapsed = loop.world.dt if args.headless else min(clock.tick(60) / 1000., .25)
                reset = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        loop.reset(episodes.next_seed())
                        brain.reset()
                        reset = True
                        print(f'Trial {episodes.episode_number}: seed={episodes.current_seed} '
                              f'heading={loop.world.fly.heading:.6f} rad', flush=True)
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
                        loop.reset(episodes.current_seed)
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
                outcome = 'complete' if loop.world.tick_count >= args.steps else 'interrupted'
                checkpoint = '750-updates' if args.control == 'banc-trained' else 'fixed'
                print(f'Headless smoke {outcome}: mode={args.control} ticks={loop.world.tick_count} '
                      f'checkpoint={checkpoint}')
            return 0
        finally:
            pygame.quit()


if __name__ == '__main__':
    raise SystemExit(main())
