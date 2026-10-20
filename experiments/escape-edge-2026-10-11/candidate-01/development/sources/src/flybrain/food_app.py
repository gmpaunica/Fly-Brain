"""Interactive frozen approach-and-stop controller with replayable trials."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import TYPE_CHECKING

import pygame

from . import app as original_app
from . import verified_app
from .behavior import EpisodeSequence
from .simulation import FixedStepSimulation
from .verified_app import VerifiedWorldView
from .world.arena import Arena
from .world.odor import OdorSource

if TYPE_CHECKING:
    from .foraging import ForagingSimulation
    from .visualization.food_brain_view import FoodBrainView


class FoodWorldView(VerifiedWorldView):
    """Show the promoted policy, trial identity and measured stop behavior."""

    def __init__(self, surface: pygame.Surface, loop: ForagingSimulation) -> None:
        super().__init__(surface, loop, trained=True)

    def draw(self, simulation: FixedStepSimulation) -> None:
        super().draw(simulation)
        pygame.draw.rect(self.surface, (18, 24, 32),
                         (3, 3, self.surface.get_width() - 6, 80))
        for index, line in enumerate((
            'FlyBrain | 85-CELL FROZEN APPROACH + STOP',
            '83 sourced BANC cells + 2 engineered contact cells | learned readout',
            'R: new random trial   Backspace: exact replay   Esc / close: quit',
        )):
            self.surface.blit(self.font.render(line, True, (220, 230, 240)),
                              (14, 12 + index * 24))


def build_simulation(seed: int) -> ForagingSimulation:
    """Load the accepted frozen head into the existing 20 ms 2D world."""
    from .foraging import ForagingSimulation
    from .neural.accepted_food import load_accepted_controller

    arena = Arena()
    world = FixedStepSimulation(arena, arena.spawn(seed),
                                odor_source=OdorSource(.75 * arena.width,
                                                       .5 * arena.height))
    return ForagingSimulation(world, load_accepted_controller())


def build_views(surface: pygame.Surface, loop: ForagingSimulation) -> tuple[FoodWorldView, FoodBrainView]:
    """Create read-only world and sourced/engineered brain panels."""
    from .visualization.food_brain_view import FoodBrainView

    width, height = int(loop.world.arena.width), int(loop.world.arena.height)
    network = loop.controller.network
    topology = network.topology
    sourced = replace(topology, nodes=topology.nodes[:83],
                      root_ids=topology.root_ids[:83], groups=topology.groups[:83],
                      _weights=topology.weights[:83, :83])
    return (FoodWorldView(surface.subsurface((0, 0, width, height)), loop),
            FoodBrainView(surface.subsurface((width, 0,
                                             original_app.BRAIN_PANEL_WIDTH, height)),
                          sourced))


def main(argv: list[str] | None = None) -> int:
    """Run accepted food inference, retaining all historical control modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help='replayable trial sequence; omitted uses a fresh seed')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--steps', type=original_app.positive_steps,
                        help='stop after N physics ticks')
    parser.add_argument('--control', choices=('food-trained', 'banc-trained', 'banc',
                                              'neural', 'manual'), default='food-trained')
    args = parser.parse_args(argv)
    if args.headless and args.steps is None:
        parser.error('--headless requires --steps')
    if args.control != 'food-trained':
        forwarded = ['--control', args.control]
        if args.seed is not None:
            forwarded.extend(('--seed', str(args.seed)))
        if args.headless:
            forwarded.append('--headless')
        if args.steps is not None:
            forwarded.extend(('--steps', str(args.steps)))
        return verified_app.main(forwarded)
    try:
        episodes = EpisodeSequence(args.seed)
        loop = build_simulation(episodes.current_seed)
    except ModuleNotFoundError as exc:
        if exc.name == 'torch':
            parser.error('food-trained control requires the optional neural dependencies')
        raise
    except (OSError, ValueError) as exc:
        parser.error(f'Accepted food-stop controller is unavailable: {exc}')

    with original_app.headless_drivers(args.headless):
        try:
            pygame.init()
            world_size = (int(loop.world.arena.width), int(loop.world.arena.height))
            surface = pygame.display.set_mode(original_app.display_size(world_size, 'banc'))
            pygame.display.set_caption('FlyBrain | 85-CELL FROZEN APPROACH + STOP')
            world, brain = build_views(surface, loop)
            world.episodes = episodes
            print(f'Trial 1: seed={episodes.current_seed} '
                  f'heading={loop.world.fly.heading:.6f} rad', flush=True)
            clock = pygame.time.Clock()
            running = True
            while running:
                elapsed = loop.world.dt if args.headless else min(clock.tick(60) / 1000., .25)
                reset = False
                for event in pygame.event.get():
                    if (event.type == pygame.QUIT
                            or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
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
                print(f'Headless smoke {outcome}: mode=food-trained '
                      f'ticks={loop.world.tick_count} checkpoint=approach-stop')
            return 0
        finally:
            pygame.quit()


if __name__ == '__main__':
    raise SystemExit(main())
