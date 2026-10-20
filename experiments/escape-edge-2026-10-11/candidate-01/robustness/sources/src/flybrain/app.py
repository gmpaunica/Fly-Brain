"""Pygame adapter for explicit manual or neural closed-loop control."""

import argparse
from collections.abc import Container
from contextlib import contextmanager
import os

import pygame

from .simulation import FixedStepSimulation
from .visualization.world_view import WorldView
from .world.arena import Arena
from .world.fly import MotorCommand
from .world.odor import OdorSource


BRAIN_PANEL_WIDTH = 400


def display_size(world_size: tuple[int, int], control_mode: str) -> tuple[int, int]:
    """Return the manual world size or neural two-panel window size."""
    if control_mode not in ('manual', 'neural', 'banc'):
        raise ValueError("control_mode must be 'manual', 'neural' or 'banc'")
    width, height = world_size
    return (width + (BRAIN_PANEL_WIDTH if control_mode != 'manual' else 0), height)


def build_views(
    surface: pygame.Surface,
    world_size: tuple[int, int],
    control_mode: str,
    topology: object | None = None,
) -> tuple[WorldView, object | None]:
    """Build a world viewport and the optional lazily imported brain panel."""
    width, height = world_size
    world_surface = surface.subsurface((0, 0, width, height))
    world_view = WorldView(world_surface, control_mode=control_mode)
    if control_mode == 'manual':
        return world_view, None
    if control_mode not in ('neural', 'banc'):
        raise ValueError("control_mode must be 'manual', 'neural' or 'banc'")
    if topology is None:
        raise ValueError('neural mode requires a topology')
    from .visualization.brain_view import BrainView

    brain_surface = surface.subsurface((width, 0, BRAIN_PANEL_WIDTH, height))
    return world_view, BrainView(brain_surface, topology,
                                 kind='banc' if control_mode == 'banc' else 'synthetic')


def command_from_keys(keys: Container[int]) -> MotorCommand:
    """Map held key codes; aliases combine, opposing turns cancel."""
    forward = pygame.K_w in keys or pygame.K_UP in keys
    left = pygame.K_a in keys or pygame.K_LEFT in keys
    right = pygame.K_d in keys or pygame.K_RIGHT in keys
    return MotorCommand(float(forward), float(right) - float(left))


def positive_steps(value: str) -> int:
    """Argparse converter requiring a positive integer physics-tick limit."""
    try:
        steps = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('steps must be a positive integer') from exc
    if steps <= 0:
        raise argparse.ArgumentTypeError('steps must be a positive integer')
    return steps


@contextmanager
def headless_drivers(enabled: bool):
    """Temporarily configure SDL dummy drivers before Pygame initialization."""
    names = ('SDL_VIDEODRIVER', 'SDL_AUDIODRIVER')
    original = {name: os.environ.get(name) for name in names}
    try:
        if enabled:
            for name in names:
                os.environ[name] = 'dummy'
        yield
    finally:
        if enabled:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def main(argv: list[str] | None = None) -> int:
    """Run manual demo or exact-tick headless smoke; always close Pygame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--steps', type=positive_steps, help='stop after N physics ticks')
    parser.add_argument('--control', choices=('manual', 'neural', 'banc'), default='manual')
    args = parser.parse_args(argv)
    if args.headless and args.steps is None:
        parser.error('--headless requires --steps')

    arena = Arena()
    source = OdorSource(0.75 * arena.width, 0.5 * arena.height)
    simulation = FixedStepSimulation(arena, arena.spawn(args.seed), odor_source=source)
    closed_loop = None
    network = None
    if args.control != 'manual':
        try:
            # Keep the default/manual import path usable without PyTorch.
            from .closed_loop import ClosedLoopSimulation
            from .neural.controller import NeuralController
            from .neural.motor import MotorDecoder
            from .neural.network import SyntheticNetwork
        except ModuleNotFoundError as exc:
            if exc.name == 'torch':
                parser.error('neural control requires the optional neural dependencies')
            raise
        if args.control == 'banc':
            from .neural.banc_network import load_verified_banc_config
            from .neural.banc_motor import make_banc_controller
            try:
                controller = make_banc_controller(load_verified_banc_config())
            except (FileNotFoundError, ValueError) as exc:
                parser.error(f'BANC dynamics are not verified: {exc}')
            network = controller.network
        else:
            network = SyntheticNetwork(seed=args.seed)
            controller = NeuralController(network, MotorDecoder(network.topology))
        closed_loop = ClosedLoopSimulation(simulation, controller)
    with headless_drivers(args.headless):
        try:
            pygame.init()
            world_size = (int(arena.width), int(arena.height))
            surface = pygame.display.set_mode(display_size(world_size, args.control))
            mode_title = {'manual': 'MANUAL DEBUG', 'neural': 'NEURAL CLOSED LOOP',
                          'banc': 'BANC SUBSET | UNTRAINED'}[args.control]
            pygame.display.set_caption(f'FlyBrain — {mode_title}')
            view, brain_view = build_views(
                surface,
                world_size,
                args.control,
                network.topology if network is not None else None,
            )
            if args.headless:
                for _ in range(args.steps):
                    if closed_loop is None:
                        simulation.step(MotorCommand())
                    else:
                        snapshot = closed_loop.step()
                        brain_view.observe(snapshot)
                    view.draw(simulation)
                    if brain_view is not None:
                        brain_view.draw()
                    pygame.display.flip()
                print(f'Headless smoke complete: mode={args.control} '
                      f'ticks={simulation.tick_count}')
                return 0

            clock = pygame.time.Clock()
            running = True
            while running:
                elapsed = min(clock.tick(60) / 1000.0, 0.25)
                reset = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            if closed_loop is None:
                                simulation.reset(args.seed)
                            else:
                                closed_loop.reset(args.seed)
                                brain_view.reset()
                            reset = True
                if not running:
                    break
                if not reset:
                    # The optional bound counts physics ticks, never drawn frames.
                    if args.steps is not None:
                        elapsed = min(elapsed, (args.steps - simulation.tick_count) * simulation.dt)
                    if closed_loop is None:
                        pressed = pygame.key.get_pressed()
                        key_codes = (pygame.K_w, pygame.K_UP, pygame.K_a,
                                     pygame.K_LEFT, pygame.K_d, pygame.K_RIGHT)
                        command = command_from_keys(
                            {key for key in key_codes if pressed[key]}
                        )
                        simulation.advance(elapsed, command)
                    else:
                        for snapshot in closed_loop.advance_snapshots(elapsed):
                            brain_view.observe(snapshot)
                view.draw(simulation)
                if brain_view is not None:
                    brain_view.draw()
                pygame.display.flip()
                if args.steps is not None and simulation.tick_count >= args.steps:
                    running = False
            return 0
        finally:
            pygame.quit()


if __name__ == '__main__':
    raise SystemExit(main())
