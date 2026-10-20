"""Save an actual verified-BANC frame for static visual review, not a live UI test."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pygame

from flybrain.app import build_views, headless_drivers
from flybrain.closed_loop import ClosedLoopSimulation
from flybrain.neural.banc_motor import make_banc_controller
from flybrain.neural.banc_network import load_verified_banc_config
from flybrain.simulation import FixedStepSimulation
from flybrain.world.arena import Arena
from flybrain.world.odor import OdorSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/fb09-app-verification'))
    args = parser.parse_args()
    output = args.output_dir
    if output.exists():
        raise FileExistsError('Choose an unused visual verification directory')
    config = load_verified_banc_config()
    controller = make_banc_controller(config)
    arena = Arena()
    world = FixedStepSimulation(arena, arena.spawn(7), odor_source=OdorSource(600, 300))
    loop = ClosedLoopSimulation(world, controller)
    with headless_drivers(True):
        pygame.init()
        try:
            surface = pygame.display.set_mode((1200, 600))
            world_view, brain_view = build_views(surface, (800, 600), 'banc', controller.network.topology)
            for _ in range(120):
                snapshot = loop.step()
                brain_view.observe(snapshot)
                world_view.draw(world)
                brain_view.draw()
            pixels = pygame.image.tobytes(surface, 'RGB')
            output.mkdir(parents=True)
            pygame.image.save(surface, str(output / 'banc-frame.png'))
            record = {'kind': 'static headless frame, not live interactive acceptance',
                      'seed': 7, 'world_ticks': world.tick_count,
                      'neural_substeps': controller.network.population.step_count,
                      'config': asdict(config), 'body': asdict(snapshot.body),
                      'command': asdict(snapshot.command), 'nodes': len(brain_view.positions),
                      'displayed_edges': len(brain_view.edges),
                      'effective_edges': len(controller.network.topology.edges),
                      'rgb_sha256': hashlib.sha256(pixels).hexdigest(),
                      'png_sha256': hashlib.sha256((output / 'banc-frame.png').read_bytes()).hexdigest(),
                      'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            with (output / 'frame.json').open('x', encoding='utf-8') as file:
                json.dump(record, file, indent=2, allow_nan=False)
            print(json.dumps(record, indent=2))
        finally:
            pygame.quit()


if __name__ == '__main__':
    main()
