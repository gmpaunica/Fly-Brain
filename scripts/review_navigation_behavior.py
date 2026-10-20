"""One capped development diagnosis of the existing arrival-trained policy.

This uses two already exposed starts, continues beyond arrival, and never fits.
Run once from the project root; the output directory refuses replacement.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import torch

from flybrain.closed_loop import ClosedLoopSimulation
from flybrain.neural.verified_banc import load_frozen_controller
from flybrain.simulation import FixedStepSimulation
from flybrain.training.evaluation import locked_cases
from flybrain.world.arena import Arena
from flybrain.world.fly import Fly, MAX_FORWARD_SPEED
from flybrain.world.odor import OdorSource


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'experiments/behavior-review-2026-09-28'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main() -> None:
    """Save declaration before stepping, then all measured trajectories/results."""
    OUTPUT.mkdir()  # Refuse another run over these diagnostic results.
    arena = Arena()
    app_fly = arena.spawn(7)
    train = locked_cases()['train'][0]
    cases = [
        {'case_id': 'historical-app-seed-7', 'seed': 7,
         'x': app_fly.x, 'y': app_fly.y, 'heading': app_fly.heading},
        {'case_id': 'historical-train-0', 'seed': train.seed,
         'x': train.x, 'y': train.y, 'heading': train.heading},
    ]
    existing = [*sorted((ROOT / 'src').rglob('*.py')),
                *sorted(p for p in (ROOT / 'experiments').rglob('*')
                        if p.is_file() and OUTPUT not in p.parents)]
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in existing}
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Development diagnosis, historical starts; not a fresh assay',
        'hypothesis': 'Constant-speed imitation and first-arrival scoring leave '
                      'the saved controller moving through or around food.',
        'cases': cases, 'world_dt_seconds': .02,
        'source': asdict(OdorSource(600., 300.)),
        'horizon_ticks_per_case': 1000, 'max_episodes': 2,
        'max_world_ticks': 2000, 'max_neural_steps': 40000,
        'optimizer_steps': 0, 'learning_rate': None,
        'network_size': 83, 'readout_parameters': 370,
        'checkpoint': 'experiments/fb10-extra-training/trained.pt',
        'checkpoint_sha256': digest(ROOT / 'experiments/fb10-extra-training/trained.pt'),
        'development_measures': {'arrival_radius': 20., 'slow_speed_max': 1.,
                                 'slow_angular_speed_max': .05, 'hold_ticks': 100},
        'stop_condition': 'Run exactly these two full horizons or stop on error; '
                          'no fit, selection, retries, or new evaluation identities.',
        'software': {'python': platform.python_version(), 'torch': torch.__version__,
                     'device': 'cpu', 'platform': platform.platform()},
        'commit': None, 'source_and_history_sha256': hashes,
    }
    write_json(OUTPUT / 'protocol.json', protocol)
    write_json(OUTPUT / 'execution.json', {'status': 'consumed', 'optimizer_steps': 0})
    torch.set_num_threads(1)
    controller = load_frozen_controller()
    model = controller.decoder.model
    before = {k: v.clone() for k, v in model.state_dict().items()}
    # tanh hidden activations are in [-1,1], giving a global forward bound.
    bias = float(model.output.bias[0])
    magnitude = float(model.output.weight[0].abs().sum())
    bounds = [100. / (1. + math.exp(-raw)) for raw in (bias - magnitude, bias + magnitude)]
    with (ROOT / 'experiments/fb10-projection-data/samples.csv').open(newline='') as stream:
        targets = [float(row['target_forward']) for row in csv.DictReader(stream)]
    assay = json.loads((ROOT / 'experiments/fb10-extra-evaluation/protocol.json').read_text())
    orientation = {}
    for split, group in assay['cases'].items():
        headings = [math.degrees(row['heading']) for row in group]
        orientation[split] = {'count': len(group), 'unique_headings': len(set(headings)),
                              'min_degrees': min(headings), 'max_degrees': max(headings),
                              'quadrant_counts': [sum(q * 90 <= h < (q + 1) * 90 for h in headings)
                                                  for q in range(4)]}
    summaries = []
    total_ticks = total_steps = 0
    fields = ['case_id', 'tick', 'seconds', 'x', 'y', 'heading', 'distance',
              'forward', 'turn', 'commanded_speed', 'displacement_speed',
              'inside_arrival_radius', 'slow_hold_ticks']
    with (OUTPUT / 'trajectories.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            controller.reset()
            fly = Fly(case['x'], case['y'], case['heading'])
            loop = ClosedLoopSimulation(FixedStepSimulation(arena, fly,
                odor_source=OdorSource(600., 300.)), controller)
            first = None
            hold = maximum_hold = inside_ticks = 0
            speeds, near_speeds, distances = [], [], []
            with torch.no_grad():
                for tick in range(1, 1001):
                    px, py = fly.x, fly.y
                    frame = loop.step()
                    speed = frame.command.forward * MAX_FORWARD_SPEED
                    displacement = math.hypot(fly.x - px, fly.y - py) / .02
                    distance = math.hypot(fly.x - 600., fly.y - 300.)
                    inside = distance <= 20.
                    if inside and first is None:
                        first = {'tick': tick, 'seconds': tick * .02, 'speed': speed}
                    slow = (inside and speed <= 1. and displacement <= 1.
                            and abs(fly.angular_speed) <= .05)
                    hold = hold + 1 if slow else 0
                    maximum_hold = max(hold, maximum_hold)
                    inside_ticks += int(inside)
                    speeds.append(speed)
                    distances.append(distance)
                    if inside:
                        near_speeds.append(speed)
                    writer.writerow(dict(zip(fields, [case['case_id'], tick, tick * .02,
                        fly.x, fly.y, fly.heading, distance, frame.command.forward,
                        frame.command.turn, speed, displacement, inside, hold])))
            total_ticks += loop.world.tick_count
            total_steps += controller.network.population.step_count
            summaries.append({'case': case, 'first_arrival': first,
                'min_distance': min(distances), 'final_distance': distances[-1],
                'speed_min': min(speeds), 'speed_max': max(speeds),
                'speed_inside_radius_min': min(near_speeds) if near_speeds else None,
                'time_inside_radius_seconds': inside_ticks * .02,
                'longest_slow_hold_seconds': maximum_hold * .02,
                'settled_for_two_seconds': maximum_hold >= 100})
    cap_concentration = (96. / 176.) ** 2
    result = {
        'diagnostic_only': True, 'cases': summaries,
        'training_targets': {'count': len(targets), 'min': min(targets),
                             'max': max(targets), 'unique_count': len(set(targets))},
        'historical_assay_headings': orientation,
        'global_saved_model_speed_bounds_units_per_second': bounds,
        'sensory_cap': {'concentration': cap_concentration,
                        'antenna_distance_at_cap': math.sqrt(1000. / cap_concentration - 400.)},
        'world_ticks': total_ticks, 'neural_steps': total_steps, 'optimizer_steps': 0,
        'parameters_unchanged': all(torch.equal(before[k], v) for k, v in model.state_dict().items()),
        'prior_files_unchanged': all(digest(ROOT / p) == value for p, value in hashes.items()),
    }
    write_json(OUTPUT / 'results.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
