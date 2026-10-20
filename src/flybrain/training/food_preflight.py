"""Locked full-circle cases and renderer-free teacher behavior preflight."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

from flybrain.behavior import FoodVisit
from flybrain.simulation import FixedStepSimulation
from flybrain.world.arena import Arena
from flybrain.world.fly import Fly, MAX_FORWARD_SPEED
from flybrain.world.food import FoodPatch, sample_food
from flybrain.world.odor import OdorSource

from .evaluation import EpisodeCase
from .food_teacher import FoodTeacher


HORIZON_TICKS = 1000
SOURCE_X, SOURCE_Y = 600., 300.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / 'experiments/food-stop-preflight-2026-09-28'


def generate_cases() -> dict[str, tuple[EpisodeCase, ...]]:
    """Preselect 120 disjoint seeds and balance headings in eight sectors."""
    groups: dict[str, tuple[EpisodeCase, ...]] = {}
    for split, seed_base, count in (('train', 810000, 32),
                                    ('validation', 820000, 16),
                                    ('final', 830000, 64)):
        cases = []
        for index in range(count):
            seed = seed_base + index
            rng = random.Random(seed)
            heading = (index % 8 + rng.random()) * math.tau / 8
            for _ in range(1000):
                radius = rng.uniform(60, 220)
                angle = rng.uniform(0, math.tau)
                x = SOURCE_X + radius * math.cos(angle)
                y = SOURCE_Y + radius * math.sin(angle)
                if 6 <= x <= 794 and 6 <= y <= 594:
                    break
            else:
                raise RuntimeError('could not generate an in-bounds case')
            cases.append(EpisodeCase(f'food-{split}-{seed}', split, seed, x, y, heading))
        groups[split] = tuple(cases)
    edges = tuple(EpisodeCase(f'food-train-edge-{index}', 'train', 810100 + index,
                              400., 300., index * math.tau / 8)
                  for index in range(8))
    groups['train'] += edges
    return groups


def run_teacher_case(case: EpisodeCase) -> dict:
    """Run all 20 seconds and score contact, approach, exact stop and dwell."""
    fly = Fly(case.x, case.y, case.heading)
    source = OdorSource(SOURCE_X, SOURCE_Y)
    patch = FoodPatch(SOURCE_X, SOURCE_Y)
    world = FixedStepSimulation(Arena(), fly, odor_source=source)
    teacher = FoodTeacher()
    visit = FoodVisit()
    first_arrival_tick = None
    first_contact_tick = None
    first_stop_tick = None
    departures = 0
    prior_contact = False
    speeds: list[float] = []
    min_distance = math.hypot(fly.x - SOURCE_X, fly.y - SOURCE_Y)
    for tick in range(1, HORIZON_TICKS + 1):
        observation = sample_food(fly, source, patch)
        command = teacher.command(observation)
        x, y = fly.x, fly.y
        world.step(command)
        displacement_speed = math.hypot(fly.x - x, fly.y - y) / world.dt
        post_contact = patch.touches(fly)
        visit.observe(tick, post_contact, command, displacement_speed)
        distance = math.hypot(fly.x - SOURCE_X, fly.y - SOURCE_Y)
        min_distance = min(min_distance, distance)
        if first_arrival_tick is None and distance <= 20:
            first_arrival_tick = tick
        if first_contact_tick is None:
            speeds.append(command.forward * MAX_FORWARD_SPEED)
            if post_contact:
                first_contact_tick = tick
        if first_contact_tick is not None and first_stop_tick is None and command.forward == 0 and command.turn == 0 and displacement_speed == 0:
            first_stop_tick = tick
        if prior_contact and not post_contact:
            departures += 1
        prior_contact = post_contact
    approach_speed = (sum(speeds[-25:]) / min(25, len(speeds))) if speeds else None
    stop_latency = ((first_stop_tick - first_contact_tick) * world.dt
                    if first_contact_tick is not None and first_stop_tick is not None else None)
    success = (first_contact_tick is not None and first_contact_tick <= 850
               and approach_speed is not None and approach_speed <= 30
               and stop_latency is not None and stop_latency <= .2
               and visit.contact and visit.settled and departures == 0
               and visit.hold_seconds >= 2)
    return {**asdict(case), 'success': success,
            'first_arrival_seconds': first_arrival_tick * world.dt if first_arrival_tick else None,
            'first_contact_seconds': first_contact_tick * world.dt if first_contact_tick else None,
            'approach_speed': approach_speed, 'stop_latency_seconds': stop_latency,
            'overshoot_units': max(0., 16. - min_distance), 'departures': departures,
            'stationary_dwell_seconds': visit.hold_seconds,
            'final_contact': visit.contact, 'final_distance': distance,
            'final_commanded_speed': visit.commanded_speed,
            'final_actual_speed': visit.displacement_speed}


def prepare_preflight(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    """Lock the train-only teacher gate and all future case identities first."""
    if output_dir.exists():
        raise FileExistsError('preflight output already exists')
    cases = generate_cases()
    source_files = ('src/flybrain/neural/food_contact.py',
                    'src/flybrain/training/food_teacher.py',
                    'src/flybrain/training/food_preflight.py')
    record = {
        'schema': 'food_stop_preflight_v1',
        'hypothesis': 'A sensory-only demonstration controller reaches food, slows, and remains stopped on all 40 training starts.',
        'cases': {split: [asdict(case) for case in group]
                  for split, group in cases.items()},
        'software_device': {'python': __import__('sys').version.split()[0], 'device': 'CPU'},
        'network_size': 85, 'optimizer_steps': 0, 'learning_rate': None,
        'objective': 'All training cases meet contact-by-17s, approach <=30 units/s, exact stop within 0.2s and remaining stationary contact >=2s.',
        'budget': {'teacher_episodes': 40, 'world_ticks': 40000, 'neural_steps': 0},
        'stop_condition': 'If any teacher training case fails, record all preselected training outcomes and stop before collecting neural training data or fitting.',
        'source_sha256': {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
                          for name in source_files},
        'prepared_utc': datetime.now(timezone.utc).isoformat(),
        'commit': None,
    }
    output_dir.mkdir(parents=True)
    path = output_dir / 'protocol.json'
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return path


def run_preflight(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    """Consume the prepared teacher probe exactly once and save every outcome."""
    protocol_path = output_dir / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if (protocol.get('schema') != 'food_stop_preflight_v1'
            or protocol.get('cases') != {split: [asdict(case) for case in group]
                                         for split, group in generate_cases().items()}
            or any(hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != digest
                   for name, digest in protocol['source_sha256'].items())):
        raise ValueError('preflight protocol differs from installed candidate')
    if (output_dir / 'results.json').exists():
        raise FileExistsError('preflight result already exists')
    receipt = output_dir / 'execution.json'
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump({'status': 'consumed',
                   'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                   'started_utc': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        stream.write('\n')
    rows = [run_teacher_case(case) for case in generate_cases()['train']]
    result = {'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              'teacher_cases': rows, 'successes': sum(row['success'] for row in rows),
              'passed': all(row['success'] for row in rows),
              'actual_episodes': len(rows), 'actual_world_ticks': len(rows) * HORIZON_TICKS,
              'actual_neural_steps': 0, 'optimizer_steps': 0,
              'completed_utc': datetime.now(timezone.utc).isoformat()}
    (output_dir / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n',
                                             encoding='utf-8')
    return result
