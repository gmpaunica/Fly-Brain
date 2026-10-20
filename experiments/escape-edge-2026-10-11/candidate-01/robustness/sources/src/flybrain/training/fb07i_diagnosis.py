"""FB-07I: scoring-only tick traces for the failed validation navigation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from .records import ensure_new_run_directory
import platform
import sys
from collections.abc import Callable, Sequence

import torch

from flybrain.closed_loop import ClosedLoopSimulation
from flybrain.neural.controller import NeuralController
from flybrain.neural.readout import TrainableMotorReadout
from flybrain.simulation import FixedStepSimulation
from flybrain.world.arena import Arena
from flybrain.world.fly import Fly
from flybrain.world.odor import OdorSource

from .evaluation import (
    EpisodeCase, EvaluationConfig, _git_commit, fixed_controller, locked_cases,
)
from .fb07h_training import candidate_controller_factory, load_matched_checkpoint
from .readout_experiment import teacher_command, topology_sha256


BEARING_THRESHOLD_RADIANS = 0.1
TURN_THRESHOLD = 0.05
CONTROLLERS = ('fixed', 'seven', 'nine')


def bearing_error(x: float, y: float, heading: float,
                  source_x: float, source_y: float) -> float:
    """Signed clockwise source bearing from a pre-movement world pose."""
    target_heading = math.atan2(source_y - y, source_x - x)
    delta = target_heading - heading
    return math.atan2(math.sin(delta), math.cos(delta))


def summarize_ticks(ticks: Sequence[dict]) -> dict:
    """Count bearing signs, wall segments and realized approach ticks."""
    first_wall = next((i + 1 for i, tick in enumerate(ticks)
                       if tick['collision']), None)
    result = {
        'ticks': len(ticks),
        'teacher_alignment': {'matching': 0, 'eligible': 0},
        'command_alignment': {'matching': 0, 'eligible': 0},
        'wall_ticks': sum(bool(tick['collision']) for tick in ticks),
        'first_wall_tick': first_wall,
        'distance_change_before_wall': sum(
            tick['distance_change'] for i, tick in enumerate(ticks, 1)
            if first_wall is None or i < first_wall),
        'distance_change_from_wall': sum(
            tick['distance_change'] for i, tick in enumerate(ticks, 1)
            if first_wall is not None and i >= first_wall),
        'approach_ticks': sum(tick['distance_change'] > 0 for tick in ticks),
        'retreat_ticks': sum(tick['distance_change'] < 0 for tick in ticks),
        'teacher_command_turn_mae': (sum(
            abs(tick['teacher_turn'] - tick['command_turn']) for tick in ticks)
            / len(ticks) if ticks else None),
        'mean_forward_command': (sum(tick['command_forward'] for tick in ticks)
                                 / len(ticks) if ticks else None),
    }
    for tick in ticks:
        bearing = tick['bearing_error']
        if abs(bearing) <= BEARING_THRESHOLD_RADIANS:
            continue
        for field, target in (('teacher_turn', 'teacher_alignment'),
                              ('command_turn', 'command_alignment')):
            turn = tick[field]
            if abs(turn) <= TURN_THRESHOLD:
                continue
            result[target]['eligible'] += 1
            result[target]['matching'] += int((bearing > 0) == (turn > 0))
    return result


def trace_episode(
    case: EpisodeCase, config: EvaluationConfig,
    controller_factory: Callable[[], NeuralController], *, ticks: int,
) -> tuple[list[dict], dict]:
    """Score each causal tick without adding coordinates to the controller."""
    if type(ticks) is not int or not 0 < ticks <= config.horizon_ticks:
        raise ValueError('trace ticks must fit the evaluation horizon')
    arena = Arena(config.arena_width, config.arena_height)
    fly = Fly(case.x, case.y, case.heading)
    arena.validate_body(fly)
    source = OdorSource(config.source_x, config.source_y,
                        config.source_strength, config.source_softening)
    loop = ClosedLoopSimulation(
        FixedStepSimulation(arena, fly, config.world_dt_seconds, source),
        controller_factory(),
    )
    rows = []
    distance = math.hypot(fly.x - source.x, fly.y - source.y)
    success = distance <= config.success_radius
    while not success and loop.world.tick_count < ticks:
        pre_x, pre_y, pre_heading = fly.x, fly.y, fly.heading
        pre_distance = distance
        error = bearing_error(pre_x, pre_y, pre_heading, source.x, source.y)
        snapshot = loop.step()
        distance = math.hypot(fly.x - source.x, fly.y - source.y)
        observation = snapshot.pre_movement_observation
        rows.append({
            'tick': snapshot.tick_id,
            'observation': {'odor_left': observation.odor_left,
                            'odor_right': observation.odor_right},
            'teacher_turn': float(teacher_command(observation)[1].item()),
            'command_forward': snapshot.command.forward,
            'command_turn': snapshot.command.turn,
            'bearing_error': error,
            'pre_distance': pre_distance,
            'post_distance': distance,
            'distance_change': pre_distance - distance,
            'collision': snapshot.collision,
            'pre_heading': pre_heading,
            'post_heading': snapshot.body.heading,
        })
        success = distance <= config.success_radius
    endpoint = {
        'ticks': loop.world.tick_count,
        'outcome': 'success' if success else 'timeout',
        'final_x': fly.x, 'final_y': fly.y,
        'final_heading': fly.heading, 'final_distance': distance,
    }
    return rows, endpoint


def trace_cases(
    cases: Sequence[EpisodeCase], config: EvaluationConfig, *, ticks: int,
    models: dict[str, TrainableMotorReadout],
) -> dict:
    """Trace fixed and learned controllers on locked train/validation cases."""
    allowed = set(locked_cases()['train'] + locked_cases()['validation'])
    if not cases or any(case not in allowed for case in cases):
        raise ValueError('trace cases must be locked train or validation cases')
    if set(models) != {'seven', 'nine'} or models['seven'].feature_count != 7 or (
            models['nine'].feature_count != 9):
        raise ValueError('trace requires seven- and nine-feature readouts')
    factories = {
        'fixed': lambda: fixed_controller(config.network_seed),
        'seven': candidate_controller_factory(models['seven'], config.network_seed),
        'nine': candidate_controller_factory(models['nine'], config.network_seed),
    }
    all_ticks = []
    case_rows = []
    for case in cases:
        for name in CONTROLLERS:
            rows, endpoint = trace_episode(case, config, factories[name], ticks=ticks)
            all_ticks.extend({'case_id': case.case_id, 'split': case.split,
                              'controller': name, **row} for row in rows)
            case_rows.append({'case_id': case.case_id, 'split': case.split,
                              'controller': name, 'endpoint': endpoint,
                              'metrics': summarize_ticks(rows)})
    summaries = {}
    for name in CONTROLLERS:
        selected = [row for row in case_rows if row['controller'] == name]
        summaries[name] = {
            'episodes': len(selected),
            'successes': sum(row['endpoint']['outcome'] == 'success'
                             for row in selected),
            'ticks': sum(row['metrics']['ticks'] for row in selected),
            'wall_ticks': sum(row['metrics']['wall_ticks'] for row in selected),
            'approach_ticks': sum(row['metrics']['approach_ticks'] for row in selected),
            'retreat_ticks': sum(row['metrics']['retreat_ticks'] for row in selected),
            'teacher_alignment': {
                key: sum(row['metrics']['teacher_alignment'][key] for row in selected)
                for key in ('matching', 'eligible')},
            'command_alignment': {
                key: sum(row['metrics']['command_alignment'][key] for row in selected)
                for key in ('matching', 'eligible')},
            'mean_teacher_command_turn_mae': sum(
                row['metrics']['teacher_command_turn_mae'] or 0.0
                for row in selected) / len(selected),
        }
    return {'cases': case_rows, 'ticks': all_ticks, 'summary': summaries}


def run_diagnostic(
    output_dir: Path, training_result_path: Path, *,
    config: EvaluationConfig | None = None,
    train_ticks: int = 200, validation_ticks: int | None = None,
) -> Path:
    """Replay FB-07H checkpoints without training or opening final cases."""
    ensure_new_run_directory(output_dir)
    config = config or EvaluationConfig()
    validation_ticks = (config.horizon_ticks if validation_ticks is None
                        else validation_ticks)
    trained_path = Path(training_result_path)
    trained = json.loads(trained_path.read_text(encoding='utf-8'))
    if (trained.get('experiment') != 'fb07h_matched_voltage_readout'
            or trained.get('config') != asdict(config)
            or trained.get('final_cases_evaluated') is not False):
        raise ValueError('FB-07H protocol or final-case seal differs')
    models = {}
    for name, size in (('seven', 7), ('nine', 9)):
        checkpoint = trained_path.parent / trained['models'][name]['checkpoint']
        if (hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                != trained['models'][name]['checkpoint_sha256']):
            raise ValueError(f'{name} checkpoint hash differs')
        models[name] = load_matched_checkpoint(
            checkpoint, network_seed=config.network_seed, feature_count=size)
    cases = locked_cases()
    traces = {
        'train': trace_cases(cases['train'], config, ticks=train_ticks, models=models),
        'validation': trace_cases(cases['validation'], config,
                                  ticks=validation_ticks, models=models),
    }
    expected = trained['results']['validation']
    replay_exact = all(
        row['endpoint']['ticks'] == prior['ticks']
        and row['endpoint']['outcome'] == prior['outcome']
        and row['endpoint']['final_x'] == prior['final_x']
        and row['endpoint']['final_y'] == prior['final_y']
        and row['endpoint']['final_heading'] == prior['final_heading']
        for name in CONTROLLERS
        for row, prior in zip(
            (entry for entry in traces['validation']['cases']
             if entry['controller'] == name), expected[name])
    )
    if not replay_exact:
        raise RuntimeError('diagnostic replay differs from FB-07H validation')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'ticks.csv').open('w', newline='', encoding='utf-8') as file:
        fields = ('split', 'case_id', 'controller', 'tick', 'odor_left', 'odor_right',
                  'teacher_turn', 'command_forward', 'command_turn',
                  'bearing_error', 'pre_distance', 'post_distance',
                  'distance_change', 'collision', 'pre_heading', 'post_heading')
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for split in ('train', 'validation'):
            for row in traces[split]['ticks']:
                observation = row['observation']
                writer.writerow({**{key: value for key, value in row.items()
                                   if key != 'observation'},
                                 **observation})
    payload = {
        'experiment': 'fb07i_embodied_failure_diagnosis',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'objective': 'locate teacher, command and wall contributors to no arrival',
        'bearing_rule': ('clockwise angle from pre-movement heading to source; '
                         'scoring only, never a controller or teacher input'),
        'alignment_thresholds': {'bearing_radians': BEARING_THRESHOLD_RADIANS,
                                 'turn_command': TURN_THRESHOLD},
        'config': asdict(config),
        'probe_ticks': {'train': train_ticks, 'validation': validation_ticks},
        'network_seed': config.network_seed,
        'network_size': 100,
        'topology_sha256': topology_sha256(config.network_seed),
        'source_training_result': str(trained_path),
        'checkpoint_sha256': {name: trained['models'][name]['checkpoint_sha256']
                              for name in ('seven', 'nine')},
        'validation_replay_exact': replay_exact,
        'cases': {split: traces[split]['cases']
                  for split in ('train', 'validation')},
        'summary': {split: traces[split]['summary']
                    for split in ('train', 'validation')},
        'tick_rows': {split: len(traces[split]['ticks'])
                      for split in ('train', 'validation')},
        'training_run': False,
        'final_cases_evaluated': False,
        'software': {'python': sys.version.split()[0], 'torch': torch.__version__,
                     'pygame': version('pygame'), 'platform': platform.platform(),
                     'device': 'cpu'},
    }
    path = output_dir / 'results.json'
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the declared scoring-only failure diagnosis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/fb07i-embodied-diagnosis'))
    parser.add_argument('--training-result', type=Path,
                        default=Path('experiments/fb07h-matched-readout/results.json'))
    args = parser.parse_args(argv)
    path = run_diagnostic(args.output_dir, args.training_result)
    result = json.loads(path.read_text(encoding='utf-8'))
    print(f'Wrote train/validation tick diagnosis: {path}')
    for name in CONTROLLERS:
        row = result['summary']['validation'][name]
        print(f"validation {name}: arrivals={row['successes']}/"
              f"{row['episodes']} wall={row['wall_ticks']}/{row['ticks']} "
              f"aligned={row['command_alignment']['matching']}/"
              f"{row['command_alignment']['eligible']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
