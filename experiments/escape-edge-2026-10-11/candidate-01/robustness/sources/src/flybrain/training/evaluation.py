"""Locked start cases and renderer-free embodied evaluation for FB-06."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence

import torch

from flybrain.closed_loop import ClosedLoopSimulation
from flybrain.neural.controller import NeuralController
from flybrain.neural.motor import MotorDecoder
from flybrain.neural.network import SyntheticNetwork
from flybrain.simulation import FixedStepSimulation
from flybrain.world.arena import Arena
from flybrain.world.fly import Fly
from flybrain.world.odor import OdorSource
from .records import ensure_new_run_directory


@dataclass(frozen=True)
class EvaluationConfig:
    """Fixed world, network, horizon and center-to-center success criterion."""

    horizon_ticks: int = 600
    success_radius: float = 20.0
    network_seed: int = 7
    world_dt_seconds: float = 0.02
    arena_width: float = 800.0
    arena_height: float = 600.0
    source_x: float = 600.0
    source_y: float = 300.0
    source_strength: float = 1000.0
    source_softening: float = 20.0

    def __post_init__(self) -> None:
        if type(self.horizon_ticks) is not int or self.horizon_ticks <= 0:
            raise ValueError('horizon_ticks must be a positive integer')
        if not math.isfinite(self.success_radius) or self.success_radius <= 0:
            raise ValueError('success_radius must be positive and finite')
        if not math.isclose(self.world_dt_seconds, 0.02, rel_tol=0, abs_tol=1e-12):
            raise ValueError('world_dt_seconds must match 20 neural substeps')


@dataclass(frozen=True)
class EpisodeCase:
    """Recorded initial pose; seed identifies its local generator stream."""

    case_id: str
    split: str
    seed: int
    x: float
    y: float
    heading: float


@dataclass(frozen=True)
class EpisodeResult:
    """One complete episode, including failures and actual wall-limited path."""

    case_id: str
    split: str
    seed: int
    outcome: str
    ticks: int
    elapsed_seconds: float
    time_to_food_seconds: float | None
    initial_distance: float
    final_distance: float
    min_distance: float
    path_length: float
    progress_fraction: float
    path_efficiency: float | None
    collision_count: int
    spike_count: int
    absolute_turn_radians: float
    final_x: float
    final_y: float
    final_heading: float


def locked_cases() -> dict[str, tuple[EpisodeCase, ...]]:
    """Return deterministic disjoint train, validation and final starts."""
    splits = {
        'train': range(8),
        'validation': range(100, 104),
        'final': range(200, 208),
    }
    cases = {}
    for split, seeds in splits.items():
        generated = []
        for seed in seeds:
            rng = random.Random(seed)
            angle = rng.uniform(0, math.tau)
            radius = math.sqrt(rng.uniform(60.0 ** 2, 180.0 ** 2))
            heading = rng.uniform(0, math.tau)
            generated.append(EpisodeCase(
                case_id=f'{split}-{seed}',
                split=split,
                seed=seed,
                x=600.0 + radius * math.cos(angle),
                y=300.0 + radius * math.sin(angle),
                heading=heading,
            ))
        cases[split] = tuple(generated)
    return cases


def fixed_controller(network_seed: int) -> NeuralController:
    """Build the unchanged hand-wired controller and decoder for one episode."""
    network = SyntheticNetwork(seed=network_seed)
    return NeuralController(network, MotorDecoder(network.topology))


def run_episode(
    case: EpisodeCase,
    config: EvaluationConfig,
    controller_factory: Callable[[], NeuralController] | None = None,
) -> EpisodeResult:
    """Measure one fresh closed loop; only this scoring layer reads coordinates."""
    arena = Arena(config.arena_width, config.arena_height)
    fly = Fly(case.x, case.y, case.heading)
    arena.validate_body(fly)
    if not fly.radius <= fly.x <= arena.width - fly.radius or not (
        fly.radius <= fly.y <= arena.height - fly.radius
    ):
        raise ValueError('episode start must fit within arena bounds')
    source = OdorSource(
        config.source_x, config.source_y,
        config.source_strength, config.source_softening,
    )
    controller = (controller_factory() if controller_factory is not None
                  else fixed_controller(config.network_seed))
    loop = ClosedLoopSimulation(
        FixedStepSimulation(arena, fly, config.world_dt_seconds, source),
        controller,
    )
    initial_distance = math.hypot(fly.x - source.x, fly.y - source.y)
    min_distance = initial_distance
    path_length = 0.0
    collisions = 0
    spikes = 0
    absolute_turn = 0.0
    success = initial_distance <= config.success_radius

    while not success and loop.world.tick_count < config.horizon_ticks:
        previous_x, previous_y = fly.x, fly.y
        tick = loop.step()
        path_length += math.hypot(fly.x - previous_x, fly.y - previous_y)
        collisions += int(tick.collision)
        spikes += int(tick.activity.spike_counts.sum().item())
        absolute_turn += abs(tick.command.turn) * math.pi * config.world_dt_seconds
        distance = math.hypot(fly.x - source.x, fly.y - source.y)
        min_distance = min(min_distance, distance)
        success = distance <= config.success_radius

    final_distance = math.hypot(fly.x - source.x, fly.y - source.y)
    elapsed = loop.world.tick_count * config.world_dt_seconds
    progress = (min(1.0, (initial_distance - final_distance)
                    / (initial_distance - config.success_radius))
                if initial_distance > config.success_radius else 1.0)
    efficiency = (
        (initial_distance - config.success_radius) / path_length
        if success and path_length > 0 and initial_distance > config.success_radius
        else None
    )
    return EpisodeResult(
        case_id=case.case_id,
        split=case.split,
        seed=case.seed,
        outcome='success' if success else 'timeout',
        ticks=loop.world.tick_count,
        elapsed_seconds=elapsed,
        time_to_food_seconds=elapsed if success else None,
        initial_distance=initial_distance,
        final_distance=final_distance,
        min_distance=min_distance,
        path_length=path_length,
        progress_fraction=progress,
        path_efficiency=efficiency,
        collision_count=collisions,
        spike_count=spikes,
        absolute_turn_radians=absolute_turn,
        final_x=fly.x,
        final_y=fly.y,
        final_heading=fly.heading,
    )


def evaluate_cases(
    cases: Sequence[EpisodeCase],
    config: EvaluationConfig,
    controller_factory: Callable[[], NeuralController] | None = None,
) -> tuple[EpisodeResult, ...]:
    """Evaluate cases independently, with a fresh controller per episode."""
    return tuple(run_episode(case, config, controller_factory) for case in cases)


def summarize(results: Sequence[EpisodeResult]) -> dict[str, float | int | None]:
    """Aggregate failures and successes without hiding the denominator."""
    if not results:
        raise ValueError('cannot summarize zero episodes')
    successes = [result for result in results if result.outcome == 'success']
    count = len(results)

    def mean(values: Sequence[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        'episodes': count,
        'successes': len(successes),
        'timeouts': count - len(successes),
        'success_rate': len(successes) / count,
        'mean_progress_fraction': mean([r.progress_fraction for r in results]),
        'mean_final_distance': mean([r.final_distance for r in results]),
        'mean_min_distance': mean([r.min_distance for r in results]),
        'mean_path_length': mean([r.path_length for r in results]),
        'mean_collision_count': mean([float(r.collision_count) for r in results]),
        'mean_spike_count': mean([float(r.spike_count) for r in results]),
        'mean_absolute_turn_radians': mean([r.absolute_turn_radians for r in results]),
        'mean_success_time_seconds': mean([
            r.time_to_food_seconds for r in successes
            if r.time_to_food_seconds is not None
        ]),
        'mean_success_path_efficiency': mean([
            r.path_efficiency for r in successes if r.path_efficiency is not None
        ]),
    }


def _git_commit() -> str | None:
    result = subprocess.run(
        ['git', 'rev-parse', '--verify', 'HEAD'],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_run(
    output_dir: Path,
    config: EvaluationConfig,
    cases: Mapping[str, Sequence[EpisodeCase]],
    results: Sequence[EpisodeResult],
) -> tuple[Path, Path]:
    """Write full run metadata/poses and flat per-episode results as JSON/CSV."""
    ensure_new_run_directory(output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        split: summarize(selected)
        for split in cases
        if (selected := [r for r in results if r.split == split])
    }
    payload = {
        'experiment': 'fb06_fixed_baseline',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'controller': 'fixed_synthetic_100',
        'objective': 'maximize held-out food-arrival rate; report all-case normalized progress',
        'training_steps': 0,
        'learning_rate': None,
        'network_size': 100,
        'checkpoint': None,
        'software': {
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'pygame': version('pygame'),
            'platform': platform.platform(),
            'device': 'cpu',
        },
        'config': asdict(config),
        'cases': {split: [asdict(case) for case in group]
                  for split, group in cases.items()},
        'summaries': summaries,
        'results': [asdict(result) for result in results],
    }
    json_path = output_dir / 'results.json'
    csv_path = output_dir / 'episodes.csv'
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    with csv_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(EpisodeResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    """Record the locked validation/final fixed baseline without a display."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/fb06-fixed-baseline'))
    args = parser.parse_args(argv)
    ensure_new_run_directory(args.output_dir)
    config = EvaluationConfig()
    cases = locked_cases()
    measured = evaluate_cases(cases['validation'] + cases['final'], config)
    json_path, csv_path = write_run(args.output_dir, config, cases, measured)
    print(f'Wrote {len(measured)} episodes: {json_path}, {csv_path}')
    for split in ('validation', 'final'):
        report = summarize([result for result in measured if result.split == split])
        print(f"{split}: success={report['successes']}/{report['episodes']} "
              f"progress={report['mean_progress_fraction']:.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
