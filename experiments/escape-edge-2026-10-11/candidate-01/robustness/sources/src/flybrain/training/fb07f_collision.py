"""FB-07F: locate exact feature collisions along the fixed sensory path."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from .records import ensure_new_run_directory
import platform
import sys
from collections.abc import Sequence

import torch

from flybrain.neural.network import NeuralActivity, SyntheticNetwork
from flybrain.neural.readout import GROUP_NAMES, SpikeFeatureFilter
from flybrain.world.sensing import SensoryObservation

from .evaluation import EpisodeCase, EvaluationConfig, _git_commit, locked_cases, run_episode
from .fb07d_diagnosis import ORIGINAL_SCALE, WIDER_SCALE
from .fb07e_probe import CANDIDATE_SCALE, ENCODERS
from .readout_experiment import (
    FEATURE_ALPHA, _RecordingController, teacher_command, topology_sha256,
)


STAGES = ('current', 'sensory_counts', 'all_counts', 'group_means', 'ema')
PAIR_CATEGORIES = (
    'raw_to_current', 'current_to_spikes',
    'sensory_spikes_to_group_means', 'other_spikes_to_group_means',
    'group_means_to_ema',
)


@dataclass(frozen=True)
class StageTrace:
    """One tick's readout-relevant values, with original raw sensory input."""

    raw: tuple[float, float]
    turn: float
    current: tuple[float, ...]
    sensory_counts: tuple[int, ...]
    all_counts: tuple[int, ...]
    group_means: tuple[float, ...]
    ema: tuple[float, ...]
    sensory_voltage: tuple[float, float] = ()


def stage_metrics(traces: Sequence[StageTrace]) -> dict[str, dict[str, int | float]]:
    """Measure exact collisions and irreducible turn error at every stage."""
    result = {}
    for stage in STAGES:
        buckets: dict[tuple, list[StageTrace]] = {}
        for trace in traces:
            buckets.setdefault(getattr(trace, stage), []).append(trace)
        aliases = [items for items in buckets.values()
                   if len({item.raw for item in items}) > 1]
        error = 0.0
        for items in buckets.values():
            mean_turn = sum(item.turn for item in items) / len(items)
            error += sum((item.turn - mean_turn) ** 2 for item in items)
        result[stage] = {
            'ticks': len(traces),
            'unique_vectors': len(buckets),
            'aliased_ticks': sum(len(items) for items in aliases),
            'turn_mse_floor': error / max(len(traces), 1),
        }
    return result


def pair_attribution(traces: Sequence[StageTrace]) -> dict[str, int]:
    """Classify conflicting final-feature pairs by latest prior difference.

    Values are tick pairs, not distinct observations. Temporal network state
    means this label identifies a differing snapshot, not a causal transition.
    """
    buckets: dict[tuple[float, ...], list[StageTrace]] = {}
    for trace in traces:
        buckets.setdefault(trace.ema, []).append(trace)
    counts = {'conflicting_pairs': 0, **{name: 0 for name in PAIR_CATEGORIES}}
    for items in buckets.values():
        for index, left in enumerate(items):
            for right in items[index + 1:]:
                if left.raw == right.raw or left.turn == right.turn:
                    continue
                counts['conflicting_pairs'] += 1
                if left.group_means != right.group_means:
                    category = 'group_means_to_ema'
                elif left.all_counts != right.all_counts:
                    category = ('sensory_spikes_to_group_means'
                                if left.sensory_counts != right.sensory_counts
                                else 'other_spikes_to_group_means')
                elif left.current != right.current:
                    category = 'current_to_spikes'
                else:
                    category = 'raw_to_current'
                counts[category] += 1
    return counts


def _make_traces(
    observations: Sequence[SensoryObservation],
    activities: Sequence[NeuralActivity], network: SyntheticNetwork,
) -> list[StageTrace]:
    """Read one network's interval activities with a fresh feature EMA."""
    group_indices = [network.topology.indices(name) for name in GROUP_NAMES]
    sensory_indices = (network.topology.indices('sensory_left')
                       + network.topology.indices('sensory_right'))
    left_indices = list(network.topology.indices('sensory_left'))
    right_indices = list(network.topology.indices('sensory_right'))
    feature_filter = SpikeFeatureFilter(network.topology, FEATURE_ALPHA)
    traces = []
    for observation, activity in zip(observations, activities):
        current = network.encode(observation)
        counts = activity.spike_counts
        means = tuple(float(counts[list(indices)].to(torch.float32).mean().item())
                      for indices in group_indices)
        traces.append(StageTrace(
            raw=(observation.odor_left, observation.odor_right),
            turn=float(teacher_command(observation)[1].item()),
            current=(float(current[sensory_indices[0]].item()),
                     float(current[sensory_indices[10]].item())),
            sensory_counts=tuple(int(counts[index].item())
                                 for index in sensory_indices),
            all_counts=tuple(int(value) for value in counts.tolist()),
            group_means=means,
            ema=tuple(float(value) for value in feature_filter.update(activity).tolist()),
            sensory_voltage=(
                float(activity.snapshot.voltage[left_indices].mean().item()),
                float(activity.snapshot.voltage[right_indices].mean().item()),
            ),
        ))
    return traces


def trace_cases(
    cases: Sequence[EpisodeCase], config: EvaluationConfig, *, ticks: int,
) -> dict:
    """Replay locked train/validation observations through unchanged graphs."""
    allowed = set(locked_cases()['train'] + locked_cases()['validation'])
    if not cases or any(case not in allowed for case in cases):
        raise ValueError('trace cases must be locked train or validation cases')
    if type(ticks) is not int or not 0 < ticks <= config.horizon_ticks:
        raise ValueError('trace ticks must fit the evaluation horizon')
    sampled_config = replace(config, horizon_ticks=ticks)
    all_traces: dict[str, list[StageTrace]] = {name: [] for name in ENCODERS}
    case_rows = []
    topology_hash = topology_sha256(config.network_seed)
    topology_unchanged = True
    for case in cases:
        recorder = _RecordingController(config.network_seed, ORIGINAL_SCALE)
        run_episode(case, sampled_config, lambda: recorder)
        observations = [observation for observation, _ in recorder.records]
        networks = {
            'original': recorder.network,
            'wider': SyntheticNetwork(seed=config.network_seed,
                                     encoder_scale=WIDER_SCALE),
            'candidate': SyntheticNetwork(seed=config.network_seed,
                                          encoder_scale=CANDIDATE_SCALE,
                                          encoder_mode='rational'),
        }
        topology_unchanged &= all(
            hashlib.sha256(network.topology.weights.numpy().tobytes()).hexdigest()
            == topology_hash for network in networks.values()
        )
        activities = {
            'original': [activity for _, activity in recorder.records],
            'wider': [networks['wider'].advance(observation)
                      for observation in observations],
            'candidate': [networks['candidate'].advance(observation)
                          for observation in observations],
        }
        case_row = {'case_id': case.case_id, 'split': case.split,
                    'ticks': len(observations)}
        for name in ENCODERS:
            traces = _make_traces(observations, activities[name], networks[name])
            case_row[name] = {
                'stages': stage_metrics(traces),
                'final_pair_attribution': pair_attribution(traces),
            }
            all_traces[name].extend(traces)
        case_rows.append(case_row)
    result = {'cases': case_rows, 'topology_unchanged': bool(topology_unchanged)}
    for name in ENCODERS:
        result[name] = {
            'stages': stage_metrics(all_traces[name]),
            'final_pair_attribution': pair_attribution(all_traces[name]),
        }
    return result


def run_diagnostic(
    output_dir: Path, *, config: EvaluationConfig | None = None,
    train_ticks: int = 200, validation_ticks: int | None = None,
) -> Path:
    """Record train/validation stage measurements, without model changes."""
    ensure_new_run_directory(output_dir)
    config = config or EvaluationConfig()
    validation_ticks = (config.horizon_ticks if validation_ticks is None
                        else validation_ticks)
    cases = locked_cases()
    probe = {
        'train': trace_cases(cases['train'], config, ticks=train_ticks),
        'validation': trace_cases(cases['validation'], config,
                                  ticks=validation_ticks),
    }
    if not all(probe[split]['topology_unchanged'] for split in probe):
        raise RuntimeError('fixed topology changed')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'cases.csv').open('w', newline='', encoding='utf-8') as file:
        fields = ('split', 'case_id', 'encoder', 'ticks',
                  *[f'{stage}_aliased_ticks' for stage in STAGES],
                  'ema_turn_mse_floor', 'conflicting_pairs', *PAIR_CATEGORIES)
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for split in ('train', 'validation'):
            for case in probe[split]['cases']:
                for name in ENCODERS:
                    measured = case[name]
                    writer.writerow({
                        'split': split, 'case_id': case['case_id'],
                        'encoder': name, 'ticks': case['ticks'],
                        **{f'{stage}_aliased_ticks': measured['stages'][stage]['aliased_ticks']
                           for stage in STAGES},
                        'ema_turn_mse_floor': measured['stages']['ema']['turn_mse_floor'],
                        **measured['final_pair_attribution'],
                    })
    payload = {
        'experiment': 'fb07f_collision_stages',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'objective': 'locate remaining exact readout-feature collisions',
        'stage_order': STAGES,
        'pair_categories': PAIR_CATEGORIES,
        'pair_rule': ('pairs of ticks with distinct raw readings and teacher turns '
                      'but identical EMA features; label the latest earlier '
                      'representation that differs; this is not causal proof'),
        'encoder_rules': {
            'original': '2*min(c/0.025,1)',
            'wider': '2*min(c/0.05,1)',
            'candidate': '2.5*c/(c+0.005)',
        },
        'config': asdict(config),
        'probe_ticks': {'train': train_ticks, 'validation': validation_ticks},
        'network_seed': config.network_seed,
        'network_size': 100,
        'topology_sha256': topology_sha256(config.network_seed),
        'topology_unchanged': True,
        'feature_alpha': FEATURE_ALPHA,
        'cases': {split: [asdict(case) for case in cases[split]]
                  for split in ('train', 'validation')},
        'probe': probe,
        'training_run': False,
        'final_cases_evaluated': False,
        'software': {
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'pygame': version('pygame'),
            'platform': platform.platform(),
            'device': 'cpu',
        },
    }
    result_path = output_dir / 'results.json'
    result_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')
    return result_path


def main(argv: list[str] | None = None) -> int:
    """Run the declared collision-stage diagnosis from the project root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/fb07f-collision-stages'))
    args = parser.parse_args(argv)
    path = run_diagnostic(args.output_dir)
    payload = json.loads(path.read_text(encoding='utf-8'))
    print(f'Wrote train/validation collision stages: {path}')
    for split in ('train', 'validation'):
        for name in ENCODERS:
            stages = payload['probe'][split][name]['stages']
            print(f"{split} {name}: current={stages['current']['aliased_ticks']} "
                  f"sensory={stages['sensory_counts']['aliased_ticks']} "
                  f"group={stages['group_means']['aliased_ticks']} "
                  f"EMA={stages['ema']['aliased_ticks']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
