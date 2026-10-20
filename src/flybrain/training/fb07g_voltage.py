"""FB-07G: read-only sensory voltage feature probe on locked observations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
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

from flybrain.neural.network import SyntheticNetwork
from flybrain.neural.readout import sensory_voltage_means

from .evaluation import EpisodeCase, EvaluationConfig, _git_commit, locked_cases, run_episode
from .fb07e_probe import CANDIDATE_SCALE
from .fb07f_collision import StageTrace, _make_traces
from .readout_experiment import (
    FEATURE_ALPHA, _RecordingController, topology_sha256,
)


def feature_metrics(traces: Sequence[StageTrace]) -> dict:
    """Compare exact seven-spike and nine-value hybrid feature collisions."""
    result: dict = {}
    buckets_by_stage = {}
    for stage in ('ema', 'hybrid'):
        buckets: dict[tuple[float, ...], list[StageTrace]] = {}
        for trace in traces:
            key = trace.ema if stage == 'ema' else trace.ema + trace.sensory_voltage
            if stage == 'hybrid' and len(trace.sensory_voltage) != 2:
                raise ValueError('each trace needs two sensory voltage means')
            buckets.setdefault(key, []).append(trace)
        buckets_by_stage[stage] = buckets
        error = 0.0
        aliases = 0
        for items in buckets.values():
            if len({item.raw for item in items}) > 1:
                aliases += len(items)
            mean_turn = sum(item.turn for item in items) / len(items)
            error += sum((item.turn - mean_turn) ** 2 for item in items)
        result[stage] = {
            'ticks': len(traces), 'unique_vectors': len(buckets),
            'aliased_ticks': aliases,
            'turn_mse_floor': error / max(len(traces), 1),
        }
    conflicting = resolved = 0
    for items in buckets_by_stage['ema'].values():
        for index, left in enumerate(items):
            for right in items[index + 1:]:
                if left.raw == right.raw or left.turn == right.turn:
                    continue
                conflicting += 1
                if left.sensory_voltage != right.sensory_voltage:
                    resolved += 1
    result['conflicting_ema_pairs'] = conflicting
    result['pairs_resolved_by_voltage'] = resolved
    result['resolution_fraction'] = resolved / conflicting if conflicting else 0.0
    return result


def trace_cases(cases: Sequence[EpisodeCase], config: EvaluationConfig, *,
                ticks: int) -> dict:
    """Replay locked cases through the fixed rational encoder with reset state."""
    allowed = set(locked_cases()['train'] + locked_cases()['validation'])
    if not cases or any(case not in allowed for case in cases):
        raise ValueError('trace cases must be locked train or validation cases')
    if type(ticks) is not int or not 0 < ticks <= config.horizon_ticks:
        raise ValueError('trace ticks must fit the evaluation horizon')
    sampled = replace(config, horizon_ticks=ticks)
    all_traces: list[StageTrace] = []
    case_rows = []
    topology_hash = topology_sha256(config.network_seed)
    topology_unchanged = True
    for case in cases:
        recorder = _RecordingController(config.network_seed)
        run_episode(case, sampled, lambda: recorder)
        observations = [observation for observation, _ in recorder.records]
        network = SyntheticNetwork(seed=config.network_seed,
                                   encoder_scale=CANDIDATE_SCALE,
                                   encoder_mode='rational')
        topology_unchanged &= (hashlib.sha256(
            network.topology.weights.numpy().tobytes()).hexdigest() == topology_hash)
        activities = [network.advance(observation) for observation in observations]
        traces = _make_traces(observations, activities, network)
        # Check the public voltage extraction contract against the trace data.
        if any(tuple(sensory_voltage_means(activity, network.topology).tolist())
               != trace.sensory_voltage for activity, trace in zip(activities, traces)):
            raise RuntimeError('voltage trace differs from neural snapshots')
        case_rows.append({'case_id': case.case_id, 'split': case.split,
                          'ticks': len(traces), 'features': feature_metrics(traces)})
        all_traces.extend(traces)
    return {'cases': case_rows, 'features': feature_metrics(all_traces),
            'feature_size': 9, 'topology_unchanged': bool(topology_unchanged)}


def decide_gate(probe: dict) -> dict:
    """Require more discrimination and lower turn floor on both splits."""
    decisions = {}
    for split in ('train', 'validation'):
        features = probe[split]['features']
        decisions[split] = {
            'alias_improved': (features['hybrid']['aliased_ticks']
                               < features['ema']['aliased_ticks']),
            'turn_floor_improved': (features['hybrid']['turn_mse_floor']
                                    < features['ema']['turn_mse_floor']),
            'collision_pairs_resolved': features['pairs_resolved_by_voltage'] > 0,
        }
    return {**decisions, 'passed': all(all(row.values())
                                      for row in decisions.values())}


def run_diagnostic(output_dir: Path, *, config: EvaluationConfig | None = None,
                   train_ticks: int = 200,
                   validation_ticks: int | None = None) -> Path:
    """Record one read-only train/validation comparison without learning."""
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
        fields = ('split', 'case_id', 'ticks', 'ema_aliased_ticks',
                  'hybrid_aliased_ticks', 'ema_turn_mse_floor',
                  'hybrid_turn_mse_floor', 'conflicting_ema_pairs',
                  'pairs_resolved_by_voltage')
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for split in ('train', 'validation'):
            for case in probe[split]['cases']:
                measured = case['features']
                writer.writerow({
                    'split': split, 'case_id': case['case_id'], 'ticks': case['ticks'],
                    'ema_aliased_ticks': measured['ema']['aliased_ticks'],
                    'hybrid_aliased_ticks': measured['hybrid']['aliased_ticks'],
                    'ema_turn_mse_floor': measured['ema']['turn_mse_floor'],
                    'hybrid_turn_mse_floor': measured['hybrid']['turn_mse_floor'],
                    'conflicting_ema_pairs': measured['conflicting_ema_pairs'],
                    'pairs_resolved_by_voltage': measured['pairs_resolved_by_voltage'],
                })
    payload = {
        'experiment': 'fb07g_voltage_features',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'objective': 'probe subthreshold sensory voltage as readout information',
        'feature_rule': ('seven per-group 20 ms spike-count EMA values followed by '
                         'post-interval mean membrane voltage of ten left and ten '
                         'right sensory neurons'),
        'encoder_rule': '2.5*c/(c+0.005)',
        'gate_rule': ('hybrid alias ticks and turn MSE floor strictly lower than '
                      'seven EMA features, with at least one conflicting pair '
                      'resolved, separately on train and validation'),
        'gate': decide_gate(probe),
        'config': asdict(config),
        'probe_ticks': {'train': train_ticks, 'validation': validation_ticks},
        'network_seed': config.network_seed,
        'network_size': 100,
        'topology_sha256': topology_sha256(config.network_seed),
        'topology_unchanged': True,
        'feature_alpha': FEATURE_ALPHA,
        'feature_size': 9,
        'cases': {split: [asdict(case) for case in cases[split]]
                  for split in ('train', 'validation')},
        'probe': probe,
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
    """Run the declared voltage probe from the project root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/fb07g-voltage-features'))
    args = parser.parse_args(argv)
    path = run_diagnostic(args.output_dir)
    result = json.loads(path.read_text(encoding='utf-8'))
    print(f'Wrote train/validation voltage probe: {path}')
    for split in ('train', 'validation'):
        measured = result['probe'][split]['features']
        print(f"{split}: EMA aliases={measured['ema']['aliased_ticks']} "
              f"hybrid aliases={measured['hybrid']['aliased_ticks']} "
              f"resolved pairs={measured['pairs_resolved_by_voltage']}/"
              f"{measured['conflicting_ema_pairs']}")
    print(f"Gate passed: {result['gate']['passed']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
