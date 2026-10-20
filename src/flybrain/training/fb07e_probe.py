"""FB-07E: replay one rational sensory encoder on locked observations."""

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
from torch import Tensor

from flybrain.neural.network import NeuralActivity, SyntheticNetwork
from flybrain.neural.readout import SpikeFeatureFilter
from flybrain.world.sensing import SensoryObservation

from .evaluation import EpisodeCase, EvaluationConfig, _git_commit, locked_cases, run_episode
from .fb07d_diagnosis import ORIGINAL_SCALE, WIDER_SCALE, feature_aliasing
from .readout_experiment import FEATURE_ALPHA, _RecordingController, topology_sha256


CANDIDATE_SCALE = 0.005
WEAK_CEILING = 0.025
ENCODERS = ('original', 'wider', 'candidate')


def controlled_weak_probe(network_seed: int) -> dict:
    """Record first-tick currents and spikes for two reversed weak inputs."""
    observations = (SensoryObservation(0.007, 0.012),
                    SensoryObservation(0.012, 0.007))
    result = {'observations': [[o.odor_left, o.odor_right]
                               for o in observations]}
    for name, scale, mode in (
        ('original', ORIGINAL_SCALE, 'linear'),
        ('wider', WIDER_SCALE, 'linear'),
        ('candidate', CANDIDATE_SCALE, 'rational'),
    ):
        currents = []
        spikes = []
        for observation in observations:
            network = SyntheticNetwork(seed=network_seed, encoder_scale=scale,
                                       encoder_mode=mode)
            current = network.encode(observation)
            currents.append([float(current[0].item()), float(current[10].item())])
            counts = network.advance(observation).spike_counts
            spikes.append([int(counts[:10].sum().item()),
                           int(counts[10:20].sum().item())])
        result[name] = {'currents': currents, 'sensory_spike_totals': spikes}
    return result


def _metrics(
    observations: Sequence[SensoryObservation], activities: Sequence[NeuralActivity],
    features: Tensor, *, encoder_scale: float | None,
) -> dict[str, int | float]:
    """Combine readout aliasing with sensory spikes at weak positive readings."""
    result = feature_aliasing(observations, features, encoder_scale=encoder_scale)
    weak = 0
    active = 0
    for observation, activity in zip(observations, activities):
        for concentration, indices in (
            (observation.odor_left, slice(0, 10)),
            (observation.odor_right, slice(10, 20)),
        ):
            if 0 < concentration < WEAK_CEILING:
                weak += 1
                active += int(activity.spike_counts[indices].sum().item() > 0)
    result['weak_readings'] = weak
    result['weak_spiking_readings'] = active
    return result


def probe_cases(
    cases: Sequence[EpisodeCase], config: EvaluationConfig, *, ticks: int,
) -> dict:
    """Replay identical original-controller observations in three fresh graphs."""
    allowed = set(locked_cases()['train'] + locked_cases()['validation'])
    if not cases or any(case not in allowed for case in cases):
        raise ValueError('probe cases must be locked train or validation cases')
    if type(ticks) is not int or not 0 < ticks <= config.horizon_ticks:
        raise ValueError('probe ticks must fit the evaluation horizon')
    sampled_config = replace(config, horizon_ticks=ticks)
    all_observations: list[SensoryObservation] = []
    all_activities: dict[str, list[NeuralActivity]] = {name: [] for name in ENCODERS}
    all_features: dict[str, list[Tensor]] = {name: [] for name in ENCODERS}
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
            feature_filter = SpikeFeatureFilter(networks[name].topology, FEATURE_ALPHA)
            features = torch.stack([feature_filter.update(activity)
                                    for activity in activities[name]])
            scale = (ORIGINAL_SCALE if name == 'original' else
                     WIDER_SCALE if name == 'wider' else None)
            case_row[name] = _metrics(observations, activities[name], features,
                                      encoder_scale=scale)
            all_activities[name].extend(activities[name])
            all_features[name].append(features)
        case_rows.append(case_row)
        all_observations.extend(observations)
    result = {'cases': case_rows, 'topology_unchanged': bool(topology_unchanged)}
    for name in ENCODERS:
        scale = (ORIGINAL_SCALE if name == 'original' else
                 WIDER_SCALE if name == 'wider' else None)
        result[name] = _metrics(all_observations, all_activities[name],
                                torch.cat(all_features[name]), encoder_scale=scale)
    return result


def decide_gate(probes: dict[str, dict]) -> dict:
    """Apply the predeclared split-wise information gate, without training."""
    decisions = {}
    for split in ('train', 'validation'):
        rows = probes[split]
        candidate = rows['candidate']
        prior = [rows['original'], rows['wider']]
        decisions[split] = {
            'alias_improved': candidate['aliased_ticks'] < min(
                row['aliased_ticks'] for row in prior),
            'turn_floor_not_worse': candidate['turn_mse_floor'] <= min(
                row['turn_mse_floor'] for row in prior),
            'weak_activity_improved': candidate['weak_spiking_readings'] > max(
                row['weak_spiking_readings'] for row in prior),
        }
    return {
        'train': decisions['train'],
        'validation': decisions['validation'],
        'passed': all(all(flags.values()) for flags in decisions.values()),
    }


def run_diagnostic(
    output_dir: Path, *, config: EvaluationConfig | None = None,
    train_ticks: int = 200, validation_ticks: int | None = None,
) -> Path:
    """Record one train/validation probe; never optimize or evaluate final."""
    ensure_new_run_directory(output_dir)
    config = config or EvaluationConfig()
    validation_ticks = (config.horizon_ticks if validation_ticks is None
                        else validation_ticks)
    cases = locked_cases()
    probe = {
        'train': probe_cases(cases['train'], config, ticks=train_ticks),
        'validation': probe_cases(cases['validation'], config,
                                  ticks=validation_ticks),
    }
    if not all(probe[split]['topology_unchanged'] for split in probe):
        raise RuntimeError('fixed topology changed')
    gate = decide_gate(probe)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'cases.csv').open('w', newline='', encoding='utf-8') as file:
        fields = ('split', 'case_id', 'encoder', 'ticks', 'unique_raw_pairs',
                  'unique_feature_vectors', 'aliased_ticks', 'turn_mse_floor',
                  'weak_readings', 'weak_spiking_readings')
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for split in ('train', 'validation'):
            for case in probe[split]['cases']:
                for name in ENCODERS:
                    metrics = case[name]
                    writer.writerow({
                        'split': split, 'case_id': case['case_id'],
                        'encoder': name, 'ticks': case['ticks'],
                        **{key: metrics[key] for key in fields[4:]},
                    })
    payload = {
        'experiment': 'fb07e_weak_signal_probe',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'objective': 'weak-input spike and readout feature discriminability gate',
        'encoder_rules': {
            'original': '2*min(c/0.025,1)',
            'wider': '2*min(c/0.05,1)',
            'candidate': '2.5*c/(c+0.005)',
        },
        'controlled_weak_probe': controlled_weak_probe(config.network_seed),
        'weak_input_range': '0 < concentration < 0.025',
        'gate_rule': ('candidate alias ticks below both prior scales, turn MSE floor '
                      'no worse than both, and weak spiking readings above both; '
                      'each criterion must hold on train and validation'),
        'gate': gate,
        'network_seed': config.network_seed,
        'network_size': 100,
        'topology_sha256': topology_sha256(config.network_seed),
        'topology_unchanged': True,
        'feature_alpha': FEATURE_ALPHA,
        'config': asdict(config),
        'probe_ticks': {'train': train_ticks, 'validation': validation_ticks},
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
    """Run the declared diagnostic from the project root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/fb07e-weak-signal'))
    args = parser.parse_args(argv)
    path = run_diagnostic(args.output_dir)
    payload = json.loads(path.read_text(encoding='utf-8'))
    print(f'Wrote train/validation probe: {path}')
    for split in ('train', 'validation'):
        for name in ENCODERS:
            row = payload['probe'][split][name]
            print(f"{split} {name}: alias={row['aliased_ticks']}/{row['ticks']} "
                  f"weak_spikes={row['weak_spiking_readings']}/{row['weak_readings']}")
    print(f"Gate passed: {payload['gate']['passed']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
