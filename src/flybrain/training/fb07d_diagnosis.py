"""FB-07D: paired spike-feature probe and one wider-encoder readout run."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from .records import ensure_new_run_directory
import platform
import sys
from collections.abc import Sequence

import torch
from torch import Tensor

from flybrain.neural.network import SyntheticNetwork
from flybrain.neural.readout import SpikeFeatureFilter
from flybrain.world.sensing import SensoryObservation

from .evaluation import (
    EpisodeCase, EpisodeResult, EvaluationConfig, _git_commit, evaluate_cases,
    locked_cases, run_episode, summarize,
)
from .readout_experiment import (
    FEATURE_ALPHA, LEARNING_RATE, TRAINING_STEPS, TRAIN_SAMPLE_TICKS,
    _RecordingController, collect_training_samples, learned_controller_factory,
    load_checkpoint, save_checkpoint, teacher_command, topology_sha256,
    train_readout,
)


ORIGINAL_SCALE = 0.025
WIDER_SCALE = 0.05


def controlled_probe(network_seed: int) -> dict:
    """Show whether reversed, saturated antenna readings alter first-tick spikes."""
    observations = (SensoryObservation(0.03, 0.04),
                    SensoryObservation(0.04, 0.03))
    result: dict = {'observations': [[o.odor_left, o.odor_right]
                                    for o in observations]}
    for name, scale in (('original', ORIGINAL_SCALE), ('wider', WIDER_SCALE)):
        currents = []
        spikes = []
        for observation in observations:
            network = SyntheticNetwork(seed=network_seed, encoder_scale=scale)
            current = network.encode(observation)
            left = network.topology.indices('sensory_left')
            right = network.topology.indices('sensory_right')
            currents.append([float(current[left[0]].item()),
                             float(current[right[0]].item())])
            counts = network.advance(observation).spike_counts
            spikes.append([int(counts[list(left)].sum().item()),
                           int(counts[list(right)].sum().item())])
        result[name] = {'currents': currents, 'sensory_spike_totals': spikes}
    return result


def feature_aliasing(
    observations: Sequence[SensoryObservation], features: Tensor, *,
    encoder_scale: float | None,
) -> dict[str, int | float]:
    """Count exact readout collisions among distinct sensory pairs.

    A collision is an identical seven-feature vector for at least two distinct
    raw bilateral observations. Counts include all ticks in such buckets.
    Teacher-turn conflict asks whether one vector needs different target turns.
    """
    if len(observations) != features.shape[0]:
        raise ValueError('observation and feature lengths differ')
    if (features.ndim != 2 or features.shape[1] != 7
            or features.device.type != 'cpu' or not bool(torch.isfinite(features).all())):
        raise ValueError('features must be finite CPU [N,7]')
    if encoder_scale is not None and (not math.isfinite(encoder_scale)
                                      or encoder_scale <= 0):
        raise ValueError('encoder scale must be positive and finite')
    buckets: dict[tuple[float, ...], list[tuple[tuple[float, float], float]]] = {}
    for observation, feature in zip(observations, features):
        raw = (observation.odor_left, observation.odor_right)
        turn = float(teacher_command(observation)[1].item())
        buckets.setdefault(tuple(feature.tolist()), []).append((raw, turn))
    aliased = [items for items in buckets.values()
               if len({raw for raw, _ in items}) > 1]
    conflicting = [items for items in buckets.values()
                   if len({turn for _, turn in items}) > 1]
    turn_error = 0.0
    for items in buckets.values():
        mean_turn = sum(turn for _, turn in items) / len(items)
        turn_error += sum((turn - mean_turn) ** 2 for _, turn in items)
    result = {
        'ticks': len(observations),
        'unique_raw_pairs': len({(o.odor_left, o.odor_right) for o in observations}),
        'unique_feature_vectors': len(buckets),
        'aliased_feature_vectors': len(aliased),
        'aliased_ticks': sum(len(items) for items in aliased),
        'conflicting_turn_feature_vectors': len(conflicting),
        'conflicting_turn_ticks': sum(len(items) for items in conflicting),
        'turn_mse_floor': turn_error / max(len(observations), 1),
    }
    if encoder_scale is not None:
        result.update({
            'left_saturated_ticks': sum(o.odor_left >= encoder_scale
                                        for o in observations),
            'right_saturated_ticks': sum(o.odor_right >= encoder_scale
                                         for o in observations),
            'both_saturated_ticks': sum(o.odor_left >= encoder_scale
                                        and o.odor_right >= encoder_scale
                                        for o in observations),
        })
    return result


def paired_probe(
    cases: Sequence[EpisodeCase], config: EvaluationConfig, *, ticks: int,
) -> dict:
    """Replay the same fixed-controller observations through both encoders."""
    allowed = set(locked_cases()['train'] + locked_cases()['validation'])
    if not cases or any(case not in allowed for case in cases):
        raise ValueError('probe cases must be locked train or validation cases')
    if type(ticks) is not int or not 0 < ticks <= config.horizon_ticks:
        raise ValueError('probe ticks must fit the evaluation horizon')
    sampled_config = replace(config, horizon_ticks=ticks)
    all_observations: list[SensoryObservation] = []
    all_original: list[Tensor] = []
    all_wider: list[Tensor] = []
    case_rows = []
    for case in cases:
        recorder = _RecordingController(config.network_seed, ORIGINAL_SCALE)
        run_episode(case, sampled_config, lambda: recorder)
        original_filter = SpikeFeatureFilter(recorder.network.topology, FEATURE_ALPHA)
        wider_network = SyntheticNetwork(seed=config.network_seed,
                                         encoder_scale=WIDER_SCALE)
        wider_filter = SpikeFeatureFilter(wider_network.topology, FEATURE_ALPHA)
        observations = [observation for observation, _ in recorder.records]
        original = torch.stack([original_filter.update(activity)
                                for _, activity in recorder.records])
        wider = torch.stack([wider_filter.update(wider_network.advance(observation))
                             for observation in observations])
        case_rows.append({
            'case_id': case.case_id,
            'ticks': len(observations),
            'original': feature_aliasing(observations, original,
                                         encoder_scale=ORIGINAL_SCALE),
            'wider': feature_aliasing(observations, wider,
                                      encoder_scale=WIDER_SCALE),
        })
        all_observations.extend(observations)
        all_original.append(original)
        all_wider.append(wider)
    return {
        'cases': case_rows,
        'original': feature_aliasing(all_observations, torch.cat(all_original),
                                     encoder_scale=ORIGINAL_SCALE),
        'wider': feature_aliasing(all_observations, torch.cat(all_wider),
                                  encoder_scale=WIDER_SCALE),
        'topology_unchanged': (
            topology_sha256(config.network_seed)
            == hashlib.sha256(wider_network.topology.weights.numpy().tobytes()).hexdigest()
        ),
    }


def _comparison_rows(config: EvaluationConfig) -> tuple[list[EpisodeResult], list[EpisodeResult]]:
    """Use the recorded FB-07/FB-06 rows when their full protocol matches."""
    cases = locked_cases()['validation']
    fb07_path = Path('experiments/fb07-readout/results.json')
    baseline_path = Path('experiments/fb06-fixed-baseline/results.json')
    if config == EvaluationConfig() and fb07_path.is_file() and baseline_path.is_file():
        prior = json.loads(fb07_path.read_text(encoding='utf-8'))
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        expected_ids = [case.case_id for case in cases]
        if (prior.get('config') != asdict(config) or baseline.get('config') != asdict(config)
                or [row['case_id'] for row in prior['results'] if row['split'] == 'validation']
                != expected_ids
                or [row['case_id'] for row in baseline['results']
                    if row['split'] == 'validation'] != expected_ids):
            raise ValueError('recorded comparison protocol differs')
        return (
            [EpisodeResult(**row) for row in prior['results']
             if row['split'] == 'validation'],
            [EpisodeResult(**row) for row in baseline['results']
             if row['split'] == 'validation'],
        )
    original = load_checkpoint(fb07_path.parent / 'readout.pt',
                               network_seed=config.network_seed)
    return (
        evaluate_cases(cases, config,
                       learned_controller_factory(original, config.network_seed)),
        evaluate_cases(cases, config),
    )


def run_diagnosis(
    output_dir: Path, *, config: EvaluationConfig | None = None,
    probe_ticks: int | None = None,
    sample_ticks: int = TRAIN_SAMPLE_TICKS,
    training_steps: int = TRAINING_STEPS,
    learning_rate: float = LEARNING_RATE,
) -> Path:
    """Record paired train/validation probe and one validation-only readout run."""
    ensure_new_run_directory(output_dir)
    config = config or EvaluationConfig()
    cases = locked_cases()
    probe_ticks = config.horizon_ticks if probe_ticks is None else probe_ticks
    train_probe = paired_probe(cases['train'], config,
                               ticks=min(probe_ticks, sample_ticks))
    validation_probe = paired_probe(cases['validation'], config, ticks=probe_ticks)
    topology_before = topology_sha256(config.network_seed)
    dataset = collect_training_samples(cases['train'], config, sample_ticks,
                                       encoder_scale=WIDER_SCALE)
    outcome = train_readout(dataset.features, dataset.targets,
                            steps=training_steps, learning_rate=learning_rate)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / 'readout.pt'
    save_checkpoint(checkpoint, outcome.model, network_seed=config.network_seed,
                    alpha=FEATURE_ALPHA, encoder_scale=WIDER_SCALE)
    reloaded = load_checkpoint(checkpoint, network_seed=config.network_seed,
                               encoder_scale=WIDER_SCALE)
    validation_cases = cases['validation']
    wider_rows = evaluate_cases(
        validation_cases, config,
        learned_controller_factory(reloaded, config.network_seed, WIDER_SCALE),
    )
    replay = run_episode(
        validation_cases[0], config,
        learned_controller_factory(outcome.model, config.network_seed, WIDER_SCALE),
    )
    reload_exact = replay == wider_rows[0]
    if not reload_exact:
        raise RuntimeError('reloaded readout did not replay validation episode')
    original_rows, fixed_rows = _comparison_rows(config)
    topology_unchanged = (topology_before == topology_sha256(config.network_seed)
                          and train_probe['topology_unchanged']
                          and validation_probe['topology_unchanged'])
    if not topology_unchanged:
        raise RuntimeError('fixed topology changed')
    with (output_dir / 'validation.csv').open('w', newline='', encoding='utf-8') as file:
        fields = ('case_id', 'wider_outcome', 'original_outcome', 'fixed_outcome',
                  'wider_progress_fraction', 'original_progress_fraction',
                  'fixed_progress_fraction', 'wider_final_distance',
                  'original_final_distance', 'fixed_final_distance')
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for wider, original, fixed in zip(wider_rows, original_rows, fixed_rows):
            if wider.case_id != original.case_id or wider.case_id != fixed.case_id:
                raise RuntimeError('comparison case IDs differ')
            writer.writerow({
                'case_id': wider.case_id,
                'wider_outcome': wider.outcome,
                'original_outcome': original.outcome,
                'fixed_outcome': fixed.outcome,
                'wider_progress_fraction': wider.progress_fraction,
                'original_progress_fraction': original.progress_fraction,
                'fixed_progress_fraction': fixed.progress_fraction,
                'wider_final_distance': wider.final_distance,
                'original_final_distance': original.final_distance,
                'fixed_final_distance': fixed.final_distance,
            })
    with (output_dir / 'curve.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=('step', 'loss'))
        writer.writeheader()
        writer.writerows(outcome.curve)
    payload = {
        'experiment': 'fb07d_encoder_observability',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'objective': 'measure feature aliasing; compare one wider encoder on validation arrival',
        'encoder_scales': {'original': ORIGINAL_SCALE, 'wider': WIDER_SCALE},
        'controlled_probe': controlled_probe(config.network_seed),
        'probe': {'train': train_probe, 'validation': validation_probe},
        'config': asdict(config),
        'cases': {split: [asdict(case) for case in group]
                  for split, group in cases.items() if split != 'final'},
        'train_sample_count': int(dataset.features.shape[0]),
        'train_samples_by_case': dataset.case_counts,
        'training_sample_ticks_per_case': sample_ticks,
        'probe_ticks': {'train': min(probe_ticks, sample_ticks),
                        'validation': probe_ticks},
        'training_steps': training_steps,
        'learning_rate': learning_rate,
        'feature_alpha': FEATURE_ALPHA,
        'network_size': 100,
        'trainable_scalar_count': sum(p.numel() for p in reloaded.parameters()),
        'network_seed': config.network_seed,
        'topology_sha256': topology_before,
        'topology_unchanged': topology_unchanged,
        'gradient_nonzero_scalars': outcome.gradient_nonzero_scalars,
        'gradient_min_abs': outcome.gradient_min_abs,
        'parameter_changed_scalars': outcome.parameter_changed_scalars,
        'parameter_change_min_abs': outcome.parameter_change_min_abs,
        'curve': outcome.curve,
        'checkpoint': checkpoint.name,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'reload_replay_exact': reload_exact,
        'validation': {
            'wider': summarize(wider_rows),
            'original_learned': summarize(original_rows),
            'fixed': summarize(fixed_rows),
        },
        'validation_results': [asdict(row) for row in wider_rows],
        'original_validation_results': [asdict(row) for row in original_rows],
        'fixed_validation_results': [asdict(row) for row in fixed_rows],
        'final_cases_evaluated': False,
        'comparison_sources': {
            'original_learned': ('experiments/fb07-readout/results.json'
                                 if config == EvaluationConfig() else 'recomputed'),
            'fixed': ('experiments/fb06-fixed-baseline/results.json'
                      if config == EvaluationConfig() else 'recomputed'),
        },
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
    """Run the declared diagnosis and print validation summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/fb07d-observability'))
    args = parser.parse_args(argv)
    path = run_diagnosis(args.output_dir)
    payload = json.loads(path.read_text(encoding='utf-8'))
    print(f'Wrote train/validation probe and 4 validation episodes: {path}')
    for name, summary in payload['validation'].items():
        print(f"{name}: success={summary['successes']}/{summary['episodes']} "
              f"progress={summary['mean_progress_fraction']:.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
