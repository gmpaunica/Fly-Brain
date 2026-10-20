"""FB-07H: one matched seven/nine-feature supervised readout experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from collections.abc import Callable, Sequence

import torch

from flybrain.neural.controller import NeuralController
from flybrain.neural.network import SyntheticNetwork
from flybrain.neural.readout import (
    GROUP_NAMES, HybridFeatureFilter, LearnedMotorDecoder, SpikeFeatureFilter,
    TrainableMotorReadout,
)

from .evaluation import (
    EpisodeCase, EpisodeResult, EvaluationConfig, _git_commit, evaluate_cases,
    locked_cases, run_episode, summarize,
)
from .fb07e_probe import CANDIDATE_SCALE
from .records import ensure_new_run_directory
from .readout_experiment import (
    FEATURE_ALPHA, LEARNING_RATE, TRAINING_STEPS, TRAIN_SAMPLE_TICKS,
    TrainingDataset, _RecordingController, _write_curve_svg, teacher_command,
    topology_sha256, train_readout,
)


FEATURE_NAMES = GROUP_NAMES + ('sensory_left_voltage', 'sensory_right_voltage')


def collect_paired_samples(
    cases: Sequence[EpisodeCase], config: EvaluationConfig,
    sample_ticks: int = TRAIN_SAMPLE_TICKS,
) -> tuple[TrainingDataset, TrainingDataset]:
    """Collect seven and nine values on identical locked train observations."""
    allowed = set(locked_cases()['train'])
    if not cases or any(case not in allowed for case in cases):
        raise ValueError('sample cases must be locked train cases')
    if type(sample_ticks) is not int or not 0 < sample_ticks <= config.horizon_ticks:
        raise ValueError('sample_ticks must fit the evaluation horizon')
    sampled = replace(config, horizon_ticks=sample_ticks)
    seven_features = []
    nine_features = []
    targets = []
    counts = {}
    for case in cases:
        recorder = _RecordingController(config.network_seed)
        run_episode(case, sampled, lambda: recorder)
        network = SyntheticNetwork(seed=config.network_seed,
                                   encoder_scale=CANDIDATE_SCALE,
                                   encoder_mode='rational')
        seven_filter = SpikeFeatureFilter(network.topology, FEATURE_ALPHA)
        nine_filter = HybridFeatureFilter(network.topology, FEATURE_ALPHA)
        for observation, _ in recorder.records:
            activity = network.advance(observation)
            seven = seven_filter.update(activity)
            nine = nine_filter.update(activity)
            if not torch.equal(seven, nine[:7]):
                raise RuntimeError('paired spike features differ')
            seven_features.append(seven)
            nine_features.append(nine)
            targets.append(teacher_command(observation))
        counts[case.case_id] = len(recorder.records)
    paired_targets = torch.stack(targets)
    return (
        TrainingDataset(torch.stack(seven_features), paired_targets, counts),
        TrainingDataset(torch.stack(nine_features), paired_targets, counts),
    )


def save_matched_checkpoint(path: Path, model: TrainableMotorReadout, *,
                            network_seed: int) -> None:
    """Save a readout with the exact encoder, feature and topology contract."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema': 2,
        'feature_names': FEATURE_NAMES[:model.feature_count],
        'feature_count': model.feature_count,
        'network_seed': network_seed,
        'topology_sha256': topology_sha256(network_seed),
        'encoder_mode': 'rational',
        'encoder_scale': CANDIDATE_SCALE,
        'feature_alpha': FEATURE_ALPHA,
        'state_dict': model.state_dict(),
    }
    with path.open('xb') as file:
        torch.save(payload, file)


def load_matched_checkpoint(path: Path, *, network_seed: int,
                            feature_count: int) -> TrainableMotorReadout:
    """Reject a checkpoint with a different feature or frozen graph contract."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (payload.get('schema') != 2
            or payload.get('feature_count') != feature_count
            or tuple(payload.get('feature_names', ())) != FEATURE_NAMES[:feature_count]
            or payload.get('encoder_mode') != 'rational'
            or payload.get('encoder_scale') != CANDIDATE_SCALE
            or payload.get('feature_alpha') != FEATURE_ALPHA):
        raise ValueError('checkpoint feature or encoder contract differs')
    if (payload.get('network_seed') != network_seed
            or payload.get('topology_sha256') != topology_sha256(network_seed)):
        raise ValueError('checkpoint topology differs')
    model = TrainableMotorReadout(feature_count)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    return model


def candidate_controller_factory(
    model: TrainableMotorReadout, network_seed: int,
) -> Callable[[], NeuralController]:
    """Create a fresh rational network and matching readout state per case."""
    def create() -> NeuralController:
        network = SyntheticNetwork(seed=network_seed,
                                   encoder_scale=CANDIDATE_SCALE,
                                   encoder_mode='rational')
        decoder = LearnedMotorDecoder(network.topology, model, FEATURE_ALPHA)
        return NeuralController(network, decoder)
    return create


def validation_gate(nine: dict, seven: dict, fixed: dict) -> dict[str, bool]:
    """Read final starts only after an arrival and progress validation gain."""
    arrivals = nine['successes'] > max(seven['successes'], fixed['successes'])
    progress = nine['mean_progress_fraction'] > fixed['mean_progress_fraction']
    return {'arrival_above_both': arrivals, 'progress_above_fixed': progress,
            'passed': arrivals and progress}


def _baseline_validation(path: Path, config: EvaluationConfig) -> tuple[EpisodeResult, ...]:
    """Load FB-06's matched baseline without rerunning final episodes."""
    baseline = json.loads(Path(path).read_text(encoding='utf-8'))
    cases = locked_cases()
    expected = [case.case_id for case in cases['validation'] + cases['final']]
    if (baseline.get('experiment') != 'fb06_fixed_baseline'
            or baseline.get('config') != asdict(config)
            or baseline.get('cases') != {name: [asdict(case) for case in group]
                                             for name, group in cases.items()}
            or [row['case_id'] for row in baseline.get('results', [])] != expected):
        raise ValueError('FB-06 baseline protocol or case IDs differ')
    return tuple(EpisodeResult(**row) for row in baseline['results'][:4])


def _write_episode_table(path: Path, cases: Sequence[EpisodeCase],
                         rows: dict[str, Sequence[EpisodeResult]]) -> None:
    """Write one matched case per row, with failure outcomes visible."""
    with path.open('w', newline='', encoding='utf-8') as file:
        fields = ('case_id', 'seven_outcome', 'nine_outcome', 'fixed_outcome',
                  'seven_progress_fraction', 'nine_progress_fraction',
                  'fixed_progress_fraction', 'seven_final_distance',
                  'nine_final_distance', 'fixed_final_distance')
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, case in enumerate(cases):
            if any(rows[name][index].case_id != case.case_id
                   for name in ('seven', 'nine', 'fixed')):
                raise RuntimeError('comparison case IDs differ')
            writer.writerow({
                'case_id': case.case_id,
                **{f'{name}_{field}': getattr(rows[name][index], field)
                   for name in ('seven', 'nine', 'fixed')
                   for field in ('outcome', 'progress_fraction', 'final_distance')},
            })


def run_experiment(
    output_dir: Path, baseline_path: Path, *,
    config: EvaluationConfig | None = None,
    sample_ticks: int = TRAIN_SAMPLE_TICKS,
    steps: int = TRAINING_STEPS,
    learning_rate: float = LEARNING_RATE,
) -> Path:
    """Train once per feature set, validate, and gate final evaluation."""
    ensure_new_run_directory(output_dir)
    config = config or EvaluationConfig()
    cases = locked_cases()
    fixed_validation = _baseline_validation(baseline_path, config)
    topology_before = topology_sha256(config.network_seed)
    datasets = dict(zip(('seven', 'nine'), collect_paired_samples(
        cases['train'], config, sample_ticks)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    loaded_models = {}
    models = {}
    for name, dataset in datasets.items():
        outcome = train_readout(dataset.features, dataset.targets,
                                steps=steps, learning_rate=learning_rate)
        checkpoint = output_dir / f'{name}.pt'
        save_matched_checkpoint(checkpoint, outcome.model,
                                network_seed=config.network_seed)
        loaded = load_matched_checkpoint(checkpoint, network_seed=config.network_seed,
                                         feature_count=dataset.features.shape[1])
        curve_path = output_dir / f'{name}_curve.csv'
        with curve_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=('step', 'loss'))
            writer.writeheader()
            writer.writerows(outcome.curve)
        _write_curve_svg(curve_path, output_dir / f'{name}_curve.svg')
        outcomes[name] = outcome
        loaded_models[name] = loaded
        models[name] = {
            'feature_count': loaded.feature_count,
            'feature_names': FEATURE_NAMES[:loaded.feature_count],
            'trainable_scalar_count': sum(p.numel() for p in loaded.parameters()),
            'gradient_nonzero_scalars': outcome.gradient_nonzero_scalars,
            'gradient_min_abs': outcome.gradient_min_abs,
            'parameter_changed_scalars': outcome.parameter_changed_scalars,
            'parameter_change_min_abs': outcome.parameter_change_min_abs,
            'curve': outcome.curve,
            'checkpoint': checkpoint.name,
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    validation_cases = cases['validation']
    validation_rows = {
        name: evaluate_cases(validation_cases, config,
                             candidate_controller_factory(model, config.network_seed))
        for name, model in loaded_models.items()
    }
    validation_rows['fixed'] = fixed_validation
    for name in ('seven', 'nine'):
        replay = run_episode(validation_cases[0], config,
                             candidate_controller_factory(outcomes[name].model,
                                                          config.network_seed))
        models[name]['reload_replay_exact'] = replay == validation_rows[name][0]
        if not models[name]['reload_replay_exact']:
            raise RuntimeError(f'{name} checkpoint did not replay validation')
    _write_episode_table(output_dir / 'validation.csv', validation_cases,
                         validation_rows)
    validation = {name: summarize(rows) for name, rows in validation_rows.items()}
    gate = validation_gate(validation['nine'], validation['seven'],
                           validation['fixed'])
    results = {'validation': {name: [asdict(row) for row in rows]
                              for name, rows in validation_rows.items()}}
    summaries = {'validation': validation}
    if gate['passed']:
        final_cases = cases['final']
        baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8'))
        final_rows = {
            name: evaluate_cases(final_cases, config,
                                 candidate_controller_factory(model,
                                                              config.network_seed))
            for name, model in loaded_models.items()
        }
        final_rows['fixed'] = tuple(EpisodeResult(**row)
                                    for row in baseline['results'][4:])
        _write_episode_table(output_dir / 'final.csv', final_cases, final_rows)
        results['final'] = {name: [asdict(row) for row in rows]
                            for name, rows in final_rows.items()}
        summaries['final'] = {name: summarize(rows)
                              for name, rows in final_rows.items()}
    topology_unchanged = topology_before == topology_sha256(config.network_seed)
    if not topology_unchanged:
        raise RuntimeError('fixed topology changed')
    payload = {
        'experiment': 'fb07h_matched_voltage_readout',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'objective': 'train-only sensory imitation; test held-out food arrival',
        'teacher': {'forward': 0.7,
                    'turn': 'clip(8*(odor_right-odor_left)/max(odor_right+odor_left,1e-9),-1,1)'},
        'encoder_rule': '2.5*c/(c+0.005)',
        'optimizer': 'Adam full batch',
        'training_steps': steps,
        'learning_rate': learning_rate,
        'training_sample_ticks_per_case': sample_ticks,
        'train_sample_count': int(datasets['seven'].features.shape[0]),
        'train_samples_by_case': datasets['seven'].case_counts,
        'feature_alpha': FEATURE_ALPHA,
        'network_seed': config.network_seed,
        'network_size': 100,
        'topology_sha256': topology_before,
        'topology_unchanged': topology_unchanged,
        'models': models,
        'validation_gate_rule': ('nine validation arrivals above seven and fixed; '
                                 'nine mean progress above fixed'),
        'validation_gate': gate,
        'final_cases_evaluated': gate['passed'],
        'baseline': str(baseline_path),
        'software': {'python': sys.version.split()[0], 'torch': torch.__version__,
                     'pygame': version('pygame'), 'platform': platform.platform(),
                     'device': 'cpu'},
        'config': asdict(config),
        'cases': {split: [asdict(case) for case in cases[split]]
                  for split in ('train', 'validation') +
                  (('final',) if gate['passed'] else ())},
        'summaries': summaries,
        'results': results,
    }
    path = output_dir / 'results.json'
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')
    return path


def main(argv: list[str] | None = None) -> int:
    """Run one predeclared matched readout experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/fb07h-matched-readout'))
    parser.add_argument('--baseline', type=Path,
                        default=Path('experiments/fb06-fixed-baseline/results.json'))
    args = parser.parse_args(argv)
    path = run_experiment(args.output_dir, args.baseline)
    result = json.loads(path.read_text(encoding='utf-8'))
    print(f'Wrote matched readout experiment: {path}')
    for split, models in result['summaries'].items():
        print(f"{split}: fixed={models['fixed']['successes']} "
              f"seven={models['seven']['successes']} "
              f"nine={models['nine']['successes']} arrivals")
    print(f"Validation gate passed: {result['validation_gate']['passed']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
