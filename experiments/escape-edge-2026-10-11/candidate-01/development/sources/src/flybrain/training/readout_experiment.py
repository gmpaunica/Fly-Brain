"""FB-07: bounded sensory-teacher training above a frozen spike graph."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from flybrain.neural.controller import NeuralController, NeuralDecision
from flybrain.neural.motor import MotorDecoder
from flybrain.neural.network import NeuralActivity, SyntheticNetwork
from flybrain.neural.readout import (
    GROUP_NAMES, LearnedMotorDecoder, SpikeFeatureFilter, TrainableMotorReadout,
)
from flybrain.world.sensing import SensoryObservation
from .records import ensure_new_run_directory

from .evaluation import (
    EpisodeCase, EpisodeResult, EvaluationConfig, _git_commit, evaluate_cases,
    locked_cases, run_episode, summarize,
)


TRAIN_SAMPLE_TICKS = 200
TRAINING_STEPS = 80
LEARNING_RATE = 0.05
FEATURE_ALPHA = 0.5
TEACHER_TURN_GAIN = 8.0


@dataclass(frozen=True)
class TrainingDataset:
    """Detached train-only feature/teacher tensors and per-case sample counts."""

    features: Tensor
    targets: Tensor
    case_counts: dict[str, int]


@dataclass(frozen=True)
class TrainingOutcome:
    """Trained model and observed autograd/optimizer evidence."""

    model: TrainableMotorReadout
    curve: tuple[dict[str, float | int], ...]
    gradient_nonzero_scalars: int
    gradient_min_abs: float
    parameter_changed_scalars: int
    parameter_change_min_abs: float


class _RecordingController(NeuralController):
    """Observe fixed decisions for offline training without changing commands."""

    def __init__(self, network_seed: int, encoder_scale: float = 0.025) -> None:
        network = SyntheticNetwork(seed=network_seed, encoder_scale=encoder_scale)
        super().__init__(network, MotorDecoder(network.topology))
        self.records: list[tuple[SensoryObservation, NeuralActivity]] = []

    def decide(self, observation: SensoryObservation) -> NeuralDecision:
        decision = super().decide(observation)
        self.records.append((observation, decision.activity))
        return decision


def teacher_command(observation: SensoryObservation) -> Tensor:
    """Hand-written sensory-only imitation target, not an embodied reward."""
    left, right = observation.odor_left, observation.odor_right
    contrast = (right - left) / max(left + right, 1e-9)
    turn = max(-1.0, min(1.0, TEACHER_TURN_GAIN * contrast))
    return torch.tensor((0.7, turn), dtype=torch.float32, device='cpu')


def collect_training_samples(
    cases: Sequence[EpisodeCase], config: EvaluationConfig,
    sample_ticks: int = TRAIN_SAMPLE_TICKS,
    encoder_scale: float = 0.025,
) -> TrainingDataset:
    """Collect fixed-controller train trajectories; never touch held-out starts."""
    allowed = set(locked_cases()['train'])
    if not cases or any(case not in allowed for case in cases):
        raise ValueError('sample cases must be locked train cases')
    if type(sample_ticks) is not int or not 0 < sample_ticks <= config.horizon_ticks:
        raise ValueError('sample_ticks must fit the evaluation horizon')
    sampled_config = replace(config, horizon_ticks=sample_ticks)
    features: list[Tensor] = []
    targets: list[Tensor] = []
    case_counts: dict[str, int] = {}
    for case in cases:
        recorder = _RecordingController(config.network_seed, encoder_scale)
        run_episode(case, sampled_config, lambda: recorder)
        feature_filter = SpikeFeatureFilter(recorder.network.topology, FEATURE_ALPHA)
        for observation, activity in recorder.records:
            features.append(feature_filter.update(activity))
            targets.append(teacher_command(observation))
        case_counts[case.case_id] = len(recorder.records)
    return TrainingDataset(torch.stack(features), torch.stack(targets), case_counts)


def train_readout(
    features: Tensor, targets: Tensor, *, steps: int = TRAINING_STEPS,
    learning_rate: float = LEARNING_RATE,
) -> TrainingOutcome:
    """Fit only the small readout with full-batch Adam and record every loss."""
    if (features.ndim != 2 or features.shape[1] not in (7, 9)
            or targets.shape != (features.shape[0], 2) or features.shape[0] == 0
            or features.device.type != 'cpu' or targets.device.type != 'cpu'
            or features.dtype != torch.float32 or targets.dtype != torch.float32
            or not bool(torch.isfinite(features).all().item())
            or not bool(torch.isfinite(targets).all().item())):
        raise ValueError('training tensors must be finite CPU float32 [N,7/9] and [N,2]')
    if type(steps) is not int or steps <= 0:
        raise ValueError('steps must be a positive integer')
    if (not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool)
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('learning_rate must be positive and finite')
    model = TrainableMotorReadout(feature_count=features.shape[1])
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial_loss = float(torch.nn.functional.mse_loss(model(features), targets).item())
    curve: list[dict[str, float | int]] = [{'step': 0, 'loss': initial_loss}]
    gradient_parts: list[Tensor] = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(features), targets)
        loss.backward()
        if step == 1:
            gradient_parts = [parameter.grad.detach().abs().flatten().clone()
                              for parameter in model.parameters()]
        optimizer.step()
        with torch.no_grad():
            measured = float(torch.nn.functional.mse_loss(model(features), targets).item())
        curve.append({'step': step, 'loss': measured})
    gradients = torch.cat(gradient_parts)
    changes = torch.cat([
        (parameter.detach() - before).abs().flatten()
        for parameter, before in zip(model.parameters(), initial)
    ])
    if not bool(torch.isfinite(gradients).all().item()) or not math.isfinite(curve[-1]['loss']):
        raise RuntimeError('nonfinite training gradient or loss')
    model.eval()
    return TrainingOutcome(
        model=model,
        curve=tuple(curve),
        gradient_nonzero_scalars=int(torch.count_nonzero(gradients).item()),
        gradient_min_abs=float(gradients.min().item()),
        parameter_changed_scalars=int(torch.count_nonzero(changes).item()),
        parameter_change_min_abs=float(changes.min().item()),
    )


def topology_sha256(network_seed: int) -> str:
    """Hash the frozen matrix in W[post,pre] order for checkpoint validation."""
    weights = SyntheticNetwork(seed=network_seed).topology.weights
    return hashlib.sha256(weights.numpy().tobytes()).hexdigest()


def save_checkpoint(path: Path, model: TrainableMotorReadout, *,
                    network_seed: int, alpha: float,
                    encoder_scale: float = 0.025) -> None:
    """Save only readout weights plus the frozen input/topology contract."""
    if model.feature_count != len(GROUP_NAMES):
        raise ValueError('original checkpoint requires seven spike features')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema': 1,
        'group_names': GROUP_NAMES,
        'network_seed': network_seed,
        'topology_sha256': topology_sha256(network_seed),
        'feature_alpha': alpha,
        'encoder_scale': encoder_scale,
        'state_dict': model.state_dict(),
    }
    with path.open('xb') as file:
        torch.save(payload, file)


def load_checkpoint(path: Path, *, network_seed: int,
                    alpha: float = FEATURE_ALPHA,
                    encoder_scale: float = 0.025) -> TrainableMotorReadout:
    """Reject a checkpoint whose input groups or fixed graph differ."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (payload['schema'] != 1 or tuple(payload['group_names']) != GROUP_NAMES
            or payload['network_seed'] != network_seed
            or payload['topology_sha256'] != topology_sha256(network_seed)
            or payload['feature_alpha'] != alpha):
        raise ValueError('checkpoint topology or feature contract differs')
    if payload.get('encoder_scale', 0.025) != encoder_scale:
        raise ValueError('checkpoint encoder scale differs')
    model = TrainableMotorReadout()
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    return model


def learned_controller_factory(
    model: TrainableMotorReadout, network_seed: int,
    encoder_scale: float = 0.025,
) -> Callable[[], NeuralController]:
    """Make fresh spiking and EMA state per evaluation episode."""
    def create() -> NeuralController:
        network = SyntheticNetwork(seed=network_seed, encoder_scale=encoder_scale)
        return NeuralController(
            network, LearnedMotorDecoder(network.topology, model, FEATURE_ALPHA),
        )
    return create


def _write_curve_svg(csv_path: Path, svg_path: Path) -> None:
    """Plot loss by reading the already recorded CSV, with no plot dependency."""
    with csv_path.open(newline='', encoding='utf-8') as file:
        points = [(int(row['step']), float(row['loss']))
                  for row in csv.DictReader(file)]
    max_step = max(step for step, _ in points)
    low = min(loss for _, loss in points)
    high = max(loss for _, loss in points)
    span = max(high - low, 1e-9)
    polyline = ' '.join(
        f'{60 + 540 * step / max(max_step, 1):.2f},'
        f'{260 - 210 * (loss - low) / span:.2f}'
        for step, loss in points
    )
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="320" '
        'viewBox="0 0 640 320">\n'
        '<rect width="640" height="320" fill="white"/>\n'
        '<path d="M60 40 V260 H610" fill="none" stroke="#333"/>\n'
        f'<polyline points="{polyline}" fill="none" stroke="#205ca4" '
        'stroke-width="2"/>\n'
        '<text x="300" y="305" font-size="14">optimizer step</text>\n'
        '<text x="60" y="30" font-size="14">mean squared teacher error</text>\n'
        f'<text x="4" y="55" font-size="12">{high:.4f}</text>\n'
        f'<text x="4" y="260" font-size="12">{low:.4f}</text>\n'
        '</svg>\n', encoding='utf-8',
    )


def run_experiment(
    output_dir: Path, baseline_path: Path, *,
    config: EvaluationConfig | None = None,
    sample_ticks: int = TRAIN_SAMPLE_TICKS,
    steps: int = TRAINING_STEPS,
    learning_rate: float = LEARNING_RATE,
) -> Path:
    """Train once, reload, and compare the same held-out cases with FB-06."""
    ensure_new_run_directory(output_dir)
    config = config or EvaluationConfig()
    cases = locked_cases()
    baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8'))
    expected_cases = {split: [asdict(case) for case in group]
                      for split, group in cases.items()}
    eval_cases = cases['validation'] + cases['final']
    expected_ids = [case.case_id for case in eval_cases]
    if (baseline.get('experiment') != 'fb06_fixed_baseline'
            or baseline.get('config') != asdict(config)
            or baseline.get('cases') != expected_cases
            or [row['case_id'] for row in baseline.get('results', [])] != expected_ids):
        raise ValueError('FB-06 baseline protocol or case IDs differ')
    fixed_rows = [EpisodeResult(**row) for row in baseline['results']]
    topology_before = topology_sha256(config.network_seed)
    dataset = collect_training_samples(cases['train'], config, sample_ticks)
    first_indices = []
    offset = 0
    for count in dataset.case_counts.values():
        first_indices.append(offset)
        offset += count
    first_features = dataset.features[first_indices]
    first_teacher_turns = dataset.targets[first_indices, 1]
    outcome = train_readout(dataset.features, dataset.targets,
                            steps=steps, learning_rate=learning_rate)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'readout.pt'
    save_checkpoint(checkpoint_path, outcome.model,
                    network_seed=config.network_seed, alpha=FEATURE_ALPHA)
    reloaded = load_checkpoint(checkpoint_path, network_seed=config.network_seed)
    learned_rows = evaluate_cases(
        eval_cases, config, learned_controller_factory(reloaded, config.network_seed),
    )
    replay = run_episode(
        eval_cases[0], config,
        learned_controller_factory(outcome.model, config.network_seed),
    )
    reload_replay_exact = replay == learned_rows[0]
    if not reload_replay_exact:
        raise RuntimeError('saved checkpoint did not reproduce the first validation episode')
    topology_unchanged = topology_before == topology_sha256(config.network_seed)
    if not topology_unchanged:
        raise RuntimeError('fixed spiking topology changed')

    curve_path = output_dir / 'curve.csv'
    with curve_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=('step', 'loss'))
        writer.writeheader()
        writer.writerows(outcome.curve)
    _write_curve_svg(curve_path, output_dir / 'curve.svg')
    with (output_dir / 'episodes.csv').open('w', newline='', encoding='utf-8') as file:
        fields = list(EpisodeResult.__dataclass_fields__) + [
            'fixed_outcome', 'fixed_progress_fraction', 'fixed_final_distance',
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for learned, fixed in zip(learned_rows, fixed_rows):
            row = asdict(learned)
            row.update({
                'fixed_outcome': fixed.outcome,
                'fixed_progress_fraction': fixed.progress_fraction,
                'fixed_final_distance': fixed.final_distance,
            })
            writer.writerow(row)
    summaries = {}
    for split in ('validation', 'final'):
        current = [row for row in learned_rows if row.split == split]
        previous = [row for row in fixed_rows if row.split == split]
        learned_summary = summarize(current)
        fixed_summary = summarize(previous)
        summaries[split] = {
            'learned': learned_summary,
            'fixed': fixed_summary,
            'success_rate_delta': (learned_summary['success_rate']
                                   - fixed_summary['success_rate']),
            'mean_progress_delta': (learned_summary['mean_progress_fraction']
                                    - fixed_summary['mean_progress_fraction']),
        }
    payload = {
        'experiment': 'fb07_supervised_spike_readout',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'controller': 'frozen_synthetic_100_learned_7x2_readout',
        'objective': 'MSE to sensory-only hand-written turn teacher; evaluate held-out arrival',
        'teacher': {
            'forward': 0.7,
            'turn': 'clip(8*(odor_right-odor_left)/max(odor_right+odor_left,1e-9),-1,1)',
        },
        'optimizer': 'Adam full batch',
        'training_steps': steps,
        'learning_rate': learning_rate,
        'training_sample_ticks_per_case': sample_ticks,
        'train_sample_count': int(dataset.features.shape[0]),
        'train_samples_by_case': dataset.case_counts,
        'first_tick_unique_feature_vectors': int(
            torch.unique(first_features, dim=0).shape[0]),
        'first_tick_unique_teacher_turns': int(
            torch.unique(first_teacher_turns).numel()),
        'network_size': 100,
        'trainable_scalar_count': sum(p.numel() for p in reloaded.parameters()),
        'feature_groups': GROUP_NAMES,
        'feature_alpha': FEATURE_ALPHA,
        'network_seed': config.network_seed,
        'topology_sha256': topology_before,
        'topology_unchanged': topology_unchanged,
        'gradient_nonzero_scalars': outcome.gradient_nonzero_scalars,
        'gradient_min_abs': outcome.gradient_min_abs,
        'parameter_changed_scalars': outcome.parameter_changed_scalars,
        'parameter_change_min_abs': outcome.parameter_change_min_abs,
        'checkpoint': checkpoint_path.name,
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'reload_replay_exact': reload_replay_exact,
        'baseline': str(baseline_path),
        'software': {
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'pygame': version('pygame'),
            'platform': platform.platform(),
            'device': 'cpu',
        },
        'config': asdict(config),
        'cases': expected_cases,
        'curve': outcome.curve,
        'summaries': summaries,
        'fixed_results': [asdict(row) for row in fixed_rows],
        'results': [asdict(row) for row in learned_rows],
    }
    result_path = output_dir / 'results.json'
    result_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')
    return result_path


def main(argv: list[str] | None = None) -> int:
    """Run the predeclared FB-07 attempt and write all measured artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/fb07-readout'))
    parser.add_argument('--baseline', type=Path,
                        default=Path('experiments/fb06-fixed-baseline/results.json'))
    args = parser.parse_args(argv)
    result_path = run_experiment(args.output_dir, args.baseline)
    payload = json.loads(result_path.read_text(encoding='utf-8'))
    print(f"Wrote {len(payload['results'])} held-out episodes: {result_path}")
    for split in ('validation', 'final'):
        summary = payload['summaries'][split]
        learned = summary['learned']
        fixed = summary['fixed']
        print(f"{split}: learned {learned['successes']}/{learned['episodes']} "
              f"vs fixed {fixed['successes']}/{fixed['episodes']}; "
              f"progress {learned['mean_progress_fraction']:.3f} "
              f"vs {fixed['mean_progress_fraction']:.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
