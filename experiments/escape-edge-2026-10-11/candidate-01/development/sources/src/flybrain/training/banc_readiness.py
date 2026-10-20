"""Bounded train-only teacher/endpoint diagnostics; never fits a model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

from flybrain.neural.banc_motor import make_banc_controller
from flybrain.neural.banc_network import load_verified_banc_config
from flybrain.neural.controller import NeuralController, NeuralDecision
from flybrain.world.fly import MotorCommand
from flybrain.world.sensing import SensoryObservation
from .evaluation import EpisodeResult, EvaluationConfig, evaluate_cases, locked_cases, summarize
from .readout_experiment import teacher_command
from .records import ensure_new_run_directory


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def capture_provenance(project_root: Path, output_dir: Path) -> dict[str, dict[str, str]]:
    """Hash accepted evidence, raw data, Python sources and prior run artifacts."""
    root = project_root.resolve()
    package = root / 'src/flybrain'
    bundle = [package / 'data' / name for name in (
        'banc_dynamics_config.json', 'banc_dynamics_results.json',
        'banc_dynamics_protocol.json', 'banc_v888_va2_dna02.json')]
    groups = {
        'accepted_bundle_and_subset_sha256': bundle,
        'source_code_sha256': sorted((root / 'src').rglob('*.py')),
        'previous_files_sha256': sorted(path for path in (root / 'experiments').rglob('*')
            if path.is_file() and not path.resolve().is_relative_to(output_dir.resolve())),
    }
    return {name: {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in paths} for name, paths in groups.items()}


def verify_provenance(record: dict[str, dict[str, str]]) -> bool:
    """Reject changed or missing files from a pre-run provenance record."""
    return all(Path(path).is_file() and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
               for group in record.values() for path, digest in group.items())


def weights_sha256(controller: NeuralController) -> str:
    """Hash actual effective float32 CPU weights in destination/source order."""
    return hashlib.sha256(controller.network.population.weights.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_new_json(path: Path, payload: dict) -> None:
    """Write one new artifact, preserving any existing partial or completed file."""
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def execution_accounting(teacher: Sequence[EpisodeResult], fixed: Sequence[EpisodeResult],
                         sample_count: int) -> dict:
    """Reconcile actual ticks and samples against the sixteen locked episodes."""
    teacher_ticks = sum(episode.ticks for episode in teacher)
    fixed_ticks = sum(episode.ticks for episode in fixed)
    expected = [(case.case_id, case.split, case.seed) for case in locked_cases()['train']]
    complete = all([(episode.case_id, episode.split, episode.seed) for episode in group] == expected
                   for group in (teacher, fixed))
    within_horizons = all(type(episode.ticks) is int and 0 <= episode.ticks <= 600
                          for episode in (*teacher, *fixed))
    return {
        'teacher_world_ticks': teacher_ticks, 'fixed_banc_world_ticks': fixed_ticks,
        'actual_world_ticks': teacher_ticks + fixed_ticks,
        'execution_budget_passed': (complete and within_horizons
            and teacher_ticks + fixed_ticks <= 9600 and sample_count == teacher_ticks),
    }


def summarize_features(features: torch.Tensor, targets: torch.Tensor) -> dict:
    """Report variation, constant baseline and exact-feature alias error."""
    if (features.ndim != 2 or features.shape[1] != 8 or not len(features)
            or targets.shape != (len(features), 2)
            or not torch.isfinite(features).all() or not torch.isfinite(targets).all()):
        raise ValueError('expected finite nonempty [N,8] features and [N,2] targets')
    features = features.detach().to(device='cpu', dtype=torch.float64)
    targets = targets.detach().to(device='cpu', dtype=torch.float64)
    unique, inverse = torch.unique(features, dim=0, return_inverse=True)
    counts = torch.bincount(inverse)
    sums = torch.zeros(len(unique), dtype=torch.float64, device='cpu')
    sums.scatter_add_(0, inverse, targets[:, 1])
    aliases = sums / counts
    mean = targets.mean(dim=0)
    return {
        'samples': len(features),
        'feature_std': features.std(dim=0, correction=0).tolist(),
        'unique_feature_rows': len(unique),
        'mean_command': mean.tolist(),
        'constant_mse': ((targets - mean) ** 2).mean(dim=0).tolist(),
        'exact_alias_turn_mse_lower_bound': float(((targets[:, 1] - aliases[inverse]) ** 2).mean()),
    }


class TeacherRecorder(NeuralController):
    """Training-data scaffold: teacher moves the body; BANC only emits features."""

    def __init__(self, config, rows: list[dict], episode: int) -> None:
        controller = make_banc_controller(config)
        super().__init__(controller.network, controller.decoder)
        self.rows = rows
        self.episode = episode
        self.tick = 0

    def reset(self) -> None:
        """Reset episode state and tick labels, retaining externally owned rows."""
        super().reset()
        self.tick = 0

    def decide(self, observation: SensoryObservation) -> NeuralDecision:
        decision = super().decide(observation)
        target = teacher_command(observation)
        self.rows.append({
            'case_id': f'train-{self.episode}', 'tick': self.tick,
            'left_odor': observation.odor_left, 'right_odor': observation.odor_right,
            **dict(zip(self.decoder.feature_names, self.decoder.feature_rates.tolist())),
            'target_forward': float(target[0]), 'target_turn': float(target[1]),
        })
        self.tick += 1
        return NeuralDecision(decision.activity, MotorCommand(float(target[0]), float(target[1])))


def run_readiness(output_dir: Path) -> dict:
    """Run eight teacher and eight fixed-BANC training starts, at most 9,600 ticks."""
    output_dir = Path(output_dir).resolve()
    ensure_new_run_directory(output_dir)
    provenance = capture_provenance(PROJECT_ROOT, output_dir)
    config = load_verified_banc_config()
    cases = locked_cases()['train']
    world = EvaluationConfig()
    if len(cases) != 8 or any(case.split != 'train' for case in cases) or 2 * len(cases) * world.horizon_ticks > 9600:
        raise ValueError('readiness requires eight training cases and at most 9,600 world ticks')
    reference = make_banc_controller(config)
    weight_digest = weights_sha256(reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        'hypothesis': 'The accepted endpoint interface supplies finite varying features on teacher food-approach trajectories.',
        'max_world_ticks': 9600, 'training_steps': 0, 'optimizer_steps': 0, 'learning_rate': None,
        'checkpoint': None, 'cases': [asdict(case) for case in cases],
        'world_config': asdict(world), 'dynamics_config': asdict(config),
        'gates': ['teacher succeeds on all eight training starts',
                  'both steering feature channels have nonzero variation'],
        'descriptive_only': ['exact-feature aliasing', 'fixed BANC arrival performance'],
        'stop_condition': 'one teacher and one fixed-BANC episode per training case; no retries or optimization',
        'provenance': provenance,
        'effective_weights_sha256': weight_digest,
        'weight_serialization': 'contiguous CPU float32 bytes; W[post,pre]',
    }
    save_new_json(output_dir / 'protocol.json', protocol)
    protocol_digest = hashlib.sha256((output_dir / 'protocol.json').read_bytes()).hexdigest()
    rows: list[dict] = []
    next_episode = iter(range(8))
    controllers: list[NeuralController] = []
    def factory(teacher: bool) -> NeuralController:
        controller = (TeacherRecorder(config, rows, next(next_episode)) if teacher
                      else make_banc_controller(config))
        if weights_sha256(controller) != weight_digest:
            raise ValueError('readiness controller effective weights changed before episode')
        controllers.append(controller)
        return controller
    with torch.no_grad():
        teacher = evaluate_cases(cases, world, lambda: factory(True))
        fixed = evaluate_cases(cases, world, lambda: factory(False))
    names = reference.decoder.feature_names
    features = torch.tensor([[row[name] for name in names] for row in rows], dtype=torch.float64, device='cpu')
    targets = torch.tensor([[row['target_forward'], row['target_turn']] for row in rows], dtype=torch.float64, device='cpu')
    metrics = summarize_features(features, targets)
    odors = [row[side] for row in rows for side in ('left_odor', 'right_odor')]
    with (output_dir / 'samples.csv').open('x', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    teacher_summary = summarize(teacher)
    execution = execution_accounting(teacher, fixed, len(rows))
    provenance_preserved = verify_provenance(provenance)
    weights_unchanged = all(weights_sha256(controller) == weight_digest for controller in controllers)
    protocol_unchanged = hashlib.sha256((output_dir / 'protocol.json').read_bytes()).hexdigest() == protocol_digest
    result = {
        **protocol, 'experiment': 'fb09_banc_training_readiness',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'software': {'python': sys.version.split()[0], 'torch': torch.__version__,
                     'platform': platform.platform(), 'device': 'cpu'},
        'git_commit': None, 'network_size': 83, 'feature_names': list(names),
        'teacher_summary': teacher_summary, 'fixed_banc_summary': summarize(fixed),
        'teacher_episodes': [asdict(r) for r in teacher],
        'fixed_banc_episodes': [asdict(r) for r in fixed],
        'features': metrics, 'observed_odor_range': [min(odors), max(odors)],
        **execution,
        'protocol_sha256': protocol_digest,
        'provenance_preserved': provenance_preserved,
        'effective_weights_unchanged': weights_unchanged,
        'protocol_unchanged': protocol_unchanged,
        'readiness_passed': (teacher_summary['successes'] == 8
            and all(s > 0 for s in metrics['feature_std'][:2])
            and execution['execution_budget_passed'] and provenance_preserved
            and weights_unchanged and protocol_unchanged),
        'samples_sha256': hashlib.sha256((output_dir / 'samples.csv').read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save_new_json(output_dir / 'results.json', result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/fb09-banc-readiness'))
    args = parser.parse_args(argv)
    result = run_readiness(args.output_dir)
    print(json.dumps({key: result[key] for key in ('readiness_passed', 'teacher_summary', 'fixed_banc_summary', 'features')}, indent=2))
    return 0 if result['readiness_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
