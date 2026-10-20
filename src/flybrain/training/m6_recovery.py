"""One bounded, train-only numerical recovery of the M6 motor readout."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys

import torch
from torch import Tensor

from flybrain.neural.readout import TrainableMotorReadout

from .evaluation import EvaluationConfig, _git_commit, evaluate_cases, locked_cases, summarize
from .fb07h_training import (
    collect_paired_samples, candidate_controller_factory, load_matched_checkpoint,
    save_matched_checkpoint,
)
from .readout_experiment import topology_sha256


MAX_UPDATES = 500
TURN_GATE_MSE = 0.016752


@dataclass(frozen=True)
class ConditionedFit:
    """Folded nine-input readout and observed optimizer evidence."""

    model: TrainableMotorReadout
    curve: tuple[dict[str, float | int], ...]
    gradient_nonzero_scalars: int
    gradient_min_abs: float
    parameter_changed_scalars: int
    parameter_change_min_abs: float
    mean: Tensor
    scale: Tensor


def fold_standardization(model: TrainableMotorReadout, mean: Tensor,
                         scale: Tensor) -> TrainableMotorReadout:
    """Fold z=(x-mean)/scale into linear weights without changing outputs."""
    if (model.feature_count != 9 or mean.shape != (9,) or scale.shape != (9,)
            or not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(scale).all())
            or not bool((scale > 0).all())):
        raise ValueError('nine finite feature means and positive scales required')
    folded = TrainableMotorReadout(9)
    with torch.no_grad():
        raw_weight = model.linear.weight / scale[None, :]
        folded.linear.weight.copy_(raw_weight)
        folded.linear.bias.copy_(model.linear.bias - raw_weight @ mean)
    folded.eval()
    return folded


def fit_conditioned_readout(features: Tensor, targets: Tensor, *,
                            steps: int = MAX_UPDATES) -> ConditionedFit:
    """Fit once with train-only feature scaling and a strict update ceiling."""
    if (features.ndim != 2 or features.shape[1] != 9
            or features.shape[0] == 0 or targets.shape != (len(features), 2)
            or features.dtype != torch.float32 or targets.dtype != torch.float32
            or features.device.type != 'cpu' or targets.device.type != 'cpu'
            or not bool(torch.isfinite(features).all())
            or not bool(torch.isfinite(targets).all())):
        raise ValueError('finite CPU float32 [N,9] features and [N,2] targets required')
    if type(steps) is not int or not 1 <= steps <= MAX_UPDATES:
        raise ValueError('steps must be an integer from 1 through 500')
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / scale
    model = TrainableMotorReadout(9)
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    def measure(step: int) -> dict[str, float | int]:
        with torch.no_grad():
            error = (model(normalized) - targets).square().mean(dim=0)
        values = {'step': step, 'forward_mse': float(error[0]),
                  'turn_mse': float(error[1])}
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError('nonfinite readout error')
        return values

    curve = [measure(0)]
    first_gradient: Tensor | None = None
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(normalized) - targets).square().mean()
        loss.backward()
        gradients = torch.cat([parameter.grad.detach().flatten().clone()
                               for parameter in model.parameters()])
        if not bool(torch.isfinite(gradients).all()):
            raise RuntimeError('nonfinite readout gradient')
        if first_gradient is None:
            first_gradient = gradients
        optimizer.step()
        curve.append(measure(step))

    assert first_gradient is not None
    changes = torch.cat([(parameter.detach() - before).abs().flatten()
                         for parameter, before in zip(model.parameters(), initial)])
    folded = fold_standardization(model, mean, scale)
    with torch.no_grad():
        deviation = (folded(features) - model(normalized)).abs().max()
    if float(deviation) > 2e-5:
        raise RuntimeError('folded readout disagrees with scaled training model')
    return ConditionedFit(
        model=folded, curve=tuple(curve),
        gradient_nonzero_scalars=int(torch.count_nonzero(first_gradient).item()),
        gradient_min_abs=float(first_gradient.abs().min()),
        parameter_changed_scalars=int(torch.count_nonzero(changes).item()),
        parameter_change_min_abs=float(changes.min()), mean=mean, scale=scale,
    )


def run_recovery(output_dir: Path) -> Path:
    """Record one train-only fit and eight gain-one training episodes."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('choose an unused experiment directory')
    config = EvaluationConfig()
    cases = locked_cases()['train']
    topology_before = topology_sha256(config.network_seed)
    dataset = collect_paired_samples(cases, config)[1]
    outcome = fit_conditioned_readout(dataset.features, dataset.targets)
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / 'nine.pt'
    save_matched_checkpoint(checkpoint, outcome.model,
                            network_seed=config.network_seed)
    reloaded = load_matched_checkpoint(checkpoint, network_seed=config.network_seed,
                                       feature_count=9)
    with torch.no_grad():
        model_error = (reloaded(dataset.features) - dataset.targets).square().mean(dim=0)
        mean_command = dataset.targets.mean(dim=0)
        constant_error = (mean_command - dataset.targets).square().mean(dim=0)
    train_rows = evaluate_cases(cases, config,
                                candidate_controller_factory(reloaded, config.network_seed))
    replay_rows = evaluate_cases(cases[:1], config,
                                 candidate_controller_factory(outcome.model,
                                                              config.network_seed))
    reload_replay_exact = replay_rows[0] == train_rows[0]
    topology_unchanged = topology_before == topology_sha256(config.network_seed)
    if not reload_replay_exact or not topology_unchanged:
        raise RuntimeError('checkpoint replay or fixed topology check failed')
    finite_updates = (outcome.gradient_nonzero_scalars == 20
                      and outcome.parameter_changed_scalars == 20
                      and math.isfinite(outcome.gradient_min_abs)
                      and math.isfinite(outcome.parameter_change_min_abs))
    gate = {
        'turn_mse_at_most_0_016752': float(model_error[1]) <= TURN_GATE_MSE,
        'train_arrivals_at_least_7_of_8': summarize(train_rows)['successes'] >= 7,
        'finite_intended_gradients_and_updates': finite_updates,
        'topology_unchanged': topology_unchanged,
        'checkpoint_replay_exact': reload_replay_exact,
    }
    gate['passed'] = all(gate.values())
    payload = {
        'experiment': 'm6_readout_recovery',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'hypothesis': 'raw feature offsets/scales impair the original Adam fit',
        'correction': 'train-only per-feature standardization folded into raw nine-input weights',
        'objective': 'imitate unchanged sensory teacher on existing train samples',
        'optimizer': 'Adam full batch', 'learning_rate': 0.05,
        'optimizer_steps': MAX_UPDATES, 'optimizer_step_ceiling': MAX_UPDATES,
        'sample_count': len(dataset.features),
        'sample_ticks_per_case': 200, 'sample_counts': dataset.case_counts,
        'network_seed': config.network_seed, 'network_size': 100,
        'topology_sha256': topology_before,
        'feature_mean': outcome.mean.tolist(), 'feature_scale': outcome.scale.tolist(),
        'gradient_nonzero_scalars': outcome.gradient_nonzero_scalars,
        'gradient_min_abs': outcome.gradient_min_abs,
        'parameter_changed_scalars': outcome.parameter_changed_scalars,
        'parameter_change_min_abs': outcome.parameter_change_min_abs,
        'mean_command': mean_command.tolist(),
        'constant_mse_forward_turn': constant_error.tolist(),
        'readout_mse_forward_turn': model_error.tolist(),
        'turn_mse_reduction_vs_constant_fraction': float(1 - model_error[1] / constant_error[1]),
        'checkpoint': checkpoint.name,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'gate': gate, 'train_summary': summarize(train_rows),
        'train_results': [asdict(row) for row in train_rows],
        'cases': [asdict(case) for case in cases], 'config': asdict(config),
        'software': {'python': sys.version.split()[0], 'torch': torch.__version__,
                     'pygame': version('pygame'), 'platform': platform.platform(),
                     'device': 'cpu'},
        'validation_cases_evaluated': False, 'final_cases_evaluated': False,
    }
    with (output_dir / 'curve.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=('step', 'forward_mse', 'turn_mse'))
        writer.writeheader()
        writer.writerows(outcome.curve)
    with (output_dir / 'train_episodes.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=tuple(asdict(train_rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in train_rows)
    result_path = output_dir / 'results.json'
    result_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')
    return result_path


def main(argv: list[str] | None = None) -> int:
    """Run the single predeclared M6 recovery fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/m6-readout-recovery'))
    args = parser.parse_args(argv)
    path = run_recovery(args.output_dir)
    result = json.loads(path.read_text(encoding='utf-8'))
    print(f"Wrote {path}: turn MSE {result['readout_mse_forward_turn'][1]:.6f}, "
          f"train arrivals {result['train_summary']['successes']}/8, "
          f"gate passed {result['gate']['passed']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
