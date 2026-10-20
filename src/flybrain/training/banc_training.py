"""Execute one separately invoked frozen FB-10 training attempt; no held-out run."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import json

import torch
from torch import Tensor

from flybrain.neural.banc_motor import make_banc_controller
from flybrain.neural.banc_network import load_verified_banc_config
from flybrain.neural.banc_readout import (BancReadout, load_checkpoint,
                                        make_readout_controller, save_checkpoint)
from .banc_protocol import (DEFAULT_OUTPUT, LEARNING_RATE, MAX_UPDATES,
                            load_protocol, sha256, software_record)
from .banc_readiness import save_new_json, verify_provenance, weights_sha256
from .evaluation import EpisodeCase, EvaluationConfig, evaluate_cases, locked_cases, run_episode, summarize


def fit_readout(model: BancReadout, features: Tensor, targets: Tensor, *,
                updates: int = MAX_UPDATES, learning_rate: float = LEARNING_RATE) -> dict:
    """Fit detached features only; return actual updates and numerical evidence.

    Short budgets exist for artificial-data software tests. The experiment
    runner always consumes the frozen 500-update protocol, never a CLI override.
    """
    if (type(updates) is not int or not 1 <= updates <= MAX_UPDATES
            or isinstance(learning_rate, bool) or not math.isfinite(learning_rate)
            or learning_rate <= 0):
        raise ValueError('fit requires 1..500 updates and a finite positive learning rate')
    if not isinstance(features, Tensor) or features.ndim != 2:
        raise ValueError('fit requires a matrix of training samples')
    model(features)  # Validate rates before starting an optimizer.
    if (targets.device.type != 'cpu' or not targets.is_floating_point()
            or targets.shape != (len(features), 2) or not torch.isfinite(targets).all()
            or torch.any(targets[:, 0] < 0) or torch.any(targets[:, 0] > 1)
            or torch.any(targets[:, 1].abs() > 1)):
        raise ValueError('expected finite CPU [N,2] bounded training commands')
    x, y = features.detach().float(), targets.detach().float()
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    curve, completed, nonzero, failure = [], 0, 0, None
    gradient_norm = None
    for update in range(updates + 1):
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            failure = 'nonfinite readout parameters'
            break
        errors = (model(x) - y).square().mean(dim=0)
        loss = errors.sum()
        if not torch.isfinite(loss):
            failure = 'nonfinite readout loss'
            break
        curve.append({'update': update, 'forward_mse': float(errors[0].detach()),
                      'turn_mse': float(errors[1].detach()), 'loss': float(loss.detach()),
                      'gradient_l2_from_previous_update': gradient_norm})
        if update == updates:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad for p in model.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            failure = 'missing or nonfinite readout gradients'
            break
        nonzero = max(nonzero, sum(int(torch.count_nonzero(g)) for g in gradients))
        gradient_norm = math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
        optimizer.step()
        completed += 1
    changed = sum(int(torch.count_nonzero(p.detach() != start))
                  for p, start in zip(model.parameters(), initial))
    return {'optimizer_steps': completed, 'gradient_nonzero_scalars': nonzero,
            'parameter_changed_scalars': changed, 'curve': curve, 'failure': failure}


def evaluate_training_models(trained: BancReadout, reloaded: BancReadout,
                             untrained: BancReadout, cases: Sequence[EpisodeCase],
                             config: EvaluationConfig) -> dict:
    """Measure three matched train-only controls and one save/reload replay."""
    allowed = set(locked_cases()['train'])
    if not cases or len(set(cases)) != len(cases) or any(case not in allowed for case in cases):
        raise ValueError('behavior checks require training cases')
    controllers, initial_weights = [], []
    def factory(model: BancReadout | None):
        def create():
            controller = (make_readout_controller(model) if model is not None
                          else make_banc_controller(load_verified_banc_config()))
            initial_weights.append(weights_sha256(controller))
            controllers.append(controller)
            return controller
        return create
    with torch.no_grad():
        episodes = {
            'trained': evaluate_cases(cases, config, factory(reloaded)),
            'untrained': evaluate_cases(cases, config, factory(untrained)),
            'fixed': evaluate_cases(cases, config, factory(None)),
        }
        replay = run_episode(cases[0], config, factory(trained))
    ticks = sum(row.ticks for group in episodes.values() for row in group) + replay.ticks
    neural_steps = sum(controller.network.population.step_count for controller in controllers)
    return {
        'episodes': {name: [asdict(row) for row in group] for name, group in episodes.items()},
        'summaries': {name: summarize(group) for name, group in episodes.items()},
        'replay_episode': asdict(replay), 'actual_world_ticks': ticks,
        'actual_neural_steps': neural_steps, 'actual_episodes': len(controllers),
        'execution_budget_passed': (len(controllers) == 3 * len(cases) + 1
            and ticks <= (3 * len(cases) + 1) * config.horizon_ticks
            and neural_steps == ticks * 20),
        'checkpoint_replay_exact': replay == episodes['trained'][0],
        'effective_weights_unchanged': (len(set(initial_weights)) == 1
            and all(weights_sha256(c) == digest for c, digest in zip(controllers, initial_weights))),
    }


def training_gates(turn_mse: float, constant_turn_mse: float,
                    summaries: dict, checks: dict[str, bool]) -> dict:
    """Turn fit and strict matched behavioral improvement must both pass."""
    trained = summaries['trained']
    controls = [summaries[name] for name in ('untrained', 'fixed')]
    gates = {
        'turn_fit': (math.isfinite(turn_mse) and constant_turn_mse > 0
                     and turn_mse <= .25 * constant_turn_mse),
        'training_arrivals': trained['successes'] >= 7,
        'matched_arrival_improvement': all(trained['successes'] > c['successes'] for c in controls),
        'matched_progress_improvement': all(trained['mean_progress_fraction'] > c['mean_progress_fraction']
                                             for c in controls),
        'numerical_and_integrity_checks': bool(checks) and all(checks.values()),
    }
    gates['passed'] = all(gates.values())
    return gates


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_training(protocol_path: Path) -> dict:
    """Consume one prepared protocol before optimization; preserve every attempt."""
    protocol_path = Path(protocol_path).resolve()
    output = protocol_path.parent
    # Check before loading evidence: an attempt is consumed even if it crashed.
    if any((output / name).exists() for name in (
            'execution.json', 'results.json', 'untrained.pt', 'trained.pt', 'curve.csv', 'episodes.csv')):
        raise FileExistsError('FB-10 attempt already consumed or has partial artifacts; no automatic retry')
    protocol, digest, dataset = load_protocol(protocol_path)
    save_new_json(output / 'execution.json', {'protocol_sha256': digest,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'consumed',
        'stop_rule': protocol['stop_condition']})
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _execute(protocol_path, protocol, digest, dataset)
    finally:
        torch.set_num_threads(previous_threads)


def _execute(path: Path, protocol: dict, digest: str, dataset) -> dict:
    output = path.parent
    mean = torch.tensor(protocol['normalization']['mean'], dtype=torch.float32)
    scale = torch.tensor(protocol['normalization']['scale'], dtype=torch.float32)
    model = BancReadout(mean, scale, seed=protocol['seed'])
    untrained = BancReadout(mean, scale, seed=protocol['seed'])
    save_checkpoint(output / 'untrained.pt', untrained, protocol_sha256=digest)
    fit = fit_readout(model, dataset.features, dataset.targets)
    result = {'experiment': protocol['experiment'], 'protocol_sha256': digest,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'software': software_record(),
        'git_commit': protocol['git_commit'], 'network_size': 83, 'seed': protocol['seed'],
        'objective': protocol['objective'], 'learning_rate': LEARNING_RATE,
        'sample_count': len(dataset.features), **fit,
        'actual_world_ticks': 0, 'actual_neural_steps': 0, 'actual_episodes': 0,
        'checkpoint': None, 'held_out_evaluated': False, 'training_gate_passed': False}
    if fit['curve']:
        _write_csv(output / 'curve.csv', fit['curve'])
    if fit['failure'] is None:
        save_checkpoint(output / 'trained.pt', model, protocol_sha256=digest)
        reloaded = load_checkpoint(output / 'trained.pt', protocol_sha256=digest, mean=mean, scale=scale)
        with torch.no_grad():
            errors = (model(dataset.features).double() - dataset.targets).square().mean(0)
            exact = torch.equal(model(dataset.features), reloaded(dataset.features))
        cases = tuple(EpisodeCase(**case) for case in protocol['cases'])
        behavior = evaluate_training_models(model, reloaded, untrained, cases,
                                             EvaluationConfig(**protocol['world_config']))
        result.update(behavior)
        checks = {
            'checkpoint_outputs_exact': exact,
            'checkpoint_replay_exact': behavior['checkpoint_replay_exact'],
            'effective_weights_unchanged': behavior['effective_weights_unchanged'],
            'execution_budget_passed': (behavior['execution_budget_passed']
                and behavior['actual_world_ticks'] <= protocol['max_world_ticks']
                and behavior['actual_neural_steps'] <= protocol['max_neural_steps']
                and behavior['actual_episodes'] == protocol['max_episodes']),
            'optimizer_budget_passed': fit['optimizer_steps'] == MAX_UPDATES,
            'gradient_and_update_observed': fit['gradient_nonzero_scalars'] > 0
                and fit['parameter_changed_scalars'] > 0,
            'normalization_unchanged': torch.equal(model.mean, mean) and torch.equal(model.scale, scale),
            'provenance_preserved': verify_provenance(protocol['provenance']),
            'protocol_unchanged': sha256(path) == digest,
        }
        gates = training_gates(float(errors[1]),
            protocol['train_only_baseline']['constant_mse'][1], behavior['summaries'], checks)
        result.update({'checks': checks, 'gates': gates, 'training_gate_passed': gates['passed'],
            'per_output_mse': errors.tolist(), 'constant_command_mse': protocol['train_only_baseline']['constant_mse'],
            'checkpoint': 'trained.pt', 'checkpoint_sha256': sha256(output / 'trained.pt')})
        _write_csv(output / 'episodes.csv', [dict(condition=name, **row)
            for name, group in behavior['episodes'].items() for row in group]
            + [dict(condition='trained_replay', **behavior['replay_episode'])])
    else:
        result['checks'] = {'provenance_preserved': verify_provenance(protocol['provenance']),
                            'protocol_unchanged': sha256(path) == digest}
    save_new_json(output / 'results.json', result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_OUTPUT / 'protocol.json')
    args = parser.parse_args(argv)
    result = run_training(args.protocol)
    print(json.dumps({key: result[key] for key in (
        'optimizer_steps', 'failure', 'training_gate_passed', 'actual_world_ticks', 'held_out_evaluated')}, indent=2))
    return 0 if result['training_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
