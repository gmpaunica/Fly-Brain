"""One independent 750-update test of the failed FB-10 covered-data gate."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from flybrain.neural.banc_projection import ProjectionReadout, load_checkpoint, save_checkpoint
from .banc_coverage import ORIGINAL_TURN_CEILING, coverage_gates
from .banc_frozen_evaluation import PINNED_CANDIDATE, training_evidence
from .banc_projection import evaluate_models
from .banc_protocol import sha256, software_record
from .banc_readiness import PROJECT_ROOT, capture_provenance, save_new_json, verify_provenance
from .evaluation import EvaluationConfig, _git_commit, locked_cases
from .records import ensure_new_run_directory


DEFAULT_OUTPUT = PROJECT_ROOT / 'experiments/fb10-extra-training'
OLD_CANDIDATE = PINNED_CANDIDATE
OLD_DIRECTORY = PROJECT_ROOT / 'experiments/fb10-projection-readout'
MAX_UPDATES = 750
LEARNING_RATE = .01


def fit(model: ProjectionReadout, features: Tensor, targets: Tensor, *,
        updates: int = MAX_UPDATES) -> dict:
    """Fit the same full-batch Adam objective with one declared update ceiling."""
    if type(updates) is not int or not 1 <= updates <= MAX_UPDATES:
        raise ValueError('updates must be an integer from 1 through 750')
    if (not isinstance(features, Tensor) or features.ndim != 2
            or not isinstance(targets, Tensor) or targets.shape != (len(features), 2)
            or targets.device.type != 'cpu' or not targets.is_floating_point()
            or not torch.isfinite(targets).all() or torch.any(targets[:, 0] < 0)
            or torch.any(targets[:, 0] > 1) or torch.any(targets[:, 1].abs() > 1)):
        raise ValueError('finite [N,20] features and bounded [N,2] targets required')
    model(features)  # Check the existing neural-rate and feature contract.
    x, y = features.detach().float(), targets.detach().float()
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    curve: list[dict] = []
    completed, nonzero, failure = 0, 0, None
    gradient_norm = None
    for update in range(updates + 1):
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
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
        gradients = [parameter.grad for parameter in model.parameters()]
        if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
            failure = 'missing or nonfinite readout gradients'
            break
        nonzero = max(nonzero, sum(int(torch.count_nonzero(gradient)) for gradient in gradients))
        gradient_norm = math.sqrt(sum(float(gradient.double().square().sum())
                                      for gradient in gradients))
        optimizer.step()
        completed += 1
    changed = sum(int(torch.count_nonzero(parameter.detach() != start))
                  for parameter, start in zip(model.parameters(), initial))
    return {'optimizer_steps': completed, 'gradient_nonzero_scalars': nonzero,
            'parameter_changed_scalars': changed, 'curve': curve, 'failure': failure}


def protocol_definition() -> dict:
    """Reconstruct the one hypothesis from pinned old fit and train-only data."""
    old, old_digest, _, evidence = training_evidence()
    if evidence['covered_gates']['passed'] is not False:
        raise ValueError('expected historical covered fit failure')
    return {'schema': 'fb10_extra_training_v1', 'experiment': 'fb10_extra_training',
        'hypothesis': 'The same readout and full-batch Adam configuration pass the extra covered-data error ceiling with 750 total updates from seed 1010.',
        'old_training_protocol_sha256': old_digest, 'candidate_sha256': OLD_CANDIDATE,
        'original_failure': evidence['failed_gate'],
        'data_sha256': old['collection_sha256'], 'contract': old['contract'],
        'normalization': old['normalization'], 'train_only_baseline': old['train_only_baseline'],
        'case_counts': old['case_counts'], 'sample_count': old['sample_count'],
        'seed': 1010, 'network_size': 83, 'trainable_parameters': 370,
        'optimizer': 'Adam full batch', 'learning_rate': LEARNING_RATE,
        'optimizer_step_ceiling': MAX_UPDATES, 'train_only': True,
        'objective': old['objective'], 'cases': old['cases'],
        'world_config': old['world_config'], 'software': software_record(),
        'budget': {'optimizer_updates': 750, 'episodes': 25,
                   'world_ticks': 15000, 'neural_steps': 300000},
        'gates': old['gates'],
        'stop_condition': 'Exactly one 750-update candidate from seed 1010; stop immediately on nonfinite numerics, record original matched training behavior after a finite fit, close on any failed gate. No sweep/retry or fresh case in this fit.'}


def prepare(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    """Write the single-use hypothesis, budget and provenance before training."""
    out = Path(output_dir).resolve()
    ensure_new_run_directory(out)
    definition = protocol_definition()
    record = {**definition, 'output_dir': str(out),
              'prepared_utc': datetime.now(timezone.utc).isoformat(),
              'git_commit': _git_commit(), 'provenance': capture_provenance(PROJECT_ROOT, out)}
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'protocol.json'
    save_new_json(path, record)
    with path.with_suffix('.sha256').open('x', encoding='utf-8') as stream:
        stream.write(sha256(path) + '\n')
    return path


def load_protocol(path: Path) -> tuple[dict, str]:
    """Reject any edited definition, stale source/history or relocated run."""
    path = Path(path).resolve()
    digest = sha256(path)
    if path.with_suffix('.sha256').read_text(encoding='utf-8').strip() != digest:
        raise ValueError('extra training protocol hash differs')
    record = json.loads(path.read_text(encoding='utf-8'))
    if path != Path(record['output_dir']).resolve() / 'protocol.json':
        raise ValueError('extra training protocol destination differs')
    inventory = capture_provenance(PROJECT_ROOT, path.parent)
    recorded = record.get('provenance', {})
    if (set(inventory) != set(recorded) or any(
            not recorded[group] or set(recorded[group]) != set(files)
            for group, files in inventory.items()) or not verify_provenance(recorded)):
        raise ValueError('extra training source/history differs')
    if any(record.get(key) != value for key, value in protocol_definition().items()):
        raise ValueError('extra training protocol definition differs')
    return record, digest


def may_evaluate_fresh(result: dict) -> bool:
    """Require a real boolean passing training gate before a separate assay."""
    value = result.get('training_gate_passed')
    if type(value) is not bool:
        raise ValueError('boolean training gate required')
    return value


def run(path: Path = DEFAULT_OUTPUT / 'protocol.json') -> dict:
    """Consume one prepared fit and save the complete negative or positive result."""
    path = Path(path).resolve()
    out = path.parent
    if any((out / name).exists() for name in (
            'execution.json', 'results.json', 'aborted.json', 'trained.pt',
            'untrained.pt', 'curve.csv', 'episodes.csv')):
        raise FileExistsError('extra training attempt consumed; no retry')
    protocol, digest = load_protocol(path)
    save_new_json(out / 'execution.json', {'status': 'consumed',
        'protocol_sha256': digest, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'stop_rule': protocol['stop_condition']})
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _execute(path, protocol, digest)
    except Exception as error:
        save_new_json(out / 'aborted.json', {'protocol_sha256': digest,
            'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        torch.set_num_threads(previous_threads)


def _execute(path: Path, protocol: dict, digest: str) -> dict:
    """Keep optimization distinct from embodied training-case verification."""
    _, _, _, evidence = training_evidence()
    from .banc_projection_fit import load_training_protocol
    _, _, data = load_training_protocol(OLD_DIRECTORY / 'protocol.json')
    mean = torch.tensor(protocol['normalization']['mean'])
    scale = torch.tensor(protocol['normalization']['scale'])
    trained = ProjectionReadout(mean, scale, seed=protocol['seed'])
    untrained = ProjectionReadout(mean, scale, seed=protocol['seed'])
    save_checkpoint(path.parent / 'untrained.pt', untrained, protocol_sha256=digest)
    fitted = fit(trained, data.features, data.targets)
    result = {'experiment': protocol['experiment'], 'protocol_sha256': digest,
        'software': software_record(), 'git_commit': protocol['git_commit'],
        'seed': protocol['seed'], 'network_size': 83, 'learning_rate': LEARNING_RATE,
        'objective': protocol['objective'], 'sample_count': len(data.features),
        'historical_covered_failure': evidence['failed_gate'],
        **fitted, 'training_gate_passed': False, 'held_out_evaluated': False,
        'actual_world_ticks': 0, 'actual_neural_steps': 0, 'actual_episodes': 0,
        'checkpoint': None}
    if fitted['curve']:
        with (path.parent / 'curve.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fitted['curve'][0]))
            writer.writeheader()
            writer.writerows(fitted['curve'])
    if fitted['failure'] is None:
        old_result = json.loads((OLD_DIRECTORY / 'results.json').read_text(encoding='utf-8'))
        if fitted['curve'][500] != old_result['curve'][500]:
            raise ValueError('original 500-update optimization prefix differs')
        result['historical_500_update_prefix_exact'] = True
        save_checkpoint(path.parent / 'trained.pt', trained, protocol_sha256=digest)
        reloaded = load_checkpoint(path.parent / 'trained.pt', protocol_sha256=digest,
                                   mean=mean, scale=scale)
        with torch.no_grad():
            error = (trained(data.features).double() - data.targets).square()
            errors = error.mean(0)
            original_errors = error[:1008].mean(0)
            exact = torch.equal(trained(data.features), reloaded(data.features))
        behavior = evaluate_models(trained, reloaded, untrained,
                                   locked_cases()['train'], EvaluationConfig())
        result.update(behavior)
        checks = {
            'checkpoint_outputs_exact': exact,
            'checkpoint_replay_exact': behavior['checkpoint_replay_exact'],
            'effective_weights_unchanged': behavior['effective_weights_unchanged'],
            'execution_budget_passed': behavior['execution_budget_passed']
                and behavior['actual_episodes'] == 25
                and behavior['actual_world_ticks'] <= 15000
                and behavior['actual_neural_steps'] <= 300000,
            'optimizer_budget_passed': fitted['optimizer_steps'] == MAX_UPDATES,
            'gradients_and_updates_observed': fitted['gradient_nonzero_scalars'] > 0
                and fitted['parameter_changed_scalars'] > 0,
            'normalization_unchanged': torch.equal(trained.mean, mean)
                and torch.equal(trained.scale, scale),
            'provenance_preserved': verify_provenance(protocol['provenance']),
            'protocol_unchanged': sha256(path) == digest,
        }
        gates = coverage_gates(float(errors[1]), float(original_errors[1]),
                               protocol['train_only_baseline']['constant_mse'][1],
                               behavior['summaries'], checks)
        result.update({'checks': checks, 'gates': gates,
            'per_output_mse': errors.tolist(),
            'original_per_output_mse': original_errors.tolist(),
            'constant_command_mse': protocol['train_only_baseline']['constant_mse'],
            'training_gate_passed': gates['passed'], 'checkpoint': 'trained.pt',
            'checkpoint_sha256': sha256(path.parent / 'trained.pt')})
        from .banc_training import _write_csv
        _write_csv(path.parent / 'episodes.csv', [
            dict(condition=name, **row)
            for name, group in behavior['episodes'].items() for row in group
        ] + [dict(condition='trained_replay', **behavior['replay_episode'])])
    save_new_json(path.parent / 'results.json', result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--fit', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    if args.prepare:
        path = prepare(args.output_dir)
        print(json.dumps({'protocol': str(path), 'sha256': sha256(path),
                          'optimizer_steps': 0}))
        return 0
    result = run(args.protocol or args.output_dir / 'protocol.json')
    print(json.dumps({key: result[key] for key in (
        'training_gate_passed', 'optimizer_steps', 'actual_world_ticks',
        'actual_neural_steps')}, indent=2))
    return 0 if result['training_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
