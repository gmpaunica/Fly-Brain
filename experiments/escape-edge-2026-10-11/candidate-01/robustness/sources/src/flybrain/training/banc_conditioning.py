"""One predeclared conditioning repair fit; preserve the original FB-10 failure."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json

import torch

from flybrain.neural.banc_conditioning import (ConditionedBancReadout,
    conditioned_contract, normalization, save_checkpoint, load_checkpoint)
from .banc_protocol import (DEFAULT_READINESS, SEED, MAX_UPDATES, LEARNING_RATE,
                            load_readiness, sha256, software_record)
from .banc_readiness import (PROJECT_ROOT, capture_provenance, verify_provenance,
                             save_new_json, summarize_features)
from .banc_training import fit_readout, evaluate_training_models, training_gates, _write_csv
from .evaluation import EpisodeCase, EvaluationConfig, locked_cases, _git_commit
from .records import ensure_new_run_directory


DEFAULT_OUTPUT = Path('experiments/fb10-conditioned-readout')


def protocol_definition(dataset) -> dict:
    """Return one fixed candidate and the unchanged original scientific gates."""
    mean, scale = normalization(dataset.features)
    metrics = summarize_features(dataset.features, dataset.targets)
    return {'schema': 'fb10_conditioned_training_v1', 'experiment': 'fb10_conditioned_readout',
        'hypothesis': 'Separately normalized common activity and bilateral contrast allow the fixed endpoint readout to pass the original fit and training-behavior gates.',
        'seed': SEED, 'contract': conditioned_contract(), 'network_size': 83,
        'trainable_parameters': 178, 'software': software_record(),
        'optimizer': 'Adam full batch', 'learning_rate': LEARNING_RATE,
        'optimizer_step_ceiling': MAX_UPDATES, 'sample_count': len(dataset.features),
        'objective': 'forward MSE + turn MSE; no gradient through spikes or conditioning',
        'normalization': {'mean': mean.tolist(), 'scale': scale.tolist()},
        'train_only_baseline': metrics, 'cases': [asdict(c) for c in locked_cases()['train']],
        'world_config': asdict(EvaluationConfig()), 'max_world_ticks': 15000,
        'max_neural_steps': 300000, 'max_episodes': 25,
        'gates': {'turn_mse_max': .25 * metrics['constant_mse'][1], 'trained_arrivals_min': 7,
            'strict_matched_arrival_and_progress_improvement': True,
            'finite_gradients_updates_frozen_core_normalization_and_exact_replay': True},
        'stop_condition': 'One 500-update candidate; stop on nonfinite numerics. Record matched training controls after a finite fit, then close on failed gates. No sweep, retry or held-out stepping in this runner.'}


def prepare_protocol(output_dir: Path) -> Path:
    """Freeze one destination-bound protocol without optimization or simulation."""
    output = Path(output_dir).resolve()
    ensure_new_run_directory(output)
    dataset = load_readiness(DEFAULT_READINESS)
    payload = {**protocol_definition(dataset), 'output_dir': str(output),
        'prepared_utc': datetime.now(timezone.utc).isoformat(), 'git_commit': _git_commit(),
        'readiness_sha256': {name: sha256(DEFAULT_READINESS / name)
                            for name in ('protocol.json', 'samples.csv', 'results.json')},
        'provenance': capture_provenance(PROJECT_ROOT, output)}
    output.mkdir(parents=True)
    path = output / 'protocol.json'; save_new_json(path, payload)
    with path.with_suffix('.sha256').open('x', encoding='utf-8') as stream:
        stream.write(sha256(path) + '\n')
    return path


def load_protocol(path: Path):
    """Reject stale, resealed, relocated or scientifically changed protocols."""
    path = Path(path).resolve(); digest = sha256(path)
    if path.with_suffix('.sha256').read_text(encoding='utf-8').strip() != digest:
        raise ValueError('protocol hash differs')
    p = json.loads(path.read_text(encoding='utf-8'))
    if path != Path(p['output_dir']).resolve() / 'protocol.json':
        raise ValueError('protocol destination differs')
    if any(sha256(DEFAULT_READINESS / name) != h for name, h in p['readiness_sha256'].items()):
        raise ValueError('original readiness bytes differ')
    data = load_readiness(DEFAULT_READINESS)
    if any(p.get(key) != value for key, value in protocol_definition(data).items()):
        raise ValueError('candidate/normalization/gates/environment differ')
    if not verify_provenance(p['provenance']):
        raise ValueError('source/circuit/history provenance differs')
    return p, digest, data


def run_training(path: Path) -> dict:
    """Consume a prepared repair once; record measured gates, including failures."""
    path = Path(path).resolve(); output = path.parent
    if any((output / name).exists() for name in ('execution.json', 'results.json',
        'untrained.pt', 'trained.pt', 'curve.csv', 'episodes.csv')):
        raise FileExistsError('conditioned attempt already consumed; no retry')
    p, digest, data = load_protocol(path)
    save_new_json(output / 'execution.json', {'status': 'consumed', 'protocol_sha256': digest,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'stop_rule': p['stop_condition']})
    previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        return _execute(path, p, digest, data)
    except Exception as error:
        save_new_json(output / 'aborted.json', {'type': type(error).__name__, 'error': str(error),
            'protocol_sha256': digest, 'instruction': 'preserve partial attempt; no automatic retry'})
        raise
    finally:
        torch.set_num_threads(previous_threads)


def _execute(path: Path, p: dict, digest: str, data) -> dict:
    output = path.parent
    mean = torch.tensor(p['normalization']['mean']); scale = torch.tensor(p['normalization']['scale'])
    model, untrained = [ConditionedBancReadout(mean, scale, seed=p['seed']) for _ in range(2)]
    save_checkpoint(output / 'untrained.pt', untrained, protocol_sha256=digest)
    fit = fit_readout(model, data.features, data.targets)
    result = {'experiment': p['experiment'], 'protocol_sha256': digest,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'software': software_record(),
        'git_commit': p['git_commit'], 'network_size': 83, 'seed': p['seed'],
        'objective': p['objective'], 'learning_rate': LEARNING_RATE, 'sample_count': len(data.features),
        **fit, 'held_out_evaluated': False, 'training_gate_passed': False,
        'actual_world_ticks': 0, 'actual_neural_steps': 0, 'actual_episodes': 0, 'checkpoint': None}
    if fit['curve']:
        _write_csv(output / 'curve.csv', fit['curve'])
    if fit['failure'] is None:
        save_checkpoint(output / 'trained.pt', model, protocol_sha256=digest)
        loaded = load_checkpoint(output / 'trained.pt', protocol_sha256=digest, mean=mean, scale=scale)
        with torch.no_grad():
            errors = (model(data.features).double() - data.targets).square().mean(0)
            exact = torch.equal(model(data.features), loaded(data.features))
        cases = tuple(EpisodeCase(**c) for c in p['cases'])
        behavior = evaluate_training_models(model, loaded, untrained, cases, EvaluationConfig(**p['world_config']))
        result.update(behavior)
        checks = {'checkpoint_outputs_exact': exact, 'checkpoint_replay_exact': behavior['checkpoint_replay_exact'],
            'effective_weights_unchanged': behavior['effective_weights_unchanged'],
            'execution_budget_passed': behavior['execution_budget_passed']
                and behavior['actual_world_ticks'] <= p['max_world_ticks']
                and behavior['actual_neural_steps'] <= p['max_neural_steps']
                and behavior['actual_episodes'] == p['max_episodes'],
            'optimizer_budget_passed': fit['optimizer_steps'] == MAX_UPDATES,
            'gradients_and_updates_observed': fit['gradient_nonzero_scalars'] > 0 and fit['parameter_changed_scalars'] > 0,
            'normalization_unchanged': torch.equal(model.mean, mean) and torch.equal(model.scale, scale),
            'provenance_preserved': verify_provenance(p['provenance']), 'protocol_unchanged': sha256(path) == digest}
        gates = training_gates(float(errors[1]), p['train_only_baseline']['constant_mse'][1], behavior['summaries'], checks)
        result.update({'checks': checks, 'gates': gates, 'per_output_mse': errors.tolist(),
            'constant_command_mse': p['train_only_baseline']['constant_mse'],
            'training_gate_passed': gates['passed'], 'checkpoint': 'trained.pt',
            'checkpoint_sha256': sha256(output / 'trained.pt')})
        _write_csv(output / 'episodes.csv', [dict(condition=name, **row)
            for name, group in behavior['episodes'].items() for row in group]
            + [dict(condition='trained_replay', **behavior['replay_episode'])])
    else:
        result['checks'] = {'provenance_preserved': verify_provenance(p['provenance']),
                            'protocol_unchanged': sha256(path) == digest}
    save_new_json(output / 'results.json', result)
    return result


def main() -> int:
    """Prepare with zero updates, or separately execute the single frozen fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_OUTPUT / 'protocol.json')
    args = parser.parse_args()
    if args.prepare:
        path = prepare_protocol(args.output_dir)
        print(json.dumps({'protocol': str(path), 'sha256': sha256(path), 'updates': 0, 'world_ticks': 0}))
        return 0
    result = run_training(args.protocol)
    print(json.dumps({key: result[key] for key in ('optimizer_steps', 'failure', 'training_gate_passed', 'actual_world_ticks')}, indent=2))
    return 0 if result['training_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
