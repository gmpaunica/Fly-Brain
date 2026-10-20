"""One fresh, zero-update assay for the separately passing 750-update head."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

import torch

from flybrain.neural.banc_network import load_verified_banc_config
from flybrain.neural.banc_motor import make_banc_controller
from flybrain.neural.banc_projection import load_checkpoint, make_projection_controller
from .banc_coverage import coverage_gates
from .banc_extra_training import DEFAULT_OUTPUT as CANDIDATE
from .banc_frozen_evaluation import historical_seeds
from .banc_memory_evaluation import assess_split
from .banc_protocol import sha256, software_record
from .banc_readiness import (PROJECT_ROOT, capture_provenance, save_new_json,
                             verify_provenance, weights_sha256)
from .banc_training import _write_csv
from .banc_projection_fit import load_training_protocol
from .evaluation import (EpisodeCase, EvaluationConfig, _git_commit,
                         locked_cases, run_episode, summarize)
from .records import ensure_new_run_directory


DEFAULT_OUTPUT = PROJECT_ROOT / 'experiments/fb10-extra-evaluation'
CENTRAL_RECEIPT = CANDIDATE / 'fresh-evaluation.execution.json'
CONTROLLERS = ('trained', 'untrained', 'fixed')
MAX_EPISODES, MAX_WORLD_TICKS, MAX_NEURAL_STEPS = 144, 86400, 1728000


def fresh_cases() -> dict[str, tuple[EpisodeCase, ...]]:
    """Reserve unused starts before seeing any new 750-update outcome."""
    groups: dict[str, tuple[EpisodeCase, ...]] = {}
    for split, start, count in (('validation', 730000, 16), ('final', 731000, 32)):
        cases = []
        for seed in range(start, start + count):
            rng = random.Random(seed)
            angle = rng.uniform(0, math.tau)
            radius = math.sqrt(rng.uniform(60 ** 2, 180 ** 2))
            heading = rng.uniform(0, math.tau)
            cases.append(EpisodeCase(f'fb10-extra-{split}-{seed}', split, seed,
                600 + radius * math.cos(angle), 300 + radius * math.sin(angle), heading))
        groups[split] = tuple(cases)
    return groups


def training_evidence() -> tuple[dict, str, dict, dict]:
    """Recompute new training gates from strict checkpoint with zero stepping."""
    candidate_files = ('protocol.json', 'results.json', 'trained.pt', 'untrained.pt')
    if any(not (CANDIDATE / name).is_file() for name in candidate_files):
        raise ValueError('passing extra-training candidate missing')
    p_path = CANDIDATE / 'protocol.json'
    digest = sha256(p_path)
    if p_path.with_suffix('.sha256').read_text(encoding='utf-8').strip() != digest:
        raise ValueError('extra-training protocol hash differs')
    p = json.loads(p_path.read_text(encoding='utf-8'))
    r = json.loads((CANDIDATE / 'results.json').read_text(encoding='utf-8'))
    if (r.get('training_gate_passed') is not True
            or r.get('historical_500_update_prefix_exact') is not True
            or r.get('optimizer_steps') != 750 or r.get('failure') is not None
            or r.get('held_out_evaluated') is not False
            or r.get('protocol_sha256') != digest
            or sha256(CANDIDATE / 'trained.pt') != r.get('checkpoint_sha256')
            or r.get('actual_episodes') != 25
            or r.get('actual_world_ticks', 15001) > 15000
            or r.get('actual_neural_steps') != 20 * r['actual_world_ticks']
            or set(r.get('checks', {})) != {
                'checkpoint_outputs_exact', 'checkpoint_replay_exact',
                'effective_weights_unchanged', 'execution_budget_passed',
                'optimizer_budget_passed', 'gradients_and_updates_observed',
                'normalization_unchanged', 'provenance_preserved',
                'protocol_unchanged'}
            or any(value is not True for value in r['checks'].values())
            or not verify_provenance(p['provenance'])):
        raise ValueError('passing training gate and complete integrity required')
    old, _, data = load_training_protocol(PROJECT_ROOT / 'experiments/fb10-projection-readout/protocol.json')
    if (p['old_training_protocol_sha256'] != sha256(PROJECT_ROOT / 'experiments/fb10-projection-readout/protocol.json')
            or p['normalization'] != old['normalization']
            or p['data_sha256'] != old['collection_sha256']):
        raise ValueError('extra-training data/normalization differs')
    mean = torch.tensor(p['normalization']['mean'])
    scale = torch.tensor(p['normalization']['scale'])
    models = {name: load_checkpoint(CANDIDATE / f'{name}.pt', protocol_sha256=digest,
                                   mean=mean, scale=scale)
              for name in ('trained', 'untrained')}
    initial = type(models['untrained'])(mean, scale, seed=1010)
    if any(not torch.equal(value, initial.state_dict()[key])
           for key, value in models['untrained'].state_dict().items()):
        raise ValueError('untrained initialization differs')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            errors = (models['trained'](data.features).double() - data.targets).square()
        covered = errors.mean(0).tolist()
        original = errors[:1008].mean(0).tolist()
    finally:
        torch.set_num_threads(previous)
    gates = coverage_gates(covered[1], original[1], p['train_only_baseline']['constant_mse'][1],
                           r['summaries'], r['checks'])
    if (covered != r['per_output_mse'] or original != r['original_per_output_mse']
            or gates != r['gates'] or not gates['passed']
            or len(data.features) != p['sample_count']
            or [(row['case_id'], row['split'], row['seed']) for row in r['episodes']['trained']]
               != [(c.case_id, c.split, c.seed) for c in locked_cases()['train']]):
        raise ValueError('passing training gate does not reproduce')
    for model in models.values():
        model.requires_grad_(False)
    return p, digest, models, r


def protocol_definition() -> dict:
    """Freeze unchanged fresh gates and an independent candidate identity."""
    p, digest, _, result = training_evidence()
    if result['training_gate_passed'] is not True:
        raise ValueError('passing training gate required')
    return {'schema': 'fb10_extra_fresh_evaluation_v1',
        'experiment': 'fb10_extra_fresh_evaluation',
        'hypothesis': 'The 750-update internal-spike readout improves fresh food arrival over the identical untrained initialization and fixed BANC decoder.',
        'training_gate_passed': True, 'training_protocol_sha256': digest,
        'candidate_sha256': {name: sha256(CANDIDATE / name) for name in
            ('protocol.json', 'results.json', 'trained.pt', 'untrained.pt')},
        'candidate_directory': str(CANDIDATE),
        'cases': {split: [asdict(case) for case in group]
                  for split, group in fresh_cases().items()},
        'config': p['world_config'], 'software': software_record(),
        'network_size': 83, 'trainable_parameters': 370,
        'optimizer_steps': 0, 'learning_rate': None,
        'objective': 'Paired fresh food arrival and all-case progress against identical untrained and fixed controls',
        'budget': {'optimizer_steps': 0, 'validation_episodes': 48,
                   'final_episodes_if_validation_passes': 96,
                   'maximum_episodes': MAX_EPISODES,
                   'maximum_world_ticks': MAX_WORLD_TICKS,
                   'maximum_neural_steps': MAX_NEURAL_STEPS},
        'gates': {'validation_arrivals_min': 13, 'final_arrivals_min': 26,
                  'final_paired_arrival_difference_min': 8,
                  'final_one_sided_p_max_per_control': .025,
                  'strict_all_case_progress_improvement': True,
                  'frozen_parameters_core_and_history': True},
        'stop_condition': 'One central receipt; stop before final if validation fails. No fitting, tuning, retry, case selection or historical-case reuse.'}


def prepare(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    """Create one unused fresh protocol without simulation or optimizer steps."""
    if CENTRAL_RECEIPT.exists():
        raise FileExistsError('extra-trained candidate fresh assay consumed')
    out = Path(output_dir).resolve()
    ensure_new_run_directory(out)
    definition = protocol_definition()
    seeds = {row['seed'] for group in definition['cases'].values() for row in group}
    if seeds.intersection(historical_seeds(exclude=out)):
        raise ValueError('fresh case seed overlaps historical record')
    p = {**definition, 'output_dir': str(out),
         'prepared_utc': datetime.now(timezone.utc).isoformat(),
         'git_commit': _git_commit(), 'provenance': capture_provenance(PROJECT_ROOT, out)}
    out.mkdir(parents=True)
    path = out / 'protocol.json'
    save_new_json(path, p)
    with path.with_suffix('.sha256').open('x', encoding='utf-8') as stream:
        stream.write(sha256(path) + '\n')
    return path


def load_protocol(path: Path) -> tuple[dict, str]:
    """Reject modified cases/gates, candidate, history, source or copied path."""
    path = Path(path).resolve()
    digest = sha256(path)
    if path.with_suffix('.sha256').read_text(encoding='utf-8').strip() != digest:
        raise ValueError('fresh evaluation protocol hash differs')
    p = json.loads(path.read_text(encoding='utf-8'))
    if path != Path(p['output_dir']).resolve() / 'protocol.json':
        raise ValueError('fresh evaluation output destination differs')
    inventory = capture_provenance(PROJECT_ROOT, path.parent)
    inventory['previous_files_sha256'].pop(str(CENTRAL_RECEIPT.resolve()), None)
    recorded = p.get('provenance', {})
    if (set(inventory) != set(recorded)
            or any(not recorded[group] or set(recorded[group]) != set(files)
                   for group, files in inventory.items())
            or not verify_provenance(recorded)):
        raise ValueError('fresh evaluation source/history differs')
    if any(p.get(key) != value for key, value in protocol_definition().items()):
        raise ValueError('fresh evaluation definition differs')
    seeds = {row['seed'] for group in p['cases'].values() for row in group}
    if seeds.intersection(historical_seeds(exclude=path.parent)):
        raise ValueError('fresh case seeds became historical')
    return p, digest


def run(path: Path = DEFAULT_OUTPUT / 'protocol.json') -> dict:
    """Consume the assay once; journal exact counters and stop after bad validation."""
    path = Path(path).resolve()
    out = path.parent
    if CENTRAL_RECEIPT.exists() or any((out / name).exists() for name in (
            'execution.json', 'results.json', 'episodes.jsonl', 'aborted.json')):
        raise FileExistsError('extra-trained fresh assay consumed; no retry')
    p, digest = load_protocol(path)
    _, _, models, _ = training_evidence()
    initial = {name: {key: value.clone() for key, value in model.state_dict().items()}
               for name, model in models.items()}
    receipt = {'status': 'consumed', 'protocol_sha256': digest,
               'output_dir': str(out), 'started_utc': datetime.now(timezone.utc).isoformat()}
    save_new_json(CENTRAL_RECEIPT, receipt)
    previous = torch.get_num_threads()
    completed = ticks = steps = 0
    results: dict = {}
    summaries: dict = {}
    gates: dict = {}
    active = None
    controller = None
    try:
        save_new_json(out / 'execution.json', receipt)
        torch.set_num_threads(1)
        with (out / 'episodes.jsonl').open('x', encoding='utf-8') as journal:
            for split in ('validation', 'final'):
                if split == 'final' and not gates['validation']['passed']:
                    break
                load_protocol(path)
                cases = tuple(EpisodeCase(**row) for row in p['cases'][split])
                rows = {name: [] for name in CONTROLLERS}
                for name in CONTROLLERS:
                    for case in cases:
                        active = {'condition': name, 'case': asdict(case)}
                        controller = (make_banc_controller(load_verified_banc_config())
                                      if name == 'fixed' else make_projection_controller(models[name]))
                        weight = weights_sha256(controller)
                        with torch.no_grad():
                            row = run_episode(case, EvaluationConfig(**p['config']), lambda: controller)
                        neural_steps = controller.network.population.step_count
                        completed += 1
                        ticks += row.ticks
                        steps += neural_steps
                        active = None
                        journal.write(json.dumps({'condition': name, 'result': asdict(row),
                                                  'neural_steps': neural_steps}, allow_nan=False) + '\n')
                        journal.flush()
                        if (neural_steps != row.ticks * 20 or weights_sha256(controller) != weight
                                or any(not torch.equal(value, initial[model_name][key])
                                       for model_name, model in models.items()
                                       for key, value in model.state_dict().items())):
                            raise RuntimeError('timing, frozen model or core differs')
                        if (completed > MAX_EPISODES or ticks > MAX_WORLD_TICKS
                                or steps > MAX_NEURAL_STEPS):
                            raise RuntimeError('fresh evaluation budget exceeded')
                        if (row.case_id, row.split, row.seed) != (case.case_id, case.split, case.seed):
                            raise RuntimeError('matched episode identity differs')
                        rows[name].append(row)
                        if len(rows[name]) % 4 == 0:
                            print(f'{split}/{name}: {len(rows[name])}/{len(cases)}', flush=True)
                summaries[split] = {name: summarize(group) for name, group in rows.items()}
                gates[split] = assess_split(split,
                    {name: [row.outcome == 'success' for row in group]
                     for name, group in rows.items()}, summaries[split])
                results[split] = {name: [asdict(row) for row in group]
                                  for name, group in rows.items()}
                _write_csv(out / f'{split}.csv', [dict(condition=name, **row)
                    for name, group in results[split].items() for row in group])
                save_new_json(out / f'{split}.json', {'gates': gates[split],
                    'summaries': summaries[split], 'results': results[split]})
                print(f"{split}: arrivals={gates[split]['successes']}; passed={gates[split]['passed']}", flush=True)
        load_protocol(path)
        record = {'experiment': p['experiment'], 'protocol_sha256': digest,
            'training_protocol_sha256': p['training_protocol_sha256'],
            'software': software_record(), 'git_commit': p['git_commit'],
            'network_size': 83, 'objective': p['objective'], 'optimizer_steps': 0,
            'learning_rate': None, 'checkpoint': str(CANDIDATE / 'trained.pt'),
            'checkpoint_sha256': sha256(CANDIDATE / 'trained.pt'),
            'actual_episodes': completed, 'actual_world_ticks': ticks,
            'actual_neural_steps': steps, 'results': results, 'summaries': summaries,
            'gates': gates, 'final_cases_evaluated': 'final' in results,
            'frozen_parameters_unchanged': True, 'effective_weights_unchanged': True,
            'integrity_checks_passed': True,
            'fresh_gate_passed': gates.get('final', {}).get('passed', False),
            'completed_utc': datetime.now(timezone.utc).isoformat()}
        save_new_json(out / 'results.json', record)
        return record
    except BaseException as error:
        partial_steps = (controller.network.population.step_count
                         if active is not None and controller is not None else 0)
        save_new_json(out / 'aborted.json', {'type': type(error).__name__,
            'error': str(error), 'protocol_sha256': digest,
            'completed_episodes': completed, 'completed_world_ticks': ticks,
            'completed_neural_steps': steps, 'in_progress': active,
            'in_progress_neural_steps': partial_steps,
            'actual_neural_steps': steps + partial_steps,
            'in_progress_world_ticks': None if active is not None else 0,
            'instruction': 'Consumed even if interrupted. Unknown partial world ticks are not zero. No retry.'})
        raise
    finally:
        torch.set_num_threads(previous)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_OUTPUT / 'protocol.json')
    args = parser.parse_args()
    if args.prepare:
        path = prepare(args.output_dir)
        print(json.dumps({'protocol': str(path), 'sha256': sha256(path),
                          'optimizer_steps': 0, 'world_ticks': 0}))
        return 0
    result = run(args.protocol)
    print(json.dumps({key: result[key] for key in (
        'fresh_gate_passed', 'actual_episodes', 'actual_world_ticks',
        'actual_neural_steps')}, indent=2))
    return 0 if result['fresh_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
