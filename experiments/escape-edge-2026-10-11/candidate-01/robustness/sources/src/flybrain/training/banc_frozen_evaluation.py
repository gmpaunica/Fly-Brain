"""Explicit original-gate exception for one frozen FB-10 assay; never a new fit.

The covered fit stays failed. This module only admits its exact frozen candidate
under a separately recorded user decision, retaining the original fresh gates.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from flybrain.neural.banc_projection import ProjectionReadout, load_checkpoint, make_projection_controller
from flybrain.neural.banc_motor import make_banc_controller
from flybrain.neural.banc_network import load_verified_banc_config
from .banc_projection import DEFAULT_OUTPUT as CANDIDATE, evaluate_group, coverage_gates
from .banc_projection_fit import load_training_protocol
from .banc_projection_evaluation import fresh_cases, CENTRAL_RECEIPT, DEFAULT_OUTPUT
from .banc_memory_evaluation import assess_split
from .banc_coverage import ORIGINAL_TURN_CEILING
from .banc_protocol import sha256, software_record
from .banc_readiness import PROJECT_ROOT, capture_provenance, verify_provenance, save_new_json, weights_sha256
from .banc_training import training_gates, _write_csv
from .evaluation import EpisodeCase, EpisodeResult, EvaluationConfig, locked_cases, run_episode, summarize, _git_commit
from .records import ensure_new_run_directory


CONTROLLERS = ('trained', 'untrained', 'fixed')
ACCEPTANCE_RULE = 'original_gates_with_covered_failure_retained'
PINNED_CANDIDATE = {
    'protocol.json': '769db7336a0adfb10e28d395dd2c64eb2a8da3aef091011e478c827a3fa91290',
    'results.json': '762a04fef20685c30ec5d4cb39968b5fe0a9c61419cfcb79523bfbd458bb860e',
    'trained.pt': '117f7e8b759c8916b33cd66a814e471a11cfc7aac941f71c51495dabeeddafa1',
    'untrained.pt': '1e5baa3a4b2ce13e3a9d0a0b6f435a8ecae71dba0a41b3104abd10aef05aa5b2',
}
REQUIRED_CHECKS = frozenset((
    'checkpoint_outputs_exact', 'checkpoint_replay_exact', 'effective_weights_unchanged',
    'execution_budget_passed', 'optimizer_budget_passed', 'gradients_and_updates_observed',
    'normalization_unchanged', 'provenance_preserved', 'protocol_unchanged',
))


def validate_checks(checks: dict) -> None:
    """Missing, false, or non-boolean integrity checks cannot pass vacuously."""
    if set(checks) != REQUIRED_CHECKS or any(v is not True for v in checks.values()):
        raise ValueError('exact nine passing integrity checks required')


def historical_seeds(root: Path = PROJECT_ROOT / 'experiments', *, exclude: Path | None = None) -> set[int]:
    """Read identities from protocols, results, partial journals and episode CSVs."""
    seeds: set[int] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith('seed') and type(item) is int:
                    seeds.add(item)
                if key == 'seeds' and isinstance(item, list):
                    seeds.update(x for x in item if type(x) is int)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for path in Path(root).rglob('*'):
        if not path.is_file() or (exclude is not None and path.resolve().is_relative_to(exclude.resolve())):
            continue
        if path.suffix == '.json':
            visit(json.loads(path.read_text(encoding='utf-8-sig')))
        elif path.suffix == '.jsonl':
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():
                    visit(json.loads(line))
        elif path.suffix == '.csv':
            with path.open(newline='', encoding='utf-8-sig') as stream:
                for row in csv.DictReader(stream):
                    for key, value in row.items():
                        if key and key.endswith('seed') and value:
                            seeds.add(int(value))
    return seeds


def training_evidence() -> tuple[dict, str, dict, dict]:
    """Recompute checkpoint errors and recorded behavior with zero neural steps."""
    if any(sha256(CANDIDATE / name) != digest for name, digest in PINNED_CANDIDATE.items()):
        raise ValueError('frozen candidate bytes differ')
    p, digest, data = load_training_protocol(CANDIDATE / 'protocol.json')
    r = json.loads((CANDIDATE / 'results.json').read_text(encoding='utf-8'))
    validate_checks(r['checks'])
    if (r['training_gate_passed'] is not False or r['failure'] is not None
            or r['optimizer_steps'] != 500 or r['protocol_sha256'] != digest
            or r['held_out_evaluated'] is not False or len(data.features) != 15864):
        raise ValueError('preserved closed fit evidence differs')
    mean = torch.tensor(p['normalization']['mean'])
    scale = torch.tensor(p['normalization']['scale'])
    models = {name: load_checkpoint(CANDIDATE / (name + '.pt'), protocol_sha256=digest,
                                   mean=mean, scale=scale) for name in ('trained', 'untrained')}
    initial = ProjectionReadout(mean, scale, seed=1010)
    if any(not torch.equal(v, initial.state_dict()[k]) for k, v in models['untrained'].state_dict().items()):
        raise ValueError('untrained control is not exact seed-1010 initialization')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            errors = (models['trained'](data.features).double() - data.targets).square()
        covered = errors.mean(0).tolist()
        original = errors[:1008].mean(0).tolist()
    finally:
        torch.set_num_threads(previous)
    expected = [(c.case_id, c.split, c.seed) for c in locked_cases()['train']]
    rows = {name: [EpisodeResult(**row) for row in r['episodes'][name]] for name in CONTROLLERS}
    if any([(row.case_id, row.split, row.seed) for row in group] != expected for group in rows.values()):
        raise ValueError('original matched case identities differ')
    summaries = {name: summarize(group) for name, group in rows.items()}
    covered_gates = coverage_gates(covered[1], original[1], p['train_only_baseline']['constant_mse'][1], summaries, r['checks'])
    original_gates = training_gates(original[1], 4 * ORIGINAL_TURN_CEILING, summaries, r['checks'])
    changed = sum(int(torch.count_nonzero(v - models['untrained'].state_dict()[k]))
                  for k, v in models['trained'].named_parameters())
    replay = EpisodeResult(**r['replay_episode'])
    ticks = sum(row.ticks for group in rows.values() for row in group) + replay.ticks
    if (covered != r['per_output_mse'] or original != r['original_per_output_mse']
            or summaries != r['summaries'] or covered_gates != r['gates']
            or not original_gates['passed'] or changed != r['parameter_changed_scalars'] or changed != 370
            or replay != rows['trained'][0] or ticks != r['actual_world_ticks']
            or r['actual_neural_steps'] != ticks * 20 or r['actual_episodes'] != 25):
        raise ValueError('recomputed frozen training evidence differs')
    for model in models.values():
        model.requires_grad_(False)
    evidence = {'original_gates': original_gates, 'covered_gates': covered_gates,
                'original_per_output_mse': original, 'covered_per_output_mse': covered,
                'original_constant_turn_mse': 4 * ORIGINAL_TURN_CEILING,
                'original_fit_training_gate_passed': False,
                'matched_initialization_exact': True, 'changed_parameters': changed,
                'recorded_integrity_checks': r['checks'], 'summaries': summaries,
                'failed_gate': {'name': 'aggregate_absolute_turn_fit',
                                'observed': covered[1], 'ceiling': ORIGINAL_TURN_CEILING},
                'audit_world_ticks': 0, 'audit_neural_steps': 0, 'optimizer_updates': 0}
    return p, digest, models, evidence


def validate_acceptance(acceptance: dict | None) -> None:
    """Require an explicit recorded user decision, not an inferred completion wish."""
    if (not isinstance(acceptance, dict) or acceptance.get('rule') != ACCEPTANCE_RULE
            or acceptance.get('covered_failure_retained') is not True
            or not isinstance(acceptance.get('user_decision'), str) or not acceptance['user_decision'].strip()):
        raise ValueError('explicit original-gate acceptance with failed covered gate retained required')


def protocol_definition(acceptance: dict) -> dict:
    """Declare fixed cases, hypothesis, budgets and unchanged stopping rules."""
    validate_acceptance(acceptance)
    p, digest, _, evidence = training_evidence()
    return {
        'schema': 'fb10_frozen_original_gate_assay_v1', 'experiment': 'fb10_frozen_original_gate_assay',
        'acceptance': acceptance, 'eligibility': evidence, 'candidate_sha256': PINNED_CANDIDATE,
        'training_protocol_sha256': digest,
        'hypothesis': 'The frozen internal-spike readout improves fresh food arrival over its identical untrained initialization and fixed BANC decoder.',
        'cases': {split: [asdict(c) for c in group] for split, group in fresh_cases().items()},
        'config': p['world_config'], 'software': software_record(), 'network_size': 83,
        'trainable_parameters': 370, 'optimizer_steps': 0, 'learning_rate': None,
        'objective': 'Paired fresh arrivals and all-case progress against both controls',
        'budget': {'maximum_episodes': 144, 'maximum_world_ticks': 86400, 'maximum_neural_steps': 1728000, 'optimizer_steps': 0},
        'gates': {'validation_arrivals_min': 13, 'final_arrivals_min': 26,
                  'final_paired_arrival_difference_min': 8, 'final_one_sided_p_max_per_control': .025,
                  'strict_matched_arrival_and_all_case_progress': True, 'frozen_parameters_core_provenance': True},
        'stop_condition': 'One central receipt; stop before final if validation fails. No fitting, tuning, retry, case selection or topology claim.',
        'scope_limit': 'PN incoming-spike and endpoint histories; downstream mediation, topology benefit and biological fidelity are not established.',
    }


def prepare_protocol(output_dir: Path, *, acceptance: dict | None) -> Path:
    """Freeze the accepted assay before observing any fresh outcome."""
    validate_acceptance(acceptance)
    if CENTRAL_RECEIPT.exists():
        raise FileExistsError('frozen candidate fresh assay already consumed')
    out = Path(output_dir).resolve()
    ensure_new_run_directory(out)
    definition = protocol_definition(acceptance)
    seeds = {row['seed'] for group in definition['cases'].values() for row in group}
    if seeds.intersection(historical_seeds(exclude=out)):
        raise ValueError('fresh seeds overlap prior records')
    p = {**definition, 'output_dir': str(out), 'prepared_utc': datetime.now(timezone.utc).isoformat(),
         'git_commit': _git_commit(), 'provenance': capture_provenance(PROJECT_ROOT, out)}
    out.mkdir(parents=True)
    path = out / 'protocol.json'
    save_new_json(path, p)
    with path.with_suffix('.sha256').open('x', encoding='utf-8') as stream:
        stream.write(sha256(path) + '\n')
    return path


def load_protocol(path: Path) -> tuple[dict, str]:
    """Reject changed evidence, copied destinations, historical cases and sources."""
    path = Path(path).resolve()
    digest = sha256(path)
    if path.with_suffix('.sha256').read_text(encoding='utf-8').strip() != digest:
        raise ValueError('frozen assay protocol hash differs')
    p = json.loads(path.read_text(encoding='utf-8'))
    if path != Path(p['output_dir']).resolve() / 'protocol.json':
        raise ValueError('frozen assay destination differs')
    inventory = capture_provenance(PROJECT_ROOT, path.parent)
    # The central receipt is the only permitted new file outside the assay.
    # All other sources and historical files must be fully represented.
    inventory['previous_files_sha256'].pop(str(CENTRAL_RECEIPT.resolve()), None)
    recorded = p.get('provenance', {})
    if (set(recorded) != set(inventory)
            or any(not recorded[group] or set(recorded[group]) != set(files)
                   for group, files in inventory.items())):
        raise ValueError('frozen source/history inventory differs')
    if not verify_provenance(recorded):
        raise ValueError('frozen source/history differs')
    if any(p.get(k) != v for k, v in protocol_definition(p['acceptance']).items()):
        raise ValueError('frozen assay definition differs')
    seeds = {row['seed'] for group in p['cases'].values() for row in group}
    if seeds.intersection(historical_seeds(exclude=path.parent)):
        raise ValueError('fresh cases overlap prior records')
    return p, digest


def run_evaluation(path: Path) -> dict:
    """Execute once, preserving failed validation and partial execution honestly."""
    path = Path(path).resolve()
    out = path.parent
    if CENTRAL_RECEIPT.exists() or any((out / n).exists() for n in ('execution.json', 'results.json', 'episodes.jsonl', 'aborted.json')):
        raise FileExistsError('frozen candidate assay already consumed')
    p, digest = load_protocol(path)
    _, _, models, _ = training_evidence()
    initial = {n: {k: v.clone() for k, v in m.state_dict().items()} for n, m in models.items()}
    receipt = {'status': 'consumed', 'protocol_sha256': digest, 'output_dir': str(out),
               'started_utc': datetime.now(timezone.utc).isoformat()}
    save_new_json(CENTRAL_RECEIPT, receipt)
    previous = torch.get_num_threads()
    completed = ticks = steps = 0
    results, summaries, gates = {}, {}, {}
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
                rows = {n: [] for n in CONTROLLERS}
                for name in CONTROLLERS:
                    for case in cases:
                        active = {'condition': name, 'case': asdict(case)}
                        controller = None
                        controller = (make_banc_controller(load_verified_banc_config()) if name == 'fixed'
                                      else make_projection_controller(models[name]))
                        weight = weights_sha256(controller)
                        with torch.no_grad():
                            row = run_episode(case, EvaluationConfig(**p['config']), lambda: controller)
                        neural_steps = controller.network.population.step_count
                        completed += 1
                        ticks += row.ticks
                        steps += neural_steps
                        active = None  # Work is accounted even if journal I/O fails.
                        journal.write(json.dumps({'condition': name, 'result': asdict(row), 'neural_steps': neural_steps}, allow_nan=False) + '\n')
                        journal.flush()
                        if (neural_steps != row.ticks * 20 or weights_sha256(controller) != weight
                                or any(not torch.equal(v, initial[n][k]) for n, m in models.items() for k, v in m.state_dict().items())):
                            raise RuntimeError('timing or frozen readout/core differs')
                        if completed > 144 or ticks > 86400 or steps > 1728000:
                            raise RuntimeError('frozen assay budget exceeded')
                        if (row.case_id, row.split, row.seed) != (case.case_id, case.split, case.seed):
                            raise RuntimeError('matched episode identity differs')
                        rows[name].append(row)
                        if len(rows[name]) % 4 == 0:
                            print(f'{split}/{name}: {len(rows[name])}/{len(cases)}', flush=True)
                summaries[split] = {n: summarize(group) for n, group in rows.items()}
                gates[split] = assess_split(split, {n: [r.outcome == 'success' for r in group] for n, group in rows.items()}, summaries[split])
                results[split] = {n: [asdict(r) for r in group] for n, group in rows.items()}
                _write_csv(out / (split + '.csv'), [dict(condition=n, **r) for n, group in results[split].items() for r in group])
                save_new_json(out / (split + '.json'), {'gates': gates[split], 'summaries': summaries[split], 'results': results[split]})
                print(f"{split}: arrivals={gates[split]['successes']}; passed={gates[split]['passed']}", flush=True)
        load_protocol(path)
        r = {'experiment': p['experiment'], 'protocol_sha256': digest, 'acceptance': p['acceptance'],
             'eligibility': p['eligibility'], 'software': software_record(), 'git_commit': p['git_commit'],
             'network_size': 83, 'objective': p['objective'], 'optimizer_steps': 0, 'learning_rate': None,
             'checkpoint': str(CANDIDATE / 'trained.pt'), 'checkpoint_sha256': sha256(CANDIDATE / 'trained.pt'),
             'actual_episodes': completed, 'actual_world_ticks': ticks, 'actual_neural_steps': steps,
             'results': results, 'summaries': summaries, 'gates': gates, 'final_cases_evaluated': 'final' in results,
             'frozen_parameters_unchanged': True, 'effective_weights_unchanged': True, 'integrity_checks_passed': True,
             'fb10_gate_passed': gates.get('final', {}).get('passed', False), 'scope_limit': p['scope_limit'],
             'completed_utc': datetime.now(timezone.utc).isoformat()}
        save_new_json(out / 'results.json', r)
        return r
    except BaseException as exc:
        partial_steps = controller.network.population.step_count if active is not None and controller is not None else 0
        save_new_json(out / 'aborted.json', {'type': type(exc).__name__, 'error': str(exc),
            'protocol_sha256': digest, 'completed_episodes': completed, 'completed_world_ticks': ticks,
            'completed_neural_steps': steps, 'in_progress': active, 'in_progress_neural_steps': partial_steps,
            'actual_neural_steps': steps + partial_steps, 'in_progress_world_ticks': None if active is not None else 0,
            'instruction': 'Consumed even if interrupted. Unknown partial world ticks are not zero. No retry.'})
        raise
    finally:
        torch.set_num_threads(previous)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--acceptance-file', type=Path)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_OUTPUT / 'protocol.json')
    args = parser.parse_args()
    if args.prepare:
        acceptance = json.loads(args.acceptance_file.read_text(encoding='utf-8')) if args.acceptance_file else None
        path = prepare_protocol(args.output_dir, acceptance=acceptance)
        print(json.dumps({'protocol': str(path), 'sha256': sha256(path), 'optimizer_steps': 0, 'world_ticks': 0}))
        return 0
    r = run_evaluation(args.protocol)
    print(json.dumps({k: r[k] for k in ('fb10_gate_passed', 'actual_episodes', 'actual_world_ticks', 'actual_neural_steps')}, indent=2))
    return 0 if r['fb10_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
