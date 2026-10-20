"""Proposed one-use, final-only assay of the unchanged food-stop checkpoint.

Preparing and running this assay are separate explicit calls. Importing this
module, auditing a journal, and running its tests consume no evaluation cases.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform

import torch

from flybrain.neural.banc_motor import make_banc_controller
from flybrain.neural.banc_network import load_verified_banc_config
from flybrain.neural.food_readout import make_food_controller

from .evaluation import EpisodeCase
from .food_evaluation import run_controller_case
from .food_fresh import FIT, _fit_evidence, _models, final_gates
from .food_preflight import PROJECT_ROOT


CASE_MANIFEST = PROJECT_ROOT / 'docs/plans/FOOD-STOP-RECOVERY-CASES.json'
CASE_MANIFEST_SHA256 = '1941e4436fa3b857faf26746560c43874df1cea352a8fe55e0ffa91072ac1d32'
FIT_PROTOCOL_SHA256 = '3771cd4bd2b3304f40421fb19f7666d01fa3f8a91894a96a3c410352ad1e7452'
FIT_RESULTS_SHA256 = '0d55d9661d0a02b8690a39a0915cccfa90f879ce8b8ddf7463f3ae2e5b2c3237'
TRAINED_CHECKPOINT_SHA256 = '53526140b8b7f5e48b7f4a97f9db380171fad220275d339912638aadfaa8ee1b'
OLD_FRESH_PROTOCOL_SHA256 = '825c590d6e8c65b170c7e0363bb287503a4693e1c4f7034eb30d5190a52a0731'
OLD_VALIDATION_SHA256 = 'e7f4bc8eeb25a8899e4bf819f7ac58cdf3f31626aed18a2d9a6e0c2e18024ac5'
OLD_INTERRUPTION_SHA256 = 'aa5f2a19dbea6df9a6d25b425a99a85b542c57c099b39143ec863fe87ddfc6c5'
CONTACT_PROTOCOL_SHA256 = '66bfb9bd3c2e082f373c38b4914a17219b0d4eb68be26a0ae9bd5ac9f7748750'
CONTACT_RESULTS_SHA256 = 'fb1f19131e33e29d7ccc64248e75d7f02569ed75eb7584340d73325949a8f762'
OLD_FRESH = PROJECT_ROOT / 'experiments/food-stop-fresh-2026-09-28'
DIAGNOSTIC = PROJECT_ROOT / 'experiments/food-stop-contact-diagnostic-2026-09-29'
DEFAULT_OUTPUT = PROJECT_ROOT / 'experiments/food-stop-recovery-2026-09-29'
CONTROLLERS = ('trained', 'untrained', 'fixed')


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def recovery_cases() -> tuple[EpisodeCase, ...]:
    """Load the proposed exact case bytes and validate geometry and sectors."""
    if _digest(CASE_MANIFEST) != CASE_MANIFEST_SHA256:
        raise ValueError('recovery case manifest differs from reviewed proposal')
    manifest = json.loads(CASE_MANIFEST.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'food_stop_recovery_cases_v1':
        raise ValueError('unexpected recovery case schema')
    rows = manifest['cases']
    if len(rows) != 64 or [row['seed'] for row in rows] != list(range(840000, 840064)):
        raise ValueError('64 unique reserved recovery seeds required')
    cases = []
    sectors = [0] * 8
    for row in rows:
        case = EpisodeCase(**{key: row[key] for key in
                              ('case_id', 'split', 'seed', 'x', 'y', 'heading')})
        radius = math.hypot(case.x - 600., case.y - 300.)
        sector = int(case.heading / (math.tau / 8))
        if (case.split != 'final'
                or case.case_id != f'food-recovery-final-{case.seed}'
                or not all(math.isfinite(value) for value in
                           (case.x, case.y, case.heading))
                or not (60 <= radius <= 220 and 6 <= case.x <= 794
                        and 6 <= case.y <= 594 and 0 <= case.heading < math.tau)
                or row['heading_sector'] != sector):
            raise ValueError('invalid recovery pose or heading sector')
        sectors[sector] += 1
        cases.append(case)
    if sectors != [8] * 8:
        raise ValueError('recovery headings are not balanced')
    return tuple(cases)


def _prior_evidence() -> dict:
    """Require the recorded fit, validation, interruption and intervention."""
    fit_protocol, fit_digest, fit = _fit_evidence()
    old_protocol_path = OLD_FRESH / 'protocol.json'
    if (fit_digest != FIT_PROTOCOL_SHA256
            or _digest(FIT / 'results.json') != FIT_RESULTS_SHA256
            or fit['checkpoint_sha256'] != TRAINED_CHECKPOINT_SHA256
            or _digest(old_protocol_path) != OLD_FRESH_PROTOCOL_SHA256
            or _digest(OLD_FRESH / 'validation.json') != OLD_VALIDATION_SHA256
            or _digest(OLD_FRESH / 'interruption.json') != OLD_INTERRUPTION_SHA256
            or _digest(DIAGNOSTIC / 'protocol.json') != CONTACT_PROTOCOL_SHA256
            or _digest(DIAGNOSTIC / 'results.json') != CONTACT_RESULTS_SHA256):
        raise ValueError('pinned prior evidence differs')
    old_protocol = json.loads(old_protocol_path.read_text(encoding='utf-8'))
    interruption = json.loads((OLD_FRESH / 'interruption.json').read_text(encoding='utf-8'))
    validation = json.loads((OLD_FRESH / 'validation.json').read_text(encoding='utf-8'))
    if (old_protocol.get('fit_protocol_sha256') != fit_digest
            or interruption.get('protocol_sha256') != _digest(old_protocol_path)
            or interruption.get('status') != 'interrupted_after_validation'
            or interruption.get('validation_gate_passed') is not True
            or interruption.get('final_result_recorded') is not False
            or (OLD_FRESH / 'final.json').exists()
            or (OLD_FRESH / 'results.json').exists()
            or set(validation) != set(CONTROLLERS)
            or any(_digest(PROJECT_ROOT / name) != digest
                   for name, digest in old_protocol['source_sha256'].items())):
        raise ValueError('prior one-use attempt is not the recorded interruption')
    expected = old_protocol['cases']['validation']
    for name in CONTROLLERS:
        rows = validation[name]
        if (len(rows) != 16
                or [{key: row[key] for key in ('case_id', 'split', 'seed', 'x', 'y', 'heading')}
                    for row in rows] != expected
                or any(row['world_ticks'] != 1000 or row['neural_steps'] != 20000
                       for row in rows)
                or sum(row['success'] is True for row in rows)
                != interruption['validation_successes'][name]):
            raise ValueError('recorded validation does not reconcile')
    if interruption['validation_successes'] != {'trained': 16, 'untrained': 0, 'fixed': 0}:
        raise ValueError('recorded validation gate differs')
    diagnostic = json.loads((DIAGNOSTIC / 'results.json').read_text(encoding='utf-8'))
    diagnostic_protocol = DIAGNOSTIC / 'protocol.json'
    if (diagnostic.get('protocol_sha256') != _digest(diagnostic_protocol)
            or diagnostic.get('changes_stop') is not True
            or diagnostic.get('parameters_unchanged') is not True
            or diagnostic.get('optimizer_updates') != 0
            or diagnostic_protocol.read_text(encoding='utf-8').find(
                fit['checkpoint_sha256']) < 0):
        raise ValueError('recorded contact-path intervention differs')
    return {'fit_protocol_sha256': fit_digest,
            'fit_results_sha256': _digest(FIT / 'results.json'),
            'trained_checkpoint_sha256': fit['checkpoint_sha256'],
            'untrained_checkpoint_sha256': fit['untrained_sha256'],
            'old_fresh_protocol_sha256': _digest(old_protocol_path),
            'old_validation_sha256': _digest(OLD_FRESH / 'validation.json'),
            'old_interruption_sha256': _digest(OLD_FRESH / 'interruption.json'),
            'contact_protocol_sha256': _digest(diagnostic_protocol),
            'contact_results_sha256': _digest(DIAGNOSTIC / 'results.json'),
            'fit_source_sha256': fit_protocol['source_sha256']}


def prepare_recovery(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    """Declare the exact frozen final-only attempt without running any case."""
    if output_dir.exists():
        raise FileExistsError('recovery output directory already exists')
    cases = recovery_cases()
    evidence = _prior_evidence()
    source_root = PROJECT_ROOT / 'src/flybrain'
    sources = set((source_root / 'neural').rglob('*.py'))
    sources.update((source_root / 'data').rglob('*.py'))
    sources.update((source_root / 'world').rglob('*.py'))
    sources.update(source_root / name for name in
                   ('behavior.py', 'closed_loop.py', 'foraging.py', 'simulation.py'))
    sources.update(source_root / 'training' / name for name in
                   ('evaluation.py', 'food_evaluation.py', 'food_fresh.py',
                    'food_preflight.py', 'food_recovery.py'))
    sources.discard(source_root / 'neural/accepted_food.py')
    record = {
        'schema': 'food_stop_recovery_v1',
        'hypothesis': 'The unchanged frozen food-stop checkpoint meets the original final full-episode and paired-control gates on 64 new starts.',
        'cases_manifest_sha256': CASE_MANIFEST_SHA256,
        'cases': [asdict(case) for case in cases],
        'prior_evidence_sha256': evidence,
        'source_sha256': {str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'):
                          _digest(path) for path in sorted(sources)},
        'network_size': 85, 'trainable_parameters': 419,
        'objective': '20-second contact, braking, exact stop and uninterrupted dwell against matched frozen controls',
        'optimizer_steps': 0, 'learning_rate': None,
        'gates': {'final_successes_min': 58, 'sector_successes_min': 7,
                  'paired_gain_min': 8, 'one_sided_exact_p_max': .025,
                  'contact_block_changes_stop': True},
        'budget': {'episodes': 192, 'world_ticks': 192000,
                   'neural_steps': 3840000, 'optimizer_updates': 0},
        'stop_condition': 'One use. On any interruption or failed gate, preserve the journal and stop; no replay, tuning or additional split.',
        'software_device': {'python': platform.python_version(),
                            'torch': torch.__version__, 'device': 'CPU'},
        'commit': None, 'prepared_utc': _now(),
    }
    output_dir.mkdir(parents=True)
    path = output_dir / 'protocol.json'
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return path


def _append_episode(stream, controller: str, result: dict) -> None:
    """Persist one completed episode before advancing to the next case."""
    if controller not in CONTROLLERS:
        raise ValueError('unknown controller')
    stream.write(json.dumps({'controller': controller, 'result': result},
                            allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def audit_journal(path: Path, cases: tuple[EpisodeCase, ...]) -> list[dict]:
    """Require precisely 192 ordered, complete, matched case results."""
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 192:
        raise ValueError('192 complete final episodes required')
    for index, (case, name) in enumerate((case, name)
                                         for case in cases for name in CONTROLLERS):
        wrapper = rows[index]
        result = wrapper['result']
        if (wrapper['controller'] != name
                or any(result[key] != value for key, value in asdict(case).items())
                or result['heading_sector'] != int(case.heading / (math.tau / 8))):
            raise ValueError('journal case identity or ordering differs')
        if (type(result['success']) is not bool
                or result['world_ticks'] != 1000
                or result['neural_steps'] != 20000):
            raise ValueError('incomplete final episode in journal')
    return rows


def _write_final_tables(output_dir: Path, rows: list[dict]) -> None:
    episodes = {name: [row['result'] for row in rows if row['controller'] == name]
                for name in CONTROLLERS}
    (output_dir / 'final.json').write_text(
        json.dumps(episodes, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output_dir / 'final.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['controller', *rows[0]['result']])
        writer.writeheader()
        for row in rows:
            writer.writerow({'controller': row['controller'], **row['result']})


def run_recovery(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    """Consume a prepared protocol once and journal all 192 final episodes."""
    protocol_path = output_dir / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    cases = recovery_cases()
    if (protocol.get('schema') != 'food_stop_recovery_v1'
            or protocol.get('cases_manifest_sha256') != CASE_MANIFEST_SHA256
            or protocol.get('cases') != [asdict(case) for case in cases]
            or protocol.get('prior_evidence_sha256') != _prior_evidence()
            or any(_digest(PROJECT_ROOT / name) != digest
                   for name, digest in protocol['source_sha256'].items())):
        raise ValueError('prepared recovery protocol or frozen evidence differs')
    receipt = output_dir / 'execution.json'
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump({'status': 'consumed', 'protocol_sha256': _digest(protocol_path),
                   'started_utc': _now()}, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        models = _models({'normalization': json.loads(
            (FIT / 'results.json').read_text(encoding='utf-8'))['normalization'],
            'fit_protocol_sha256': protocol['prior_evidence_sha256']['fit_protocol_sha256']})
        before = {name: {key: value.clone() for key, value in model.state_dict().items()}
                  for name, model in models.items()}
        config = load_verified_banc_config()
        with (output_dir / 'episodes.jsonl').open('x', encoding='utf-8') as journal:
            with torch.no_grad():
                for case in cases:
                    for name in CONTROLLERS:
                        controller = (make_food_controller(models[name]) if name != 'fixed'
                                      else make_banc_controller(config))
                        result = run_controller_case(case, controller)
                        _append_episode(journal, name, result)
                        print(f"{len(cases)}-case final: {case.seed} {name} complete",
                              flush=True)
        rows = audit_journal(output_dir / 'episodes.jsonl', cases)
        unchanged = all(torch.equal(value, models[name].state_dict()[key])
                        for name, state in before.items() for key, value in state.items())
        _write_final_tables(output_dir, rows)
        final = json.loads((output_dir / 'final.json').read_text(encoding='utf-8'))
        gates = final_gates(final['trained'], final['untrained'], final['fixed'])
        result = {
            'schema': 'food_stop_recovery_result_v1',
            'protocol_sha256': _digest(protocol_path),
            'case_manifest_sha256': CASE_MANIFEST_SHA256,
            'journal_sha256': _digest(output_dir / 'episodes.jsonl'),
            'final_json_sha256': _digest(output_dir / 'final.json'),
            'final_csv_sha256': _digest(output_dir / 'final.csv'),
            'checkpoint_sha256': protocol['prior_evidence_sha256']['trained_checkpoint_sha256'],
            'validation_successes': {'trained': 16, 'untrained': 0, 'fixed': 0},
            'validation_gate_passed': True,
            'final_gates': gates,
            'contact_intervention_sha256': protocol['prior_evidence_sha256']['contact_results_sha256'],
            'contact_intervention_changes_stop': True,
            'parameters_unchanged': unchanged,
            'actual_episodes': len(rows),
            'actual_world_ticks': sum(row['result']['world_ticks'] for row in rows),
            'actual_neural_steps': sum(row['result']['neural_steps'] for row in rows),
            'optimizer_steps': 0,
            'passed': bool(gates['passed'] and unchanged and len(rows) == 192),
            'completed_utc': _now(),
        }
        (output_dir / 'results.json').write_text(
            json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return result
    except Exception as error:
        journal = output_dir / 'episodes.jsonl'
        completed = len(journal.read_text(encoding='utf-8').splitlines()) if journal.exists() else 0
        (output_dir / 'interruption.json').write_text(
            json.dumps({'status': 'interrupted', 'protocol_sha256': _digest(protocol_path),
                        'completed_episode_rows': completed, 'error': repr(error),
                        'recorded_utc': _now()}, indent=2) + '\n', encoding='utf-8')
        raise
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    """Require an explicit prepare or run action for the single-use assay."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--run', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.prepare:
        print(prepare_recovery(args.output_dir))
    else:
        print(json.dumps(run_recovery(args.output_dir), indent=2))


if __name__ == '__main__':
    main()
