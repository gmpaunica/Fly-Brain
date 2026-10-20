"""Offline reconciliation and saved-result plot for the capped FB-10 extension."""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flybrain.training.banc_extra_evaluation import (DEFAULT_OUTPUT as EVAL,
    CENTRAL_RECEIPT, assess_split, load_protocol, training_evidence)
from flybrain.training.banc_extra_training import DEFAULT_OUTPUT as FIT
from flybrain.training.banc_protocol import sha256
from flybrain.training.banc_readiness import PROJECT_ROOT, verify_provenance
from flybrain.training.evaluation import EpisodeResult, summarize


def same_rows(csv_path: Path, expected: list[dict]) -> bool:
    """Require exact CSV identity, order and values, including empty nulls."""
    with csv_path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    return rows == [{key: '' if value is None else str(value)
                     for key, value in row.items()} for row in expected]


def preserved(manifest_name: str) -> tuple[int, list[str]]:
    baseline = json.loads((PROJECT_ROOT / 'docs/verification' / manifest_name).read_text())
    changed = [name for name, digest in baseline.items()
               if not (PROJECT_ROOT / name).is_file() or hashlib.sha256(
                   (PROJECT_ROOT / name).read_bytes()).hexdigest() != digest]
    return len(baseline), changed


def audit() -> dict:
    """Check every saved run row and gate without stepping or optimizing."""
    checks: dict[str, bool] = {}
    fit_protocol_path, eval_protocol_path = FIT / 'protocol.json', EVAL / 'protocol.json'
    fit_protocol = json.loads(fit_protocol_path.read_text())
    fit_result = json.loads((FIT / 'results.json').read_text())
    eval_protocol, eval_digest = load_protocol(eval_protocol_path)
    eval_result = json.loads((EVAL / 'results.json').read_text())
    checks['fit_protocol_hash'] = (sha256(fit_protocol_path)
        == fit_protocol_path.with_suffix('.sha256').read_text().strip()
        == fit_result['protocol_sha256'])
    checks['fit_provenance'] = verify_provenance(fit_protocol['provenance'])
    checks['fit_checkpoint_hashes'] = (sha256(FIT / 'trained.pt')
        == fit_result['checkpoint_sha256']
        == eval_protocol['candidate_sha256']['trained.pt'])
    checks['fit_500_prefix'] = (fit_result['historical_500_update_prefix_exact'] is True
        and fit_result['curve'][500] == json.loads((PROJECT_ROOT /
        'experiments/fb10-projection-readout/results.json').read_text())['curve'][500])
    checks['fit_budget_and_gates'] = (fit_result['failure'] is None
        and fit_result['training_gate_passed'] is True
        and fit_result['optimizer_steps'] == 750
        and fit_result['actual_episodes'] == 25
        and fit_result['actual_world_ticks'] <= 15000
        and fit_result['actual_neural_steps'] == fit_result['actual_world_ticks'] * 20
        and all(value is True for value in fit_result['checks'].values())
        and all(value is True for value in fit_result['gates'].values()))
    checks['fit_training_evidence_recomputed'] = training_evidence()[3] == fit_result
    train_rows = [dict(condition=name, **row)
                  for name, group in fit_result['episodes'].items() for row in group]
    train_rows.append(dict(condition='trained_replay', **fit_result['replay_episode']))
    checks['fit_episode_csv'] = same_rows(FIT / 'episodes.csv', train_rows)
    checks['fit_episode_summaries'] = all(
        summarize([EpisodeResult(**row) for row in group]) == fit_result['summaries'][name]
        for name, group in fit_result['episodes'].items())
    checks['fit_replay_exact'] = fit_result['replay_episode'] == fit_result['episodes']['trained'][0]
    checks['fit_curve_csv'] = same_rows(FIT / 'curve.csv', fit_result['curve'])
    checks['eval_protocol_hash'] = (sha256(eval_protocol_path) == eval_digest
        == eval_result['protocol_sha256'])
    checks['eval_provenance'] = verify_provenance(eval_protocol['provenance'])
    checks['eval_receipts'] = (json.loads(CENTRAL_RECEIPT.read_text())['protocol_sha256']
        == json.loads((EVAL / 'execution.json').read_text())['protocol_sha256']
        == eval_digest and not (EVAL / 'aborted.json').exists())
    checks['eval_candidate_hashes'] = all(
        sha256(FIT / name) == digest
        for name, digest in eval_protocol['candidate_sha256'].items())
    journal = [json.loads(line) for line in (EVAL / 'episodes.jsonl').read_text().splitlines()
               if line.strip()]
    expected_journal = []
    for split in ('validation', 'final'):
        for name in ('trained', 'untrained', 'fixed'):
            expected_journal.extend({'condition': name, 'result': row,
                'neural_steps': row['ticks'] * 20} for row in eval_result['results'][split][name])
    checks['eval_journal_order_and_rows'] = journal == expected_journal
    checks['eval_counters'] = (len(journal) == eval_result['actual_episodes'] == 144
        and sum(row['result']['ticks'] for row in journal) == eval_result['actual_world_ticks']
        and sum(row['neural_steps'] for row in journal) == eval_result['actual_neural_steps']
        == 20 * eval_result['actual_world_ticks']
        and eval_result['actual_world_ticks'] <= 86400
        and eval_result['actual_neural_steps'] <= 1728000
        and eval_result['optimizer_steps'] == 0)
    for split in ('validation', 'final'):
        block = json.loads((EVAL / f'{split}.json').read_text())
        rows = eval_result['results'][split]
        summaries = {name: summarize([EpisodeResult(**row) for row in group])
                     for name, group in rows.items()}
        gates = assess_split(split, {name: [row['outcome'] == 'success' for row in group]
                                     for name, group in rows.items()}, summaries)
        checks[f'{split}_summaries_gates'] = (summaries == block['summaries']
            == eval_result['summaries'][split]
            and gates == block['gates'] == eval_result['gates'][split]
            and gates['passed'] is True)
        checks[f'{split}_csv'] = same_rows(EVAL / f'{split}.csv', [
            dict(condition=name, **row) for name, group in rows.items() for row in group])
        checks[f'{split}_case_identities'] = all(
            [(row['case_id'], row['split'], row['seed']) for row in rows[name]]
            == [(case['case_id'], case['split'], case['seed'])
                for case in eval_protocol['cases'][split]]
            for name in ('trained', 'untrained', 'fixed'))
    checks['eval_integrity_and_gate'] = (eval_result['fresh_gate_passed'] is True
        and eval_result['final_cases_evaluated'] is True
        and eval_result['frozen_parameters_unchanged'] is True
        and eval_result['effective_weights_unchanged'] is True
        and eval_result['integrity_checks_passed'] is True)
    count_fit, changed_fit = preserved('fb10-extra-before.json')
    count_eval, changed_eval = preserved('fb10-extra-before-fresh.json')
    checks['all_pretraining_files_preserved'] = count_fit == 303 and not changed_fit
    checks['all_preevaluation_files_preserved'] = count_eval == 312 and not changed_eval
    result = {'schema': 'fb10_extra_offline_audit_v1', 'checks': checks,
        'passed': all(checks.values()), 'checks_passed': sum(checks.values()),
        'pretraining_files': count_fit, 'preevaluation_files': count_eval,
        'changed_pretraining_files': changed_fit,
        'changed_preevaluation_files': changed_eval,
        'audit_world_ticks': 0, 'audit_neural_steps': 0, 'audit_optimizer_steps': 0,
        'training_protocol_sha256': sha256(fit_protocol_path),
        'evaluation_protocol_sha256': eval_digest,
        'checkpoint_sha256': sha256(FIT / 'trained.pt')}
    if not result['passed']:
        raise ValueError({key: value for key, value in checks.items() if not value})
    return result


def plot_saved() -> None:
    """Draw training error and paired arrivals from saved records only."""
    fit = json.loads((FIT / 'results.json').read_text())
    fresh = json.loads((EVAL / 'results.json').read_text())
    fig, (loss_ax, arrivals_ax) = plt.subplots(1, 2, figsize=(11, 4.2),
                                                layout='constrained')
    curve = fit['curve']
    loss_ax.plot([row['update'] for row in curve],
                 [row['turn_mse'] for row in curve], label='Covered turn MSE', color='#2454a6')
    loss_ax.axhline(fit['historical_covered_failure']['ceiling'],
                    color='#ad4d30', linestyle='--', label='Extra ceiling')
    loss_ax.axvline(500, color='#6d7380', linestyle=':', label='Old stop at 500')
    loss_ax.set(xlabel='Adam updates', ylabel='Training turn MSE',
                title='One capped 750-update fit')
    loss_ax.set_yscale('log')
    loss_ax.set_ylim(.015, .5)
    loss_ax.set_yticks((.02, .03, .05, .1, .2, .4),
                       ('0.02', '0.03', '0.05', '0.10', '0.20', '0.40'))
    loss_ax.legend(fontsize=8)
    groups = ('trained', 'untrained', 'fixed')
    colors = ('#2454a6', '#b86d45', '#718779')
    for index, split in enumerate(('validation', 'final')):
        for offset, (name, color) in enumerate(zip(groups, colors)):
            successes = fresh['gates'][split]['successes'][name]
            arrivals_ax.bar(index + (offset - 1) * .22, successes, .2,
                            color=color, label=name if index == 0 else None)
            arrivals_ax.text(index + (offset - 1) * .22, successes + .4,
                             str(successes), ha='center', fontsize=8)
    arrivals_ax.set_xticks((0, 1), ('Validation / 16', 'Final / 32'))
    arrivals_ax.set(ylabel='Food arrivals', title='Fresh matched navigation', ylim=(0, 35))
    arrivals_ax.legend(fontsize=8)
    for suffix in ('png', 'svg'):
        fig.savefig(EVAL / f'extra_training_summary.{suffix}', dpi=160)
    plt.close(fig)


def main() -> int:
    result = audit()
    with (EVAL / 'audit.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    plot_saved()
    print(json.dumps({'passed': result['passed'],
        'checks_passed': result['checks_passed'],
        'pretraining_files': result['pretraining_files'],
        'preevaluation_files': result['preevaluation_files'],
        'audit_optimizer_steps': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
