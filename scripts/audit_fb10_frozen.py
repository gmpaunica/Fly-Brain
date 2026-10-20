"""Audit and plot the frozen assay from saved records; never simulate or fit."""

import csv
from dataclasses import asdict
import json
from pathlib import Path

from flybrain.training import banc_frozen_evaluation as runner
from flybrain.training.banc_readiness import save_new_json, verify_provenance
from flybrain.training.evaluation import EpisodeResult, summarize


def audit(output: Path) -> dict:
    """Cross-check episode identities, all saved formats, counters and gates."""
    p, digest = runner.load_protocol(output / 'protocol.json')
    r = json.loads((output / 'results.json').read_text(encoding='utf-8'))
    journal = [json.loads(line) for line in (output / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    central = json.loads(runner.CENTRAL_RECEIPT.read_text(encoding='utf-8'))
    local = json.loads((output / 'execution.json').read_text(encoding='utf-8'))
    checks = {
        'protocol_digest': r['protocol_sha256'] == digest,
        'receipts': central == local and central['protocol_sha256'] == digest and central['status'] == 'consumed',
        'eligibility': r['eligibility'] == p['eligibility'] and r['acceptance'] == p['acceptance'],
        'covered_failure_preserved': r['eligibility']['original_fit_training_gate_passed'] is False
            and r['eligibility']['covered_gates']['aggregate_absolute_turn_fit'] is False,
        'original_gates_pass': r['eligibility']['original_gates']['passed'] is True,
        'zero_updates': r['optimizer_steps'] == p['optimizer_steps'] == 0,
        'checkpoint': runner.sha256(runner.CANDIDATE / 'trained.pt') == r['checkpoint_sha256'],
        'frozen_integrity': all(r[k] is True for k in ('frozen_parameters_unchanged', 'effective_weights_unchanged', 'integrity_checks_passed')),
        'protocol_provenance': verify_provenance(p['provenance']),
    }
    expected_journal = []
    for split, groups in r['results'].items():
        rows = {name: [EpisodeResult(**row) for row in groups[name]] for name in runner.CONTROLLERS}
        expected_ids = [(c['case_id'], c['split'], c['seed']) for c in p['cases'][split]]
        checks[split + '_identities'] = all([(row.case_id, row.split, row.seed) for row in group] == expected_ids for group in rows.values())
        summaries = {name: summarize(group) for name, group in rows.items()}
        gates = runner.assess_split(split, {name: [row.outcome == 'success' for row in group] for name, group in rows.items()}, summaries)
        saved = json.loads((output / (split + '.json')).read_text(encoding='utf-8'))
        checks[split + '_summaries_gates'] = summaries == r['summaries'][split] and gates == r['gates'][split]
        checks[split + '_split_record'] = saved == {'gates': gates, 'summaries': summaries, 'results': groups}
        expected_csv = [dict(condition=name, **asdict(row)) for name, group in rows.items() for row in group]
        with (output / (split + '.csv')).open(newline='', encoding='utf-8') as stream:
            csv_rows = list(csv.DictReader(stream))
        checks[split + '_csv'] = len(csv_rows) == len(expected_csv) and all(
            set(a) == set(b) and all(a[k] == ('' if v is None else str(v)) for k, v in b.items())
            for a, b in zip(csv_rows, expected_csv))
        expected_journal.extend({'condition': name, 'result': asdict(row), 'neural_steps': row.ticks * 20}
                                for name, group in rows.items() for row in group)
    checks['journal_exact'] = journal == expected_journal
    ticks = sum(row['result']['ticks'] for row in journal)
    steps = sum(row['neural_steps'] for row in journal)
    checks['actual_counters'] = (len(journal) == r['actual_episodes'] <= 144
        and ticks == r['actual_world_ticks'] <= 86400 and steps == r['actual_neural_steps'] == 20 * ticks <= 1728000)
    final = 'final' in r['results']
    checks['validation_stop'] = final == r['gates']['validation']['passed'] == r['final_cases_evaluated']
    checks['scientific_result'] = r['fb10_gate_passed'] == r['gates'].get('final', {}).get('passed', False)
    snapshot = json.loads((runner.PROJECT_ROOT / '.fb10-frozen-session-before.json').read_text(encoding='utf-8'))
    checks['all_pre_session_files_preserved'] = verify_provenance(snapshot)
    checks['cases_now_historical'] = {c['seed'] for group in p['cases'].values() for c in group}.issubset(runner.historical_seeds())
    report = {'experiment': 'fb10_frozen_offline_audit', 'checks': checks, 'verified': all(checks.values()),
              'actual_world_ticks': 0, 'actual_neural_steps': 0, 'optimizer_steps': 0,
              'preserved_file_counts': {name: len(files) for name, files in snapshot.items()},
              'assay_world_ticks': ticks, 'assay_neural_steps': steps, 'assay_episodes': len(journal),
              'fb10_gate_passed': r['fb10_gate_passed']}
    save_new_json(output / 'audit.json', report)
    plot(output, r)
    return report


def plot(output: Path, result: dict) -> None:
    """Scientific figure from recorded arrivals and progress, including failures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    colors = ['#0284c7', '#94a3b8', '#f59e0b']
    for col, (split, count, threshold) in enumerate((('validation', 16, 13), ('final', 32, 26))):
        if split not in result['summaries']:
            for ax in axes[:, col]:
                ax.set_axis_off()
                ax.text(.5, .5, 'Final not evaluated\nValidation stop rule applied', ha='center', va='center', transform=ax.transAxes)
            continue
        summary = result['summaries'][split]
        arrivals = [summary[n]['successes'] for n in runner.CONTROLLERS]
        ax = axes[0, col]
        bars = ax.bar(runner.CONTROLLERS, arrivals, color=colors)
        ax.bar_label(bars, labels=[f'{n}/{count}' for n in arrivals], padding=3)
        ax.axhline(threshold, color='#be123c', ls='--', label=f'Minimum {threshold}/{count}')
        ax.set(title=f'{split.title()} arrivals', ylim=(0, count + 3), ylabel='Arrivals')
        ax.legend(fontsize=8)
        ax = axes[1, col]
        progress = [summary[n]['mean_progress_fraction'] for n in runner.CONTROLLERS]
        bars = ax.bar(runner.CONTROLLERS, progress, color=colors)
        ax.bar_label(bars, fmt='%.3f', padding=3)
        ax.axhline(0, color='#64748b', lw=.7)
        ax.set(title=f'{split.title()} all-case progress', ylabel='Mean progress fraction')
        ax.margins(y=.2)
    outcome = 'PASSED' if result['fb10_gate_passed'] else 'FAILED'
    fig.suptitle(f'FB-10 frozen assay: {outcome}\nOriginal-gate exception; covered training gate remains failed; zero updates', fontsize=12)
    fig.savefig(output / 'frozen_assay.png', dpi=160)
    fig.savefig(output / 'frozen_assay.svg')
    plt.close(fig)


if __name__ == '__main__':
    report = audit(runner.DEFAULT_OUTPUT)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['verified'] else 1)
