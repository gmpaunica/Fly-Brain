"""Integrity checks for a proposed final-only, one-use recovery assay."""

from dataclasses import asdict
import csv
import hashlib
import json

import pytest

from flybrain.training.food_recovery import (
    CASE_MANIFEST_SHA256, DEFAULT_OUTPUT, _append_episode, audit_journal,
    prepare_recovery, recovery_cases, run_recovery,
)
from flybrain.training.food_fresh import final_gates


def _row(case, controller: str) -> dict:
    return {**asdict(case), 'heading_sector': int(case.heading / (3.141592653589793 / 4)),
            'success': controller == 'trained', 'world_ticks': 1000,
            'neural_steps': 20000}


def test_proposed_manifest_is_exactly_pinned_and_balanced():
    cases = recovery_cases()
    assert len(cases) == 64
    assert len({case.seed for case in cases}) == 64
    assert [case.seed for case in cases] == list(range(840000, 840064))
    assert [sum(int(case.heading / (3.141592653589793 / 4)) == sector
                for case in cases) for sector in range(8)] == [8] * 8
    assert len(CASE_MANIFEST_SHA256) == 64


def test_durable_episode_journal_and_exact_audit(tmp_path):
    cases = recovery_cases()
    journal = tmp_path / 'episodes.jsonl'
    with journal.open('x', encoding='utf-8') as stream:
        for case in cases:
            for controller in ('trained', 'untrained', 'fixed'):
                _append_episode(stream, controller, _row(case, controller))
    rows = audit_journal(journal, cases)
    assert len(rows) == 192
    assert sum(row['result']['success'] for row in rows
               if row['controller'] == 'trained') == 64
    assert all(row['result']['world_ticks'] == 1000 for row in rows)

    rows[1]['result']['seed'] = 999
    journal.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    with pytest.raises(ValueError, match='case identity'):
        audit_journal(journal, cases)


def test_incomplete_journal_cannot_be_summarized(tmp_path):
    journal = tmp_path / 'episodes.jsonl'
    cases = recovery_cases()
    with journal.open('x', encoding='utf-8') as stream:
        _append_episode(stream, 'trained', _row(cases[0], 'trained'))
    with pytest.raises(ValueError, match='192'):
        audit_journal(journal, cases)


def test_preparation_pins_existing_evidence_without_running_cases(tmp_path):
    output = tmp_path / 'recovery'
    protocol_path = prepare_recovery(output)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    assert protocol['cases_manifest_sha256'] == CASE_MANIFEST_SHA256
    assert len(protocol['cases']) == 64
    assert protocol['budget'] == {
        'episodes': 192, 'world_ticks': 192000,
        'neural_steps': 3840000, 'optimizer_updates': 0}
    assert not (output / 'execution.json').exists()
    assert not (output / 'episodes.jsonl').exists()
    with pytest.raises(FileExistsError):
        prepare_recovery(output)


def test_consumed_receipt_blocks_replay_before_any_episode(tmp_path):
    output = tmp_path / 'recovery'
    prepare_recovery(output)
    (output / 'execution.json').write_text('{"status":"consumed"}\n', encoding='utf-8')
    with pytest.raises(FileExistsError):
        run_recovery(output)
    assert not (output / 'episodes.jsonl').exists()


def test_recorded_final_assay_reconciles_without_reexecution():
    directory = DEFAULT_OUTPUT
    digest = lambda name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    receipt = json.loads((directory / 'execution.json').read_text(encoding='utf-8'))
    result = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
    journal = audit_journal(directory / 'episodes.jsonl', recovery_cases())
    final = json.loads((directory / 'final.json').read_text(encoding='utf-8'))
    with (directory / 'final.csv').open(encoding='utf-8', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))

    assert receipt['status'] == 'consumed'
    assert receipt['protocol_sha256'] == result['protocol_sha256'] == digest('protocol.json')
    assert protocol['cases_manifest_sha256'] == result['case_manifest_sha256'] == CASE_MANIFEST_SHA256
    assert result['journal_sha256'] == digest('episodes.jsonl')
    assert result['final_json_sha256'] == digest('final.json')
    assert result['final_csv_sha256'] == digest('final.csv')
    assert len(journal) == len(csv_rows) == 192
    assert [row['controller'] for row in journal] == [row['controller'] for row in csv_rows]
    for name in ('trained', 'untrained', 'fixed'):
        assert final[name] == [row['result'] for row in journal if row['controller'] == name]
    recomputed = final_gates(final['trained'], final['untrained'], final['fixed'])
    recomputed['sector_successes'] = {str(key): value for key, value in
                                      recomputed['sector_successes'].items()}
    assert recomputed == result['final_gates']
    assert result['final_gates']['trained_successes'] == 63
    assert sorted(result['final_gates']['sector_successes'].values()) == [7, 8, 8, 8, 8, 8, 8, 8]
    assert all(pair['successes'] == 0 and pair['gain'] == 63 and pair['passed']
               for pair in result['final_gates']['paired'].values())
    assert result['parameters_unchanged'] is True
    assert result['contact_intervention_changes_stop'] is True
    assert result['actual_episodes'] == 192
    assert result['actual_world_ticks'] == 192000
    assert result['actual_neural_steps'] == 3840000
    assert result['optimizer_steps'] == 0
    assert result['passed'] is True
