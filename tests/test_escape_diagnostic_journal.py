"""Reproduce Windows rename sharing and verify append-only reservation safety."""
import importlib
import json
import os
from pathlib import Path
import pytest


def api():
    assert importlib.util.find_spec('flybrain.escape_diagnostic_journal'), 'append-only I/O repair required'
    return importlib.import_module('flybrain.escape_diagnostic_journal')


def test_reader_cannot_break_persistent_append_reservations(tmp_path):
    m=api(); b=m.ReservationJournal(tmp_path/'progress.json',limit=80)
    with b.journal_path.open('rb') as reader:
        b.reserve_interval(); b.reserve_interval()
        with pytest.raises(ValueError,match='budget'): b.reserve_interval()
        assert b.steps==80
        reader.seek(0); rows=[json.loads(line) for line in reader.read().splitlines()]
        assert [r['reserved_steps'] for r in rows]==[40,80]
    b.close()
    assert json.loads((tmp_path/'progress.json').read_text())['limit']==80


def test_failed_fsync_poison_prevents_retry_and_does_not_commit_counter(tmp_path,monkeypatch):
    m=api(); b=m.ReservationJournal(tmp_path/'progress.json',limit=80)
    def failed(fd): raise OSError('durability failure')
    monkeypatch.setattr(m.os,'fsync',failed)
    with pytest.raises(OSError): b.reserve_interval()
    assert b.steps==0
    with pytest.raises(RuntimeError,match='failed'): b.reserve_interval()
    b.close()
    rows=(tmp_path/'progress.reservations.jsonl').read_text().splitlines()
    assert len(rows)==1 and json.loads(rows[0])['reserved_steps']==40


def test_no_overwrite_or_resume_existing_journal(tmp_path):
    m=api(); b=m.ReservationJournal(tmp_path/'progress.json',limit=40); b.reserve_interval(); b.close()
    with pytest.raises(FileExistsError): m.ReservationJournal(tmp_path/'progress.json',limit=40)


def test_heartbeat_appends_under_open_reader_and_terminal_file_is_fresh(tmp_path):
    m=api(); h=m.HeartbeatJournal(tmp_path/'worker')
    with h.path.open('rb') as reader:
        h.observe({'status':'running','child_pid':1})
        h.observe({'status':'running','child_pid':1})
        h.finish({'status':'exited','exit_code':0})
        reader.seek(0); rows=[json.loads(line) for line in reader.read().splitlines()]
        assert [r['status'] for r in rows]==['running','running','exited']
    assert json.loads((tmp_path/'worker.json').read_text())['exit_code']==0
