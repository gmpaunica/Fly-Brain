"""Final completion tests use saved metadata and ordinary child processes only."""
import importlib
import json
from pathlib import Path
import sys
import pytest


def api():
    assert importlib.util.find_spec('flybrain.escape_diagnostic_final'), 'fresh final-case package required'
    return importlib.import_module('flybrain.escape_diagnostic_final.contract')


def test_five_case_order_budget_and_three_source_aggregation():
    m=api(); cases=json.loads(Path('experiments/escape-diagnostic-recovery-2026-10-07/config.json').read_text())['cases']
    assert [c['seed'] for c in m.final_cases(cases)]==[1100016,1100032,1100033,1100036,1100037]
    assert m.additional_budget(m.final_cases(cases))==(7.5,15000)
    plan=m.aggregate_sources(Path('first'),Path('second'),Path('new'),cases)
    assert len(plan)==18 and len({c['seed'] for _,_,c in plan})==18
    assert [(p,k) for p,k,c in plan[:7]]==[(Path('first'),k) for k in range(7)]
    assert [(p,k) for p,k,c in plan[7:13]]==[(Path('second'),k) for k in range(6)]
    assert [(p,k) for p,k,c in plan[13:]]==[(Path('new'),k) for k in range(5)]
    with pytest.raises(ValueError): m.final_cases(cases[::-1])
    with pytest.raises(ValueError): m.additional_budget(cases)


def test_exact_failed_prior_receipt_and_durable_steps_bound(tmp_path):
    m=api(); path=tmp_path/'receipt.json'
    path.write_text(json.dumps(dict(status='failed',completed_cases=6,recorded_intervals=505,reserved_kernel_steps=20240,error='PermissionError: WinError 5')))
    m.verify_failed_prior(path,m.sha(path))
    path.write_text(path.read_text().replace('20240','20200'))
    with pytest.raises(ValueError): m.verify_failed_prior(path,'bad hash')


def test_journal_budget_under_held_reader_and_no_retry(tmp_path):
    api(); r=importlib.import_module('flybrain.escape_diagnostic_final.runner')
    budget=r.StepBudget(tmp_path/'progress.json')
    with budget.journal_path.open('rb') as reader:
        for _ in range(375): budget.reserve_interval()
        assert budget.steps==15000
        with pytest.raises(ValueError): budget.reserve_interval()
    budget.close()
    r.consume(tmp_path/'case-run')
    with pytest.raises(FileExistsError): r.consume(tmp_path/'retry')


def test_supervisor_append_heartbeat_and_failure_fallback(tmp_path):
    api(); w=importlib.import_module('flybrain.escape_diagnostic_final.worker')
    run=tmp_path/'run';run.mkdir()
    (run/'progress.json').write_text(json.dumps(dict(limit=15000,journal='progress.reservations.jsonl')))
    (run/'progress.reservations.jsonl').write_text('{"reserved_steps":40,"limit":15000}\n{"reserved_steps":')
    assert w.reservation_upper_bound(run)==80
    result=w.supervise([sys.executable,'-c','raise SystemExit(9)'],run,tmp_path/'child.log',tmp_path/'worker.json')
    assert result==9
    receipt=json.loads((run/'receipt.json').read_text())
    assert receipt['status']=='interrupted' and receipt['reserved_kernel_steps']==80 and not receipt['retry_authorized']
    assert receipt['recorded_kernel_steps_lower_bound']==0
    assert json.loads((tmp_path/'worker.json').read_text())['status']=='exited'
    assert json.loads((tmp_path/'escape-diagnostic-final-seal.json').read_text())['receipt_sha256']==w.sha(run/'receipt.json')
    before=w.sha(run/'receipt.json');w.seal_missing_receipt(run,'second');assert w.sha(run/'receipt.json')==before


def test_supervisor_keeps_normal_receipt_and_journal_terminal(tmp_path):
    api();w=importlib.import_module('flybrain.escape_diagnostic_final.worker')
    run=tmp_path/'run';run.mkdir();(run/'receipt.json').write_text('{"status":"completed"}')
    before=w.sha(run/'receipt.json')
    assert w.supervise([sys.executable,'-c','pass'],run,tmp_path/'child.log',tmp_path/'worker.json')==0
    assert w.sha(run/'receipt.json')==before
    rows=[json.loads(line) for line in (tmp_path/'worker.jsonl').read_text().splitlines()]
    assert rows[-1]['status']=='exited'


def test_heartbeat_failure_terminates_child_and_seals_without_retry(tmp_path,monkeypatch):
    api();w=importlib.import_module('flybrain.escape_diagnostic_final.worker')
    run=tmp_path/'run';run.mkdir()
    def failed(self,value): raise OSError('heartbeat durability failure')
    monkeypatch.setattr(w.HeartbeatJournal,'observe',failed)
    with pytest.raises(OSError):
        w.supervise([sys.executable,'-c','import time;time.sleep(60)'],run,tmp_path/'child.log',tmp_path/'worker.json')
    receipt=json.loads((run/'receipt.json').read_text())
    assert receipt['status']=='interrupted' and receipt['reserved_kernel_steps']==0
    assert not receipt['retry_authorized'] and 'supervision failed' in receipt['error']
