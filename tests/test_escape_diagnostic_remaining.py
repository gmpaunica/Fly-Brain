"""Saved-only case/budget/source and process supervision behavior checks."""
import importlib
import json
from pathlib import Path
import sys
import pytest


def api():
    assert importlib.util.find_spec('flybrain.escape_diagnostic_remaining'), 'additive completion package required'
    return importlib.import_module('flybrain.escape_diagnostic_remaining.contract')


def test_remaining_order_exact_budget_and_aggregate_excludes_interrupted_prefix():
    m=api()
    assert m.REMAINING_SEEDS==(1100008,1100012,1100014,1100013,1100015,1100017,1100016,1100032,1100033,1100036,1100037)
    case_list=json.loads(Path('experiments/escape-diagnostic-recovery-2026-10-07/config.json').read_text())['cases']
    assert [c['seed'] for c in m.remaining_cases(case_list)]==list(m.REMAINING_SEEDS)
    plan=m.aggregate_sources(Path('prior'),Path('fresh'),case_list)
    assert len(plan)==18 and len({c['seed'] for _,_,c in plan})==18
    assert all(root==Path('prior') and ordinal==k for k,(root,ordinal,c) in enumerate(plan[:7]))
    assert all(root==Path('fresh') and ordinal==k for k,(root,ordinal,c) in enumerate(plan[7:]))
    assert plan[7][0]==Path('fresh') and plan[7][2]['seed']==1100008
    assert m.additional_budget(m.remaining_cases(case_list))==(16.5,33000)
    with pytest.raises(ValueError): m.remaining_cases(list(reversed(case_list)))
    with pytest.raises(ValueError): m.additional_budget(case_list)


def test_prior_receipt_binding_rejects_mutation(tmp_path):
    m=api(); p=tmp_path/'receipt.json'
    p.write_text(json.dumps(dict(status='interrupted',completed_cases=7,complete_neural_chunks=553,recorded_intervals=552,reserved_kernel_steps=22120)))
    bound=m.sha(p); m.verify_prior_receipt(p,bound)
    r=json.loads(p.read_text()); r['completed_cases']=8; p.write_text(json.dumps(r))
    with pytest.raises(ValueError): m.verify_prior_receipt(p,bound)


def test_new_budget_and_consume_cannot_retry(tmp_path):
    api(); r=importlib.import_module('flybrain.escape_diagnostic_remaining.runner')
    b=r.StepBudget(tmp_path/'budget.json')
    for _ in range(825): b.reserve_interval()
    assert b.steps==33000
    with pytest.raises(ValueError): b.reserve_interval()
    r.consume(tmp_path/'new')
    with pytest.raises(FileExistsError): r.consume(tmp_path/'renamed')


def test_supervisor_child_failure_seals_once_without_neural_calls(tmp_path):
    api(); w=importlib.import_module('flybrain.escape_diagnostic_remaining.worker')
    directory=tmp_path/'run'; directory.mkdir()
    (directory/'progress.json').write_text(json.dumps(dict(reserved_steps=40,limit=33000)))
    result=w.supervise([sys.executable,'-c','raise SystemExit(7)'],directory,tmp_path/'child.log',tmp_path/'heartbeat.json')
    assert result==7
    receipt=json.loads((directory/'receipt.json').read_text())
    assert receipt['status']=='interrupted' and receipt['reserved_kernel_steps']==40
    assert receipt['complete_neural_chunks']==0 and not receipt['retry_authorized']
    seal=tmp_path/'escape-diagnostic-remaining-seal.json'
    assert json.loads(seal.read_text())['receipt_sha256']==w.sha(directory/'receipt.json')
    bound=w.sha(directory/'receipt.json')
    w.seal_missing_receipt(directory,'second observation')
    assert w.sha(directory/'receipt.json')==bound


def test_supervisor_success_keeps_existing_receipt(tmp_path):
    api(); w=importlib.import_module('flybrain.escape_diagnostic_remaining.worker')
    directory=tmp_path/'run'; directory.mkdir()
    (directory/'receipt.json').write_text(json.dumps(dict(status='completed',reserved_kernel_steps=0)))
    before=w.sha(directory/'receipt.json')
    result=w.supervise([sys.executable,'-c','pass'],directory,tmp_path/'child.log',tmp_path/'heartbeat.json')
    assert result==0 and w.sha(directory/'receipt.json')==before
    assert json.loads((tmp_path/'heartbeat.json').read_text())['status']=='exited'
