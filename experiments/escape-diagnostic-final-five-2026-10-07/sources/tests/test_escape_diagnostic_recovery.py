"""Saved-only comparator and infrastructure tests; zero neural kernel calls."""
from copy import deepcopy
import gzip
import importlib
import json
from pathlib import Path
import pytest

OLD=Path('experiments/escape-diagnostic-2026-10-07')


def comparator():
    assert importlib.util.find_spec('flybrain.escape_diagnostic_recovery'), 'additive recovery required'
    return importlib.import_module('flybrain.escape_diagnostic_recovery.comparator')


def baseline():
    with gzip.open(OLD/'00-1100000/intervals.jsonl.gz','rt') as f:
        return json.loads(next(f))['original']


def test_accepts_all_runtime_tuple_fields_without_mutating_input():
    m=comparator(); expected=baseline(); runtime=deepcopy(expected)
    runtime['pathway']={k:tuple(v) if isinstance(v,list) else v for k,v in runtime['pathway'].items()}
    before=deepcopy(runtime)
    assert runtime!=expected
    assert m.assert_historical_equal(runtime,expected)==expected
    assert runtime==before and isinstance(runtime['pathway']['current_sum'],tuple)


@pytest.mark.parametrize('field', ['rgb','actual','reference','current','voltage','phase','timing','rolling','score','threat','side','missing'])
def test_rejects_real_record_mutations(field):
    m=comparator(); old=baseline(); changed=deepcopy(old)
    if field=='rgb': changed['eyes_sha256']='0'*64
    elif field=='actual': changed['actual_counts'][0]+=1
    elif field=='reference': changed['reference_counts'][0]+=1
    elif field=='current': changed['pathway']['current_sum'][0]+=1e-15
    elif field=='voltage': changed['pathway']['peak_voltage'][0]+=1e-15
    elif field=='phase': changed['pathway']['peak_phase'][0]+=1e-15
    elif field=='timing': changed['pathway']['first_spike_seconds'][0]=.001
    elif field=='rolling': changed['wing_vector'][0]+=1
    elif field=='score': changed['wing_scores']['left']+=1
    elif field=='threat': changed['threat']['strong']=not changed['threat']['strong']
    elif field=='side': changed['selected_side']='left'
    else: del changed['pathway']['spike_counts']
    with pytest.raises(ValueError,match='historical mismatch'): m.assert_historical_equal(changed,old)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_rejects_nonfinite_values_before_equality(value):
    m=comparator(); old=baseline(); changed=deepcopy(old)
    changed['pathway']['peak_voltage'][0]=value
    with pytest.raises(ValueError): m.assert_historical_equal(changed,old)


def test_real_number_nextafter_is_not_rounded_or_tolerated():
    m=comparator(); import math
    old=baseline(); changed=deepcopy(old)
    changed['end_seconds']=math.nextafter(old['end_seconds'],1.)
    with pytest.raises(ValueError): m.assert_historical_equal(changed,old)


def runner():
    comparator()
    return importlib.import_module('flybrain.escape_diagnostic_recovery.runner')


def test_new_lock_refuses_renamed_retry_and_keeps_old_lock(tmp_path):
    m=runner(); original=tmp_path/'escape-diagnostic-once.json'; original.write_bytes(b'old consumed record')
    m.consume(tmp_path/'v2')
    assert original.read_bytes()==b'old consumed record'
    with pytest.raises(FileExistsError): m.consume(tmp_path/'renamed')


def test_interruption_seals_before_brain_construction(tmp_path,monkeypatch):
    m=runner()
    monkeypatch.setattr(m,'preflight',lambda: {})
    monkeypatch.setattr(m,'historical_records',lambda: {})
    original_write=m.write
    def interrupted_write(path,value,fresh=True):
        if path.name=='config.json': raise KeyboardInterrupt()
        original_write(path,value,fresh)
    monkeypatch.setattr(m,'write',interrupted_write)
    # Fault at the configuration write boundary, before any brain or kernel.
    with pytest.raises(KeyboardInterrupt): m.run(tmp_path/'v2')
    r=json.loads((tmp_path/'v2/receipt.json').read_text())
    assert r['status']=='interrupted' and r['reserved_kernel_steps']==0
    seal=json.loads((tmp_path/'escape-diagnostic-recovery-seal.json').read_text())
    assert seal['receipt_sha256']==m.sha(tmp_path/'v2/receipt.json')
    with pytest.raises(FileExistsError): m.consume(tmp_path/'retry')


def test_source_tampering_and_budget_excess_refuse_next_action(tmp_path):
    m=runner(); p=tmp_path/'source'; p.write_bytes(b'locked')
    bound={str(p):m.sha(p)}; m.verify_hashes(bound)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError): m.verify_hashes(bound)
    b=m.StepBudget(tmp_path/'progress.json',limit=40); b.reserve_interval()
    with pytest.raises(ValueError): b.reserve_interval()
    assert b.steps==40


def test_v2_audit_reads_fresh_seal_and_rejects_tampering(tmp_path):
    comparator()
    import shutil
    audit=importlib.import_module('flybrain.escape_diagnostic_recovery.audit')
    saved=tmp_path/'saved'; shutil.copytree(OLD,saved)
    seal={'receipt_sha256':audit.sha(saved/'receipt.json')}
    (tmp_path/'escape-diagnostic-recovery-seal.json').write_text(json.dumps(seal))
    result=audit.audit(saved)
    assert result['status']=='closed-incomplete' and result['audit_kernel_steps']==0
    (saved/'00-1100000/000.npz').write_bytes(b'forged')
    with pytest.raises(ValueError,match='hash'): audit.audit(saved)
