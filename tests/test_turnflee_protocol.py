"""Prevent extra candidates, duplicate validation cases and premature promotion."""
import importlib
import pytest


def api():
    assert importlib.util.find_spec('flybrain.turnflee.protocol'), 'locked experiment protocol required'
    return importlib.import_module('flybrain.turnflee.protocol')


def test_exact_protocol_budget_and_historical_audit(tmp_path):
    m=api(); calibration=m.cases('calibration'); validation=m.cases('validation')
    assert len(calibration)==16 and {c['elevation'] for c in calibration}=={0}
    assert len(validation)==96 and sorted({c['azimuth'] for c in validation})==[18.75+22.5*k for k in range(16)]
    assert len(m.closed_cases())==48
    assert 3*16*1.5+96*1.5+48*6==504
    audit=m.history_audit()
    assert audit['files'] and audit['calibration_historical']
    assert audit['validation_label']=='offset validation; no unseen claim'


def test_validation_requires_unique_cases_global_and_per_azimuth():
    m=api(); rows=[dict(c,success=True) for c in m.cases('validation')]
    assert m.validation_gate(rows)
    rows[0]['success']=False; assert m.validation_gate(rows)
    rows[1]['success']=False; assert not m.validation_gate(rows)
    rows=[dict(c,success=True) for c in m.cases('validation')]
    rows[-1]=rows[0].copy(); assert not m.validation_gate(rows)


@pytest.mark.parametrize('azimuth,side,expected',[(90,'left',True),(90,'right',False),(270,'right',True),(270,'left',False),(0,'right',True),(180,'left',True),(10,'right',True),(170,'right',True),(18.75,'right',False)])
def test_evasive_sign_and_front_rear_tolerance(azimuth,side,expected):
    assert api().correct_side(azimuth,side)==expected


def test_select_lowest_eligible_and_fail_closed():
    m=api(); rows=[dict(gains=g,eligible=False) for g in ((2,1),(4,2),(8,4))]
    assert m.select_candidate(rows) is None
    rows[2]['eligible']=True; rows[1]['eligible']=True
    assert m.select_candidate(rows)==(4,2)
    with pytest.raises(ValueError): m.select_candidate(rows[:2])
