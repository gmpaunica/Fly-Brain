"""Offline-only acceptance checks; refuses brain/sensor function execution."""
from collections import Counter
import contextlib
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    """Audit saved data, test tamper rejection, and check review artifact integrity."""
    calls = Counter()
    def guard(frame, event, arg):
        if event != 'call':
            return
        filename = frame.f_code.co_filename.replace('\\', '/')
        if '/src/flybrain/' not in filename:
            return
        key = filename.split('/src/flybrain/')[1] + ':' + frame.f_code.co_name
        calls[key] += 1
        if not (filename.endswith('/turnflee/audit.py') or filename.endswith('/flight3d/anatomy.py')
                or frame.f_code.co_name == '<module>'):
            raise RuntimeError('Unexpected project execution: ' + key)
    sys.setprofile(guard)
    sys.argv = ['audit', 'experiments/turnflee-2026-10-06']
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_module('flybrain.turnflee.audit', run_name='__main__')
    audit = json.loads(buf.getvalue())
    (OUT/'audit-call-guard.json').write_text(json.dumps(dict(audit=audit,project_calls=dict(calls),
         brain_sensor_simulation_calls=0),indent=2)+'\n',encoding='utf-8')
    import pytest
    calls.clear()
    code = pytest.main(['tests/test_turnflee_audit.py','-q',
                       '--junitxml='+str(OUT/'offline-audit-test.xml')])
    sys.setprofile(None)
    assert code == 0
    (OUT/'offline-test-call-guard.json').write_text(json.dumps(dict(exit_code=int(code),
        brain_sensor_simulation_calls=0,project_calls=dict(calls)),indent=2)+'\n',encoding='utf-8')
    preserved = json.loads((ROOT/'artifacts/turnflee-2026-10-06/preservation.json').read_text())
    before = json.loads((OUT/'protected-before.json').read_text())
    for collection in (preserved,before):
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in collection.items())
    def rows(name):
        with (OUT/name).open(newline='',encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    cases=rows('cases.csv'); contributions=rows('contributions.csv'); timelines=rows('timelines.csv'); comparisons=rows('comparisons.csv')
    assert (len(cases),len(contributions),len(timelines),len(comparisons))==(9,54,675,42)
    for c in cases:
        values=[int(c[k]) for k in ('i1_left','b3_left','b1_left','i1_right','b3_right','b1_right')]
        left,right=values[0]+values[1]+values[5],values[3]+values[4]+values[2]
        assert (left,right,right-left)==(int(c['left']),int(c['right']),int(c['difference']))
        assert math.sqrt(sum(x*x for x in values))==float(c['norm'])
        cells=[r for r in contributions if r['seed']==c['seed']]
        assert [int(r['window_actual'])-int(r['window_reference']) for r in cells]==values
        trigger=[r for r in timelines if r['seed']==c['seed'] and r['trigger_here']=='True']
        assert len(trigger)==1 and float(trigger[0]['seconds'])==float(c['trigger_s'])
        assert int(trigger[0]['difference'])==right-left
    report=ROOT/'docs/TURNFLEE-WRONG-SIDE-REVIEW.md'
    links=re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8'))
    assert all((report.parent/link).resolve().exists() for link in links)
    for p in ('PROJECT_STATUS.md','docs/BUILD_QUEUE.md','docs/decisions.md','docs/experiments.md'):
        assert 'TURNFLEE-WRONG-SIDE-REVIEW.md' in (ROOT/p).read_text(encoding='utf-8')
    final=dict(status='all-offline-checks-passed',saved_record_audit=audit['status'],
               offline_unit_tests_passed=1,protected_original_files=len(preserved),
               review_snapshot_files=len(before),changed_hashes=[],report_links_checked=len(links),
               cases=9,contribution_rows=54,timeline_rows=675,comparison_rows=42,
               brain_sensor_simulation_calls=0,presentations_added=0,simulated_seconds_added=0,
               paired_neural_steps_added=0,full_simulation_suite='not run: explicit zero budget')
    (OUT/'final-verification.json').write_text(json.dumps(final,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(final,indent=2))


if __name__=='__main__':
    main()
