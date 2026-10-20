"""Final saved-only failure/I/O-repair handoff verification; zero neural steps."""
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from flybrain.escape_diagnostic_remaining.audit import audit,sha

OUT=Path('artifacts/escape-diagnostic-remaining-2026-10-07')
RUN=Path('experiments/escape-diagnostic-remaining-2026-10-07')


def main() -> None:
    sealed=audit(RUN); assert sealed['status']=='closed-incomplete'
    receipt=sealed['receipt']
    assert (receipt['status'],receipt['completed_cases'],receipt['recorded_intervals'],receipt['reserved_kernel_steps'])==('failed',6,505,20240)
    assert not receipt['retry_authorized'] and 'WinError 5' in receipt['error']
    assert json.loads((RUN/'progress.json').read_text())['reserved_steps']==20200
    assert json.loads((RUN/'progress.tmp').read_text())['reserved_steps']==20240
    config=json.loads((RUN/'config.json').read_text())
    for p,h in {**dict(config['response']['source_sha256']),**config['added_sources'],**config['test_sources']}.items():
        assert sha(Path(p))==sha(RUN/'sources'/p)==h,p
    for p,h in config['protected'].items(): assert sha(Path(p))==h,p
    collections=[json.loads(Path(p).read_text()) for p in ('artifacts/turnflee-2026-10-06/preservation.json','artifacts/turnflee-wrong-side-review-2026-10-06/protected-before.json')]
    for values in collections:
        for p,h in values.items(): assert sha(Path(p))==h,p
    evidence=json.loads((OUT/'failure-audit-final.json').read_text())
    assert (evidence['complete_neural_chunks'],evidence['full_historical_records'],evidence['individual_kernel_steps'])==(505,505,20200)
    assert evidence['all_records_equal_original'] and evidence['all_recorded_cell_edge_steps_exact']
    assert evidence['added_audit_kernel_steps']==evidence['brain_sensor_simulation_calls']==0
    assert evidence['cumulative_diagnostic_steps']==42360 and evidence['unexecuted_attempted_steps']==40
    prior=json.loads(Path('artifacts/escape-diagnostic-recovery-2026-10-07/interruption-audit.json').read_text())
    four=prior['partial_five_case_evidence']+evidence['partial_five_case_evidence']
    assert len(four)==4 and all(r['four_conditions_observed'] for r in four)
    x=ET.parse(OUT/'journal-guarded-tests.xml').getroot()[0]
    assert x.attrib['tests']=='30' and all(x.attrib[k]=='0' for k in ('failures','errors','skipped'))
    assert json.loads((OUT/'journal-call-guard.json').read_text())['added_kernel_steps']==0
    assert json.loads(Path('artifacts/escape-diagnostic-2026-10-07/engineering.json').read_text())['steps']==1747
    for name,count in [('final-partial-timelines.csv',505),('final-partial-case-status.csv',11),('final-partial-deficit-evidence.csv',2)]:
        with (OUT/name).open(newline='',encoding='utf-8') as f: assert len(list(csv.DictReader(f)))==count
    assert json.loads((OUT/'worker.json').read_text())['status']=='exited'
    links=0
    for name in ('ESCAPE-DIAGNOSTIC-JOURNAL-RESULTS.md','ESCAPE-DIAGNOSTIC-FINAL-FIVE-PROPOSAL.md'):
        p=Path('docs')/name
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            assert (p.parent/link).resolve().exists(),link; links+=1
    result=dict(status='final-handoff-checks-passed',scientific_status='closed-file-I/O-failure',hypothesis='unresolved',
        complete_aggregate_cases=13,required_cases=18,supported_deficit_cases=4,required_deficit_cases=5,
        this_run_executed_steps=20200,this_run_attempted_steps=20240,cumulative_executed_steps=42360,
        cumulative_attempted_steps=42400,cumulative_executed_seconds=21.18,
        tests_passed=30,added_engineering_steps=0,cumulative_engineering_steps=1747,
        original_files_unchanged=len(collections[0]),snapshot_files_unchanged=len(collections[1]),
        all_prior_runs_and_artifacts_preserved=True,source_archives_exact=True,changed_files=[],
        I_O_repair='append-only utility implemented/tested; consumed runner preserved; no new science',
        repair_source_sha256=sha(Path('src/flybrain/escape_diagnostic_journal.py')),
        repair_tests_sha256=sha(Path('tests/test_escape_diagnostic_journal.py')),
        saved_visual_review='four-deficit-timelines.png inspected',live_review='not performed',
        full_suite='not run: explicit diagnostic restriction',activation='disabled',links_checked=links,
        final_verification_kernel_steps=0,
        artifacts_sha256={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final-verification.json'})
    (OUT/'final-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'},indent=2))


if __name__=='__main__': main()
