"""Fresh read-only completion checks; no neural kernels or rendered inputs."""
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from flybrain.escape_diagnostic_recovery.audit import audit,sha

OUT=Path('artifacts/escape-diagnostic-recovery-2026-10-07')
RUN=Path('experiments/escape-diagnostic-recovery-2026-10-07')


def main() -> None:
    record=audit(RUN); assert record['status']=='closed-incomplete'
    receipt=record['receipt']; assert receipt['status']=='interrupted' and not receipt['retry_authorized']
    assert (receipt['completed_cases'],receipt['complete_neural_chunks'],receipt['recorded_intervals'],receipt['reserved_kernel_steps'])==(7,553,552,22120)
    config=json.loads((RUN/'config.json').read_text())
    for p,h in {**dict(config['response']['source_sha256']),**config['added_sources'],**config['test_sources']}.items():
        assert sha(Path(p))==sha(RUN/'sources'/p)==h,p
    for p,h in config['protected'].items(): assert sha(Path(p))==h,p
    original=json.loads(Path('artifacts/turnflee-2026-10-06/preservation.json').read_text())
    snapshot=json.loads(Path('artifacts/turnflee-wrong-side-review-2026-10-06/protected-before.json').read_text())
    for collection in (original,snapshot):
        for p,h in collection.items(): assert sha(Path(p))==h,p
    tests=ET.parse(OUT/'recovery-tests-final.xml').getroot()[0]
    assert tests.attrib['tests']=='21'
    assert all(tests.attrib[k]=='0' for k in ('failures','errors','skipped'))
    guard=json.loads((OUT/'offline-call-guard.json').read_text()); assert guard['added_kernel_steps']==0 and guard['exit_code']==0
    evidence=json.loads((OUT/'interruption-audit.json').read_text())
    assert evidence['all_records_equal_original'] and evidence['all_recorded_cell_edge_steps_exact']
    assert evidence['brain_sensor_simulation_calls']==evidence['added_audit_kernel_steps']==0
    assert evidence['cumulative_diagnostic_steps']==22160 and len(evidence['partial_five_case_evidence'])==2
    assert all(e['four_conditions_observed'] for e in evidence['partial_five_case_evidence'])
    assert json.loads(Path('artifacts/escape-diagnostic-2026-10-07/engineering.json').read_text())['steps']==1747
    for file,count in [('partial-case-status.csv',18),('partial-timelines.csv',553),('partial-deficit-evidence.csv',2)]:
        with (OUT/file).open(newline='',encoding='utf-8') as f: assert len(list(csv.DictReader(f)))==count
    assert json.loads((OUT/'process-probe-completed.json').read_text())['completed']
    links=0
    for name in ('ESCAPE-DIAGNOSTIC-RECOVERY-RESULTS.md','ESCAPE-DIAGNOSTIC-REMAINING-PROPOSAL.md'):
        p=Path('docs')/name
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            assert (p.parent/link).resolve().exists(),link; links+=1
    result=dict(status='final-handoff-checks-passed',comparator='fixed',scientific_status='interrupted',hypothesis='unresolved',
        tests_passed=21,added_engineering_steps=0,cumulative_engineering_steps=1747,
        completed_cases=7,complete_neural_chunks=553,full_historical_intervals=552,
        scientific_steps=22120,cumulative_diagnostic_steps=22160,simulated_seconds=11.06,cumulative_diagnostic_seconds=11.08,
        partial_deficit_cases_supported=2,required_deficit_cases=5,protected_original_files=len(original),
        protected_snapshot_files=len(snapshot),prior_run_and_artifacts_preserved=True,changed_files=[],
        source_archives_exact=True,final_verification_kernel_steps=0,report_links=links,
        saved_visual_review='partial-deficit-timelines.png inspected',live_review='not performed',
        full_suite='not run: explicit diagnostic restriction',activation='disabled',
        artifacts_sha256={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final-verification.json'})
    (OUT/'final-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'},indent=2))


if __name__=='__main__': main()
