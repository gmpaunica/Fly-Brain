"""Final audit/test/preservation checks; no new simulation or decoder candidate."""
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from flybrain.flight3d.anatomy import digest

ART=Path('artifacts/escape-recovery-execution-2026-10-07')
RUN=Path('experiments/escape-recovery-execution-2026-10-07/run')


def main():
    result=json.loads((RUN/'results.json').read_text());receipt=json.loads((RUN/'receipt.json').read_text())
    assert result['status']==receipt['status']=='closed-direction-failed'
    assert (result['cases'],result['validation_successes'],result['reserved_kernel_steps'])==(96,94,288000)
    assert not result['closed_loop'] and not result['retry_authorized']
    assert digest(RUN/'receipt.json')==json.loads((RUN.parent/'escape-recovery-seal.json').read_text())['receipt_sha256']
    for p,h in receipt['outputs'].items():assert digest(RUN/p)==h,p
    config=json.loads((RUN/'config.json').read_text())
    for p,h in config['candidate']['source_sha256']:
        assert digest(Path(p))==digest(RUN/'sources'/p)==h,p
    for p,h in config['protected'].items():assert digest(Path(p))==h,p
    originals=json.loads(Path('artifacts/turnflee-2026-10-06/preservation.json').read_text())
    snapshot=json.loads(Path('artifacts/turnflee-wrong-side-review-2026-10-06/protected-before.json').read_text())
    for values in (originals,snapshot):
        for p,h in values.items():assert digest(Path(p))==h,p
    for name,count in [('targeted',27),('full',812)]:
        x=ET.parse(ART/f'{name}.xml').getroot()[0]
        assert int(x.attrib['tests'])==count and all(x.attrib[k]=='0' for k in ('failures','errors','skipped'))
    analysis=json.loads((ART/'analysis.json').read_text());audit=json.loads((ART/'audit.json').read_text())
    assert audit['status']=='audit-passed' and audit['experiment_status']==result['status'] and audit['audit_steps']==0
    assert analysis['old_failures_repaired']==9 and analysis['new_regressions']==2 and analysis['per_azimuth']['86.25']==4
    assert analysis['failures'][0]['selected'] is None and analysis['failures'][1]['selected']=='right'
    assert json.loads(Path('experiments/escape-recovery-2026-10-07/results.json').read_text())['reserved_kernel_steps']==0
    assert not (ART/'activation.json').exists()
    assert json.loads(Path('artifacts/escape-diagnostic-2026-10-07/engineering.json').read_text())['steps']==1747
    ledgers=list((ART/'engineering-ledgers-final').glob('*.json'))
    explicit=sum(json.loads(p.read_text())['steps'] for p in ledgers)
    assert explicit==529
    report=Path('docs/ESCAPE-RECOVERY-RESULTS.md');links=re.findall(r'\]\(([^)]+)\)',report.read_text())
    assert all((report.parent/p).resolve().exists() for p in links)
    for p in ('PROJECT_STATUS.md','docs/BUILD_QUEUE.md','docs/decisions.md','docs/experiments.md'):
        assert 'ESCAPE-RECOVERY-RESULTS.md' in Path(p).read_text(encoding='utf-8')
    value=dict(status='final-handoff-verified',controller='implemented and evaluated once',scientific_status=result['status'],
        historical_correct=87,candidate_correct=94,old_failures_repaired=9,new_failures=2,
        failing_azimuth_degrees=86.25,azimuth_correct=4,azimuth_required=5,
        cases=96,scientific_seconds=144,individual_kernel_steps=288000,closed_cases=0,
        targeted_passed=27,full_passed=812,skips=0,original_files_unchanged=len(originals),snapshot_files_unchanged=len(snapshot),
        all_frozen_records_and_source_archives_preserved=True,diagnostic_engineering_steps_unchanged=1747,
        new_explicit_synthetic_verification_steps=explicit,zero_step_launch_failure_preserved=True,
        new_launcher='disabled because scientific gate failed',saved_visual='failure-timelines.png reviewed',live_visual='not performed',
        final_verification_steps=0,alternative_candidates=0,parameter_tuning=False,report_links_checked=len(links),
        report_sha256=digest(report),artifacts_sha256={p.relative_to(ART).as_posix():digest(p) for p in ART.rglob('*') if p.is_file() and p.name!='final-verification.json'})
    (ART/'final-verification.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in value.items() if k!='artifacts_sha256'},indent=2))


if __name__=='__main__':main()
