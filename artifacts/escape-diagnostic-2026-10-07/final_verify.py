"""Read-only final handoff checks; zero neural steps or sensory presentations."""
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from flybrain.escape_diagnostic.audit import sha,audit

OUT=Path('artifacts/escape-diagnostic-2026-10-07')
RUN=Path('experiments/escape-diagnostic-2026-10-07')


def main():
    """Check final claims against sealed observations and preserved source bytes."""
    record=audit(RUN); assert record['status']=='closed-incomplete'
    receipt=record['receipt']; assert (receipt['completed_cases'],receipt['recorded_intervals'],receipt['reserved_kernel_steps'])==(0,1,40)
    assert receipt['status']=='failed' and not receipt['retry_authorized']
    config=json.loads((RUN/'config.json').read_text())
    for p,h in {**dict(config['response']['source_sha256']),**config['added_sources'],**config['test_sources']}.items():
        assert sha(Path(p))==sha(RUN/'sources'/p)==h,p
    original=json.loads(Path('artifacts/turnflee-2026-10-06/preservation.json').read_text())
    snapshot=json.loads(Path('artifacts/turnflee-wrong-side-review-2026-10-06/protected-before.json').read_text())
    for collection in (original,snapshot):
        for p,h in collection.items(): assert sha(Path(p))==h,p
    engineering=json.loads((OUT/'engineering.json').read_text())
    assert engineering['steps']==sum(c['steps'] for c in engineering['calls'])==1747
    assert engineering['steps']<=2000
    for name,want in [('preflight-tests.xml',10),('serialization-final.xml',6)]:
        tests=ET.parse(OUT/name).getroot()[0]
        assert int(tests.attrib['tests'])==want
        assert all(int(tests.attrib[k])==0 for k in ('failures','errors','skipped'))
    partial=json.loads((OUT/'partial-audit.json').read_text())
    assert partial['hypothesis']=='unresolved' and partial['saved_original_interval_exact']
    assert partial['complete_step_recurrence_exact'] and partial['brain_sensor_simulation_calls']==0
    assert partial['total_kernel_steps']==40 and partial['rendered_threat_intervals']==0
    for name,want in [('case-execution.csv',18),('baseline-timeline.csv',200),('all-cell-baseline.csv',589),('all-edge-baseline.csv',3636)]:
        with (OUT/name).open(newline='',encoding='utf-8') as f: assert len(list(csv.DictReader(f)))==want
    links=0
    for name in ('ESCAPE-DIAGNOSTIC.md','ESCAPE-DIAGNOSTIC-RECOVERY-PROPOSAL.md'):
        p=Path('docs')/name
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            assert (p.parent/link).resolve().exists(),link; links+=1
    for name in ('PROJECT_STATUS.md','docs/BUILD_QUEUE.md','docs/decisions.md','docs/experiments.md'):
        s=Path(name).read_text(encoding='utf-8')
        assert 'ESCAPE-DIAGNOSTIC-RECOVERY-PROPOSAL.md' in s
    hashes={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final-verification.json'}
    result=dict(status='handoff-checks-passed',scientific_status='closed-incomplete',hypothesis='unresolved',
        engineering_steps=1747,scientific_steps=40,simulated_seconds=.02,
        completed_presentations=0,threat_intervals=0,protected_original_files=len(original),
        protected_snapshot_files=len(snapshot),changed_files=[],source_archives_exact=True,
        final_verification_kernel_steps=0,saved_visual_review='baseline-timeline.png visually inspected',
        live_review='not performed',full_suite='not run: explicit user restriction',
        repair='proposal only; no comparator correction or neural repair implemented',
        links_checked=links,artifacts_sha256=hashes)
    (OUT/'final-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'},indent=2))


if __name__=='__main__': main()
