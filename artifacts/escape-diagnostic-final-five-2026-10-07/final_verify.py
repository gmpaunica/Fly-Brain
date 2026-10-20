"""Final read-only diagnosis/repair-proposal handoff; zero new neural steps."""
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from flybrain.escape_diagnostic_final.contract import aggregate_sources
from flybrain.escape_diagnostic_final.audit import sha

OUT=Path('artifacts/escape-diagnostic-final-five-2026-10-07')
RUN=Path('experiments/escape-diagnostic-final-five-2026-10-07')


def main() -> None:
    receipt=json.loads((RUN/'receipt.json').read_text())
    assert (receipt['status'],receipt['completed_cases'],receipt['recorded_intervals'],receipt['reserved_kernel_steps'])==('completed',5,375,15000)
    seal=json.loads((RUN.parent/'escape-diagnostic-final-seal.json').read_text())
    assert seal['receipt_sha256']==sha(RUN/'receipt.json')
    for p,h in receipt['outputs'].items(): assert sha(RUN/p)==h,p
    assert {p.relative_to(RUN).as_posix() for p in RUN.rglob('*') if p.is_file()}==set(receipt['outputs'])|{'receipt.json'}
    config=json.loads((RUN/'config.json').read_text())
    for p,h in {**dict(config['response']['source_sha256']),**config['added_sources'],**config['test_sources']}.items():
        assert sha(Path(p))==sha(RUN/'sources'/p)==h,p
    for p,h in config['protected'].items(): assert sha(Path(p))==h,p
    collections=[json.loads(Path(p).read_text()) for p in ('artifacts/turnflee-2026-10-06/preservation.json','artifacts/turnflee-wrong-side-review-2026-10-06/protected-before.json')]
    for values in collections:
        for p,h in values.items(): assert sha(Path(p))==h,p
    journal=[json.loads(line) for line in (RUN/'progress.reservations.jsonl').read_text().splitlines()]
    assert len(journal)==375 and [r['reserved_steps'] for r in journal]==list(range(40,15001,40))
    assert all(r['limit']==15000 for r in journal)
    assert json.loads((OUT/'worker.json').read_text())['exit_code']==0
    evidence=json.loads((OUT/'analysis/audit.json').read_text())
    assert (evidence['status'],evidence['classification'],evidence['cases'],evidence['intervals'])==('audit-passed','supported',18,1350)
    assert evidence['recorded_kernel_steps']==54000 and evidence['cumulative_diagnostic_steps']==57360 and evidence['cumulative_attempted_steps']==57400
    assert evidence['audit_kernel_steps']==0
    five=evidence['all_five_criteria']; assert len(five)==5 and all(s['support_all_four'] for s in five)
    assert all(s['trigger']['six']==[0,-1,0,0,0,0] and s['selected']=='right' and s['final_cumulative_b3']>0 for s in five)
    x=ET.parse(OUT/'final-tests.xml').getroot()[0]
    assert x.attrib['tests']=='36' and all(x.attrib[k]=='0' for k in ('failures','errors','skipped'))
    assert json.loads((OUT/'offline-call-guard.json').read_text())['added_kernel_steps']==0
    assert json.loads((OUT/'audit-call-guard.json').read_text())['brain_sensor_simulation_calls']==0
    assert json.loads(Path('artifacts/escape-diagnostic-2026-10-07/engineering.json').read_text())['steps']==1747
    plan=aggregate_sources(Path(config['prior_completion']['directory']),Path(config['prior_remaining']['directory']),RUN,config['aggregate_cases'])
    assert len(plan)==18 and len({c['seed'] for _,_,c in plan})==18
    for root,ordinal,c in plan:
        with np.load(root/f'{ordinal:02d}-{c["seed"]}/000.npz',allow_pickle=False) as a:
            assert a['step_index'].dtype==np.int64
            for b in ('actual','reference'):
                for field in ('pre_voltage','post_voltage','pre_phase','post_phase','contribution','rates','history'):
                    assert a[b+'_'+field].dtype==np.float64
    for p,count in [(OUT/'analysis/cases.csv',18),(OUT/'analysis/timelines.csv',1350),(OUT/'pair-labels.csv',9),(OUT/'five-deficit-evidence.csv',5)]:
        with p.open(newline='',encoding='utf-8') as f: assert len(list(csv.DictReader(f)))==count
    links=0
    for name in ('ESCAPE-DIAGNOSTIC-COMPLETE.md','ESCAPE-TEMPORAL-READOUT-PROPOSAL.md'):
        p=Path('docs')/name
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            assert (p.parent/link).resolve().exists(),link;links+=1
    for p in ('PROJECT_STATUS.md','docs/BUILD_QUEUE.md','docs/decisions.md','docs/experiments.md'):
        assert 'ESCAPE-TEMPORAL-READOUT-PROPOSAL.md' in Path(p).read_text(encoding='utf-8')
    result=dict(status='complete-diagnosis-handoff-passed',hypothesis='supported observationally',aggregate_cases=18,
        aggregate_intervals=1350,aggregate_complete_steps=54000,final_run_steps=15000,cumulative_executed_steps=57360,
        cumulative_attempted_steps=57400,cumulative_executed_seconds=28.68,all_five_conditions_passed=True,
        tests_passed=36,added_engineering_steps=0,cumulative_engineering_steps=1747,
        original_files_unchanged=len(collections[0]),snapshot_files_unchanged=len(collections[1]),
        all_prior_sources_records_artifacts_preserved=True,source_archives_exact=True,changed_files=[],
        audit_neural_steps=0,final_verification_neural_steps=0,max_downstream_hz=max(s['max_downstream_hz'] for s in evidence['case_summaries']),
        saved_visual_review='five-deficit-timelines-final.png and tie-timelines.png inspected',live_review='not performed',
        full_suite='not run: explicit diagnostic restriction',escape='disabled; historical 87/96 unchanged',
        replacement_readout='one fixed proposal only; not implemented/scored',report_links_checked=links,
        proposal_sha256=sha(Path('docs/ESCAPE-TEMPORAL-READOUT-PROPOSAL.md')),
        report_sha256=sha(Path('docs/ESCAPE-DIAGNOSTIC-COMPLETE.md')),
        artifacts_sha256={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final-verification.json'})
    (OUT/'final-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'},indent=2))


if __name__=='__main__': main()
