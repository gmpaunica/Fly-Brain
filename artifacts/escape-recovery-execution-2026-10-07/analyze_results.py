"""Saved-only comparison of the implemented candidate with its historical baseline."""
import csv
import gzip
import json
from pathlib import Path
from collections import defaultdict

RUN=Path('experiments/escape-recovery-execution-2026-10-07/run')
ART=Path('artifacts/escape-recovery-execution-2026-10-07')


def main():
    results=json.loads((RUN/'results.json').read_text())
    historical=json.loads(Path('experiments/turnflee-2026-10-06/results.json').read_text())['validation']
    old={r['seed']:r for r in historical};failed={r['seed'] for r in results['validation'] if not r['success']}
    rows=[];groups=defaultdict(list)
    with gzip.open(RUN/'traces.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['phase']=='validation' and r['seed'] in failed:groups[r['seed']].append(r)
    for case in results['validation']:
        if case['seed'] not in failed:continue
        records=groups[case['seed']]
        trigger=next((r for r in records if r['triggers']),None)
        onset=next((r for r in records if r['threat']['strong']),None)
        rows.append(dict(seed=case['seed'],azimuth=case['azimuth'],elevation=case['elevation'],profile=case['profile'],
            old_success=old[case['seed']]['success'],new_success=case['success'],selected=case['side'],
            strong_seconds=case['first_strong_seconds'],trigger_seconds=case['trigger_seconds'],
            cumulative_at_strong=onset['temporal']['cumulative'] if onset else None,
            temporal_scores_at_trigger={k:trigger['temporal'][k] for k in ('left','right','candidate','streak')} if trigger else None,
            rolling_scores_at_trigger=trigger['wing_scores'] if trigger else None))
    with (ART/'failed-cases.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    azimuths=sorted({r['azimuth'] for r in results['validation']})
    metrics=dict(status=results['status'],historical_correct=sum(r['success'] for r in historical),
        candidate_correct=results['validation_successes'],old_failures_repaired=sum(r['success'] for r in results['validation'] if not old[r['seed']]['success']),
        new_regressions=len(rows),per_azimuth={str(a):sum(r['success'] for r in results['validation'] if r['azimuth']==a) for a in azimuths},
        failures=rows,closed_cases=len(results['closed_loop']),replacement_decoder_evaluations=1,extra_neural_steps=0)
    (ART/'analysis.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
