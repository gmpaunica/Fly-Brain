"""Reconstruct frozen turnflee records only; never import simulation modules.

Run with .venv/Scripts/python.exe scripts/review_turnflee_wrong_side.py.
Outputs are review artifacts, not new experimental observations.
"""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'experiments/turnflee-2026-10-06'
OUT = ROOT / 'artifacts/turnflee-wrong-side-review-2026-10-06'
LABELS = ('i1_left', 'b3_left', 'b1_left', 'i1_right', 'b3_right', 'b1_right')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity(row: dict) -> tuple:
    return tuple(row[k] for k in ('phase', 'candidate', 'azimuth', 'elevation', 'profile', 'seed'))


def main() -> None:
    """Validate arithmetic, export comparisons/diagnostics, and plot saved scores."""
    OUT.mkdir(exist_ok=True)
    # Reject any accidental brain/sensor import, even if a future edit adds one.
    def deny_simulation(event: str, args: tuple) -> None:
        if event == 'import' and str(args[0]).startswith('flybrain'):
            raise RuntimeError('This review cannot import flybrain or call simulation')
    sys.addaudithook(deny_simulation)
    protected = json.loads((ROOT/'artifacts/turnflee-2026-10-06/preservation.json').read_text())
    paths = set(protected)
    for folder in ('src', 'tests', 'data', 'experiments'):
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'scripts').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts)
    paths.update(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
                 if p.is_file() and p.suffix.lower() in ('.bat', '.cmd', '.ps1')
                 and '.venv' not in p.parts)
    review_script = Path(__file__).relative_to(ROOT).as_posix()
    paths.discard(review_script)  # Newly authored reviewer is not a frozen input.
    before = {p: sha(ROOT/p) for p in sorted(paths)}
    snapshot = OUT/'protected-before.json'
    if snapshot.exists():
        saved = json.loads(snapshot.read_text())
        saved.pop(review_script, None)
        assert saved == before, 'protected bytes changed since review start'
    else:
        write_json(snapshot.name, before)
    assert all(before[p] == h for p,h in protected.items())
    result = json.loads((FROZEN/'results.json').read_text())
    config = json.loads((FROZEN/'config.json').read_text())
    receipt = json.loads((FROZEN/'receipt.json').read_text())
    assert all(sha(FROZEN/p) == h for p,h in receipt['sha256'].items())
    assert sha(FROZEN/'config.json') == json.loads((FROZEN/'lock.json').read_text())['config_sha256']
    assert all(sha(FROZEN/'sources'/p) == sha(ROOT/p) == h for p,h in config['source_sha256'].items())
    assert all(sha(ROOT/p) == h for p,h in config['history_audit']['files'].items())
    assert sha(ROOT/'data/flight3d/circuit.json') == config['circuit_sha256']
    parent = json.loads((ROOT/'experiments/turnflee-once.json').read_text())
    assert parent['consumed'] and not parent['retry_authorized']
    assert Path(parent['directory']).resolve() == FROZEN.resolve()
    response = json.loads((FROZEN/'response.json').read_text())
    assert response['gains'] == result['selected_gains'] == [4,2]
    ref_name, ref_hash = response['verification_receipt'].rsplit('#',1)
    assert sha(ROOT/ref_name) == ref_hash
    manifest = json.loads((ROOT/'data/flight3d/circuit.json').read_text())
    nodes = manifest['nodes']
    idx = {f"{n['metadata']['cell_type']}_{n['metadata']['side']}": i
           for i,n in enumerate(nodes) if n['metadata'].get('cell_type') in ('i1','b3','b1')}
    six = [idx[k] for k in LABELS]
    wings = [i for i,n in enumerate(nodes) if set(n['roles']) & {'wing_power','wing_steering'}]
    dns = [i for i,n in enumerate(nodes) if n['metadata'].get('cell_type') in ('DNp03','DNp26')]
    visual = {s:[i for i,n in enumerate(nodes) if 'vision' in n['roles'] and n['metadata']['side']==s]
              for s in ('left','right')}
    assert len(six)==6 and (len(visual['left']),len(visual['right']))==(46,50)
    groups = defaultdict(list)
    with gzip.open(FROZEN/'traces.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            groups[identity(r)].append(r)
    summaries = [dict(r,candidate=c) for c,x in enumerate(result['calibration']) for r in x['presentations']]
    summaries += result['validation']
    derived = {}
    ties = []
    for case in summaries:
        key = identity(case)
        records = groups[key]
        assert len(records)==75
        onset = trigger = side = None
        pending = -1
        timeline = []
        for j,r in enumerate(records):
            assert r['interval']==j and r['end_seconds']==(j+1)*.02
            window = records[max(0,j-9):j+1]
            a = [sum(x['actual_counts'][i] for x in window) for i in six]
            b = [sum(x['reference_counts'][i] for x in window) for i in six]
            v = [x-y for x,y in zip(a,b)]
            all_wings = [sum(x['actual_counts'][i]-x['reference_counts'][i] for x in window) for i in wings]
            assert all_wings == r['wing_vector']
            left, right = v[0]+v[1]+v[5], v[3]+v[4]+v[2]
            norm = math.sqrt(sum(x*x for x in v))
            bilateral = any(v[:3]) and any(v[3:])
            assert r['wing_scores']==dict(left=left,right=right,strength=norm,bilateral=bilateral)
            assert r['pathway']['spike_counts']==r['actual_counts']
            t = r['end_seconds']
            if r['threat']['strong']:
                pending = j+1+25
                if onset is None: onset = t
            eligible = j+1<=pending and norm>=1 and (left!=right or bilateral)
            if trigger is None and eligible:
                trigger=t
                side = ('right' if right>left else 'left') if right!=left else random.Random(case['seed']).choice(('left','right'))
                assert r['triggers']==1 and r['selected_side']==side
                if left==right: ties.append(dict(case,seeded_choice=side))
            assert r['triggers']==int(trigger is not None)
            assert r['selected_side']==side
            timeline.append(dict(seconds=t,left=left,right=right,difference=right-left,norm=norm,
                                 bilateral=bilateral,eligible=eligible,a=a,b=b,v=v))
        assert (onset,trigger,side)==(case['first_strong_seconds'],case['trigger_seconds'],case['side'])
        derived[key]=timeline
    failures = [c for c in result['validation'] if not c['success']]
    assert len(failures)==9
    def trigger_row(c: dict) -> dict:
        tl=derived[identity(c)]
        x=next(x for x in tl if x['seconds']==c['trigger_seconds'])
        later=[y for y in tl if y['seconds']>=x['seconds']]
        row=dict(phase=c['phase'],candidate=c['candidate'],seed=c['seed'],azimuth=c['azimuth'],
                 elevation=c['elevation'],profile=c['profile'],onset_s=c['first_strong_seconds'],
                 trigger_s=c['trigger_seconds'],latency_ms=round(1000*(c['trigger_seconds']-c['first_strong_seconds'])),
                 recorded_start_s=.02,recorded_end_s=tl[-1]['seconds'],post_trigger_ms=round(1000*(tl[-1]['seconds']-x['seconds'])),
                 **dict(zip(LABELS,x['v'])),left=x['left'],right=x['right'],difference=x['difference'],
                 norm=x['norm'],bilateral=x['bilateral'],tie=x['difference']==0,side=c['side'],
                 first_later_negative_s=next((y['seconds'] for y in later if y['seconds']>x['seconds'] and y['difference']<0),None),
                 later_positive_intervals=sum(y['difference']>0 for y in later[1:]),
                 later_zero_intervals=sum(y['difference']==0 for y in later[1:]),
                 later_negative_intervals=sum(y['difference']<0 for y in later[1:]),
                 final_left=tl[-1]['left'],final_right=tl[-1]['right'],final_difference=tl[-1]['difference'])
        return row
    case_rows=[trigger_row(c) for c in failures]
    write_csv('cases.csv',case_rows)
    contributions=[]; timelines=[]; comparisons=[]
    selected={identity(c):c for c in failures}
    for c in failures:
        tl=derived[identity(c)]; records=groups[identity(c)]
        tr=next(x for x in tl if x['seconds']==c['trigger_seconds'])
        for k,i in enumerate(six):
            at=records[round(c['trigger_seconds']/.02)-1]
            contributions.append(dict(seed=c['seed'],cell=LABELS[k],node_id=nodes[i]['id'],node_index=i,
                interval_actual=at['actual_counts'][i],interval_reference=at['reference_counts'][i],
                window_actual=tr['a'][k],window_reference=tr['b'][k],increment=tr['v'][k],
                contributes_to=('right' if k in (2,3,4) else 'left'),signed_contribution=tr['v'][k]))
        for x,r in zip(tl,records):
            timelines.append(dict(seed=c['seed'],seconds=x['seconds'],strong=r['threat']['strong'],
                trigger_here=x['seconds']==c['trigger_seconds'],left=x['left'],right=x['right'],
                difference=x['difference'],norm=x['norm'],bilateral=x['bilateral'],eligible=x['eligible'],
                **dict(zip(LABELS,x['v']))))
        peers=[p for p in result['validation'] if p['success'] and p['azimuth']==c['azimuth']]
        cal=[dict(p,candidate=1) for p in result['calibration'][1]['presentations'] if p['profile']==c['profile']]
        distance=lambda p:abs((p['azimuth']-c['azimuth']+180)%360-180)
        nearest=min(map(distance,cal))
        peers += [p for p in cal if distance(p)==nearest]
        assert len([p for p in peers if p['phase']=='calibration'])==1
        for p in peers:
            selected[identity(p)]=p
            peer_records=groups[identity(p)]
            def first_difference(indices: list[int], field: str) -> float | None:
                return next((a['end_seconds'] for a,b in zip(records,peer_records)
                             if any((a['pathway'][field][i] if field!='actual_counts' else a[field][i]) !=
                                    (b['pathway'][field][i] if field!='actual_counts' else b[field][i])
                                    for i in indices)),None)
            comparisons.append(dict(failure_seed=c['seed'],comparison_kind='same_azimuth_success' if p['phase']=='validation' else 'nearest_calibration',
                azimuth_difference=p['azimuth']-c['azimuth'],elevation_difference=p['elevation']-c['elevation'],
                same_profile=p['profile']==c['profile'],
                first_visual_phase_difference_s=first_difference(visual['left']+visual['right'],'peak_phase'),
                first_dn_current_difference_s=first_difference(dns,'current_sum'),
                first_dn_count_difference_s=first_difference(dns,'actual_counts'),
                first_endpoint_current_difference_s=first_difference(six,'current_sum'),
                first_endpoint_count_difference_s=first_difference(six,'actual_counts'),**trigger_row(p)))
    write_csv('contributions.csv',contributions)
    write_csv('timelines.csv',timelines)
    write_csv('comparisons.csv',comparisons)
    # Per-cell recorded diagnostics, without reconstructing missing currents/RGB.
    diagnostic=[]; fields=[]; stages=[]
    for key,c in selected.items():
        records=groups[key]
        first_visual_phase={s:None for s in visual}; first_visual_spike={s:None for s in visual}
        for side,indices in visual.items():
            for rank,i in enumerate(indices):
                fields.append(dict(phase=c['phase'],seed=c['seed'],azimuth=c['azimuth'],elevation=c['elevation'],profile=c['profile'],
                    side=side,node_id=nodes[i]['id'],rank=rank,row_start=rank%6*4,column_start=rank//6%8*6,
                    first_spike_s=next((r['pathway']['first_spike_seconds'][i] for r in records if r['pathway']['first_spike_seconds'][i] is not None),None),
                    total_actual=sum(r['actual_counts'][i] for r in records),total_reference=sum(r['reference_counts'][i] for r in records),
                    max_phase=max(r['pathway']['peak_phase'][i] for r in records)))
            first_visual_phase[side]=next((r['end_seconds'] for r in records if any(r['pathway']['peak_phase'][i]>0 for i in indices)),None)
            first_visual_spike[side]=min((r['pathway']['first_spike_seconds'][i] for r in records for i in indices
                                         if r['pathway']['first_spike_seconds'][i] is not None),default=None)
        stage=dict(phase=c['phase'],seed=c['seed'],azimuth=c['azimuth'],elevation=c['elevation'],profile=c['profile'],
                   left_visual_phase_interval_s=first_visual_phase['left'],right_visual_phase_interval_s=first_visual_phase['right'],
                   left_visual_first_spike_s=first_visual_spike['left'],right_visual_first_spike_s=first_visual_spike['right'],
                   left_visual_total=sum(r['actual_counts'][i] for r in records for i in visual['left']),
                   right_visual_total=sum(r['actual_counts'][i] for r in records for i in visual['right']))
        for i in dns+six:
            label=f"{nodes[i]['metadata']['cell_type']}_{nodes[i]['metadata']['side']}"
            stage[label+'_first_count_difference_s']=next((r['end_seconds'] for r in records if r['actual_counts'][i]!=r['reference_counts'][i]),None)
            stage[label+'_total_increment']=sum(r['actual_counts'][i]-r['reference_counts'][i] for r in records)
            for r in records:
                d=r['pathway']
                diagnostic.append(dict(phase=c['phase'],seed=c['seed'],azimuth=c['azimuth'],elevation=c['elevation'],profile=c['profile'],
                    seconds=r['end_seconds'],cell=label,node_id=nodes[i]['id'],actual=r['actual_counts'][i],reference=r['reference_counts'][i],
                    increment=r['actual_counts'][i]-r['reference_counts'][i],
                    **{k:d[k][i] for k in ('current_sum','current_peak','discarded_current','peak_voltage','first_spike_seconds')}))
        stages.append(stage)
    write_csv('pathway_cells.csv',diagnostic)
    write_csv('visual_fields.csv',fields)
    write_csv('stages.csv',stages)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True,sharey=True)
    for ax,c,summary in zip(axes.flat,failures,case_rows):
        tl=derived[identity(c)]
        ax.step([x['seconds'] for x in tl],[x['difference'] for x in tl],where='post',color='#722b83',lw=1.6,label='right minus left')
        ax.axhline(0,color='gray',lw=.6)
        ax.axvline(.5,color='gray',ls=':',lw=1,label='image series starts (first end .52 s)')
        ax.axvline(c['first_strong_seconds'],color='#d57900',ls='--',lw=2,label='first strong threat')
        ax.axvline(c['trigger_seconds'],color='#007c91',ls=':',lw=2,label='recorded trigger')
        ax.scatter([c['trigger_seconds']],[summary['difference']],color='#007c91',s=30,zorder=5)
        ax.set_title(f"{c['seed']} | az {c['azimuth']}°, el {c['elevation']}°, P{c['profile']}\ntrigger Δ={summary['difference']}, {'tie' if summary['tie'] else 'unequal'}",fontsize=10)
        ax.set_xlim(0,1.5); ax.grid(alpha=.15)
    fig.supxlabel('Recorded interval endpoint (seconds); no extrapolation beyond 1.50 s',y=.055)
    fig.supylabel('Right score − left score (signed spike increments)')
    fig.suptitle('Nine wrong-side first responses, frozen gains (4,2)\nPositive favors right; negative favors left. Later scores do not change the recorded first side.',fontsize=12)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(.5,.008))
    fig.tight_layout(rect=(.025,.085,1,.94))
    fig.savefig(OUT/'score-timelines.png',dpi=170)
    plt.close(fig)
    after={p:sha(ROOT/p) for p in before}
    assert before==after
    validation_ties=[c for c in ties if c['phase']=='validation']
    failed_ties=sum(r['tie'] for r in case_rows)
    assert len(validation_ties)==10 and failed_ties==3
    summary=dict(status='review-checks-passed',presentations_added=0,simulated_seconds_added=0,paired_neural_steps_added=0,
                 simulation_imports=0,intervals_reconstructed=sum(map(len,groups.values())),
                 failures=len(failures),failed_ties=failed_ties,unequal_score_failures=9-failed_ties,
                 validation_ties=validation_ties,comparison_rows=len(comparisons),unique_reviewed_cases=len(selected),
                 protected_original_files=len(protected),review_snapshot_files=len(before),changed_hashes=[],
                 review_script_sha256=sha(Path(__file__)),
                 frozen_status=result['status'],closed_loop_cases=len(result['closed_loop']))
    write_json('verification.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='validation_ties'},indent=2))


if __name__=='__main__':
    main()
