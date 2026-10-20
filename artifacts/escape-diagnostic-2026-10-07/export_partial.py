"""Reproducible saved-only tables and first-baseline timeline; no simulation."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np

ROOT=Path('experiments/escape-diagnostic-2026-10-07')
OUT=Path('artifacts/escape-diagnostic-2026-10-07')


def table(name,rows):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    """Display exactly the 20 recorded milliseconds, without extrapolation."""
    config=json.loads((ROOT/'config.json').read_text()); nodes=json.loads((ROOT/'circuit.json').read_text())['nodes']
    a=dict(np.load(ROOT/'00-1100000/000.npz',allow_pickle=False)); g=dict(np.load(ROOT/'graph.npz',allow_pickle=False))
    with gzip.open(ROOT/'00-1100000/intervals.jsonl.gz','rt') as f: interval=json.loads(next(f))
    rows=[]
    for k,c in enumerate(config['cases']):
        failure=config['cases'][k//2*2]; comparison=config['cases'][k//2*2+1]
        rows.append(dict(order=k+1,pair=k//2+1,role='failure' if k%2==0 else 'successful context',**c,
            elevation_difference_comparison_minus_failure=comparison['elevation']-failure['elevation'],
            profile_changed=comparison['profile']!=failure['profile'],
            status='stopped in baseline' if k==0 else 'not started',completed_intervals=int(k==0),
            simulated_seconds=.02 if k==0 else 0,individual_kernel_steps=40 if k==0 else 0,
            threat_intervals=0))
    table('case-execution.csv',rows)
    fields=('DNp03','DNp26','i1','b3','b1')
    selected=[i for i,n in enumerate(nodes) if n['metadata'].get('cell_type') in fields]
    rows=[]
    for t,step in enumerate(a['step_index']):
        for i in selected:
            rows.append(dict(step=int(step),node_id=nodes[i]['id'],cell=nodes[i]['metadata']['cell_type'],side=nodes[i]['metadata']['side'],
                actual_pre_voltage=float(a['actual_pre_voltage'][t,i]),reference_pre_voltage=float(a['reference_pre_voltage'][t,i]),
                actual_post_voltage=float(a['actual_post_voltage'][t,i]),reference_post_voltage=float(a['reference_post_voltage'][t,i]),
                actual_spike=int(a['actual_spikes'][t,i]),reference_spike=int(a['reference_spikes'][t,i]),
                cumulative_increment=int((a['actual_spikes'][:t+1,i].sum()-a['reference_spikes'][:t+1,i].sum()))))
    table('baseline-timeline.csv',rows)
    table('all-cell-baseline.csv',[dict(node_id=n['id'],cell=n['metadata'].get('cell_type'),side=n['metadata'].get('side'),
        actual_count=int(a['actual_spikes'][:,i].sum()),reference_count=int(a['reference_spikes'][:,i].sum()),
        actual_max_pre_voltage=float(a['actual_pre_voltage'][:,i].max()),reference_max_pre_voltage=float(a['reference_pre_voltage'][:,i].max()),
        cumulative_increment=int(interval['cumulative_increment'][i])) for i,n in enumerate(nodes)])
    table('all-edge-baseline.csv',[dict(pre_id=nodes[int(p)]['id'],post_id=nodes[int(q)]['id'],effective=bool(g['effective'][j]),
        effective_weight=float(g['weights'][j]),actual_contribution=float(a['actual_contribution'][:,j].sum()),
        reference_contribution=float(a['reference_contribution'][:,j].sum())) for j,(p,q) in enumerate(zip(g['pre'],g['post']))])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b3=next(i for i,n in enumerate(nodes) if n['metadata'].get('cell_type')=='b3' and n['metadata'].get('side')=='left')
    fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True,layout='constrained')
    axes[0].plot(a['step_index'],a['actual_post_voltage'][:,b3],label='actual',color='#8b4d9c',lw=3)
    axes[0].plot(a['step_index'],a['reference_post_voltage'][:,b3],label='motion-only reference',color='#067c92',ls='--')
    axes[0].set_ylabel('Left b3 post-reset voltage'); axes[0].legend()
    axes[1].step(a['step_index'],np.cumsum(a['actual_spikes'][:,b3].astype(int)-a['reference_spikes'][:,b3].astype(int)),where='post',color='#067c92')
    axes[1].set_ylabel('Cumulative b3 increment'); axes[1].set_xlabel('Recorded neural step / milliseconds')
    axes[1].set_ylim(-1,1); axes[1].set_xlim(1,20)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Stopped in first baseline interval: seed 1100000\nNo threat image; recorded 1–20 ms only. Timing hypothesis unassessed.',fontsize=12)
    fig.savefig(OUT/'baseline-timeline.png',dpi=160); plt.close(fig)
    print('Exported 18 case statuses, 200 pathway step rows, 589 cells and',len(g['pre']),'edges; zero simulation calls.')


if __name__=='__main__': main()
