"""Reproducible tables/plots from audited records; no simulation imports."""
import csv
import json
from pathlib import Path
import numpy as np

OUT=Path('artifacts/escape-diagnostic-remaining-2026-10-07')


def table(path: Path, rows: list[dict]) -> None:
    with path.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def main() -> None:
    """Retain original first selections; never apply an alternative decoder."""
    result=json.loads((OUT/'analysis/audit.json').read_text())
    summaries=result['case_summaries']
    with (OUT/'analysis/timelines.csv').open(newline='',encoding='utf-8') as f:
        timelines=list(csv.DictReader(f))
    pairs=[]
    for k in range(0,18,2):
        a,b=summaries[k:k+2]
        pairs.append(dict(failure_seed=a['seed'],comparison_seed=b['seed'],azimuth=a['azimuth'],
            failure_elevation=a['elevation'],comparison_elevation=b['elevation'],
            elevation_difference=b['elevation']-a['elevation'],failure_profile=a['profile'],comparison_profile=b['profile'],
            profile_changed=a['profile']!=b['profile'],failure_trigger_step=a['trigger_step'],comparison_trigger_step=b['trigger_step'],
            failure_side=a['selected'],comparison_side=b['selected'],label='historical context; elevation/profile confounded'))
    table(OUT/'pair-labels.csv',pairs)
    evidence=[]
    for s in result['all_five_criteria']:
        t=s['trigger_step']; start=t-200
        aw=[v for v in s['b3_actual_spike_steps'] if start<v<=t]
        rw=[v for v in s['b3_reference_spike_steps'] if start<v<=t]
        exact=s['trigger']['b3_rolling']==len(aw)-len(rw)==-1
        assert exact
        evidence.append(dict(seed=s['seed'],trigger_step=t,window_left_open_step=start,window_right_closed_step=t,
            actual_b3_window_steps=aw,reference_b3_window_steps=rw,actual_count=len(aw),reference_count=len(rw),increment=-1,
            first_differential_right_DN_delivery_step=s['first_differential_right_DN_delivery_step'],
            first_b3_timing_difference_step=s['first_b3_timing_difference_step'],
            deficit_expiration_steps=s['deficit_expiration_steps'],final_cumulative_b3=s['final_cumulative_b3'],
            redistributed_before_trigger=s['b3_redistribution_before_trigger'],all_four=s['support_all_four']))
    table(OUT/'five-deficit-evidence.csv',evidence)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,2,figsize=(13,12),sharex=True,layout='constrained')
    for row,s in zip(axes,result['all_five_criteria']):
        seed=s['seed']; t=s['trigger_step']; tl=[r for r in timelines if int(r['seed'])==seed]
        raster,counts=row
        for spikes,y,color,label in [(s['b3_actual_spike_steps'],1,'#703b89','actual'),(s['b3_reference_spike_steps'],0,'#078090','motion-only reference')]:
            raster.vlines(np.array(spikes)/1000,y-.18,y+.18,color=color,lw=1.7,label=label)
        raster.axvspan((t-200)/1000,t/1000,color='#d18100',alpha=.13,label='original trigger window')
        raster.set_yticks([0,1],['reference','actual']); raster.set_ylim(-.5,1.5)
        raster.set_title(f'{seed}: left b3 exact spikes',fontsize=10)
        times=[int(r['end_step'])/1000 for r in tl]
        counts.step(times,[int(r['b3_rolling']) for r in tl],where='post',color='#703b89',label='200 ms rolling increment')
        counts.step(times,[int(r['b3_cumulative']) for r in tl],where='post',color='#078090',ls='--',label='cumulative increment')
        counts.axhline(0,color='gray',lw=.6); counts.set_title(f'trigger {t} ms: -1; final cumulative +{s["final_cumulative_b3"]}',fontsize=10)
        for ax in row:
            ax.axvline(t/1000,color='#c77700',ls=':',lw=1.4); ax.set_xlim(.5,1.5); ax.grid(alpha=.15)
        for end in s['deficit_expiration_steps']: counts.axvline(end/1000,color='#888888',lw=.7,ls='--')
    axes[0,0].legend(loc='upper left',fontsize=7); axes[0,1].legend(loc='upper left',fontsize=7)
    axes[-1,0].set_xlabel('Recorded seconds'); axes[-1,1].set_xlabel('Recorded seconds')
    fig.suptitle('Five deficit-only failures: exact recorded timing and window arithmetic\nOrange: original first selection; shaded interval is (trigger - 200 ms, trigger]. No replacement readout.',fontsize=12)
    fig.savefig(OUT/'five-deficit-timelines.png',dpi=160); plt.close(fig)
    ties=[s for s in summaries if s['seed'] in (1100010,1100017,1100032)]
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
    for ax,s in zip(axes,ties):
        tl=[r for r in timelines if int(r['seed'])==s['seed']]
        ax.step([int(r['end_step'])/1000 for r in tl],[float(r['delta']) for r in tl],where='post',color='#703b89')
        ax.axhline(0,color='gray',lw=.7); ax.axvline(s['trigger_step']/1000,color='#c77700',ls='--')
        ax.set_title(f'{s["seed"]}: original seeded tie selects {s["selected"]} at {s["trigger_step"]} ms; six-vector {s["trigger"]["six"]}',fontsize=10)
        ax.set_ylabel('Right - left score'); ax.set_xlim(.5,1.5); ax.grid(alpha=.15)
    axes[-1].set_xlabel('Recorded seconds; original selection retained')
    fig.suptitle('Three historical seeded ties: recorded score timelines\nLater asymmetry is measured; no new decoder or response outcome is evaluated.',fontsize=12)
    fig.savefig(OUT/'tie-timelines.png',dpi=160); plt.close(fig)
    print(json.dumps(dict(classification=result['classification'],pairs=9,five_deficits=evidence),indent=2))


if __name__=='__main__': main()
