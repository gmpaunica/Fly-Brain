"""Saved-only final figure layout; legends outside recorded spike rasters."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('artifacts/escape-diagnostic-final-five-2026-10-07')
r=json.loads((OUT/'analysis/audit.json').read_text())
with (OUT/'analysis/timelines.csv').open(newline='',encoding='utf-8') as f: timeline=list(csv.DictReader(f))
fig,axes=plt.subplots(5,2,figsize=(13,12),sharex=True,layout='constrained')
for row,s in zip(axes,r['all_five_criteria']):
    raster,counts=row;t=s['trigger_step'];tl=[v for v in timeline if int(v['seed'])==s['seed']]
    for label,y,color in [('actual',1,'#703b89'),('reference',0,'#078090')]:
        raster.vlines(np.array(s['b3_'+label+'_spike_steps'])/1000,y-.18,y+.18,color=color,lw=1.7,label=label)
    raster.axvspan((t-200)/1000,t/1000,color='#d18100',alpha=.13,label='original trigger window')
    raster.set_yticks([0,1],['reference','actual']);raster.set_ylim(-.5,1.5)
    raster.set_title(f'{s["seed"]}: left b3 exact spikes',fontsize=10)
    times=[int(v['end_step'])/1000 for v in tl]
    counts.step(times,[int(v['b3_rolling']) for v in tl],where='post',color='#703b89',label='200 ms rolling increment')
    counts.step(times,[int(v['b3_cumulative']) for v in tl],where='post',color='#078090',ls='--',label='cumulative increment')
    counts.axhline(0,color='gray',lw=.6)
    counts.set_title(f'Original trigger {t} ms: -1; final cumulative +{s["final_cumulative_b3"]}',fontsize=10)
    for ax in row:
        ax.axvline(t/1000,color='#c77700',ls=':',lw=1.4);ax.set_xlim(.5,1.5);ax.grid(alpha=.15)
    for end in s['deficit_expiration_steps']:counts.axvline(end/1000,color='#888888',lw=.7,ls='--')
axes[-1,0].set_xlabel('Recorded seconds');axes[-1,1].set_xlabel('Recorded seconds')
h1,l1=axes[0,0].get_legend_handles_labels();h2,l2=axes[0,1].get_legend_handles_labels()
fig.legend(h1+h2,l1+l2,loc='outside lower center',ncol=5,fontsize=8)
fig.suptitle('Five deficit-only failures: exact recorded timing and window arithmetic\nOrange: original selection. Shading: (trigger - 200 ms, trigger]. No replacement readout evaluated.',fontsize=12)
fig.savefig(OUT/'five-deficit-timelines-final.png',dpi=160);plt.close(fig)
print('Five saved timelines rendered with external legends; zero simulation.')
