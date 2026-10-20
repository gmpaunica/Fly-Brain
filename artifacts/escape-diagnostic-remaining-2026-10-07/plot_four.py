"""Two completed deficit-case timelines; no extrapolation or new decoder."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('artifacts/escape-diagnostic-remaining-2026-10-07')
PRIOR=Path('artifacts/escape-diagnostic-recovery-2026-10-07')
r=json.loads((OUT/'failure-audit-final.json').read_text())
p=json.loads((PRIOR/'interruption-audit.json').read_text())
r['case_summaries']=p['case_summaries']+r['case_summaries']
r['partial_five_case_evidence']=p['partial_five_case_evidence']+r['partial_five_case_evidence']
with (OUT/'final-partial-timelines.csv').open(newline='',encoding='utf-8') as f: timeline=list(csv.DictReader(f))
with (PRIOR/'partial-timelines.csv').open(newline='',encoding='utf-8') as f: timeline+=list(csv.DictReader(f))
fig,axes=plt.subplots(4,2,figsize=(12,12),sharex=True,layout='constrained')
for row,e in zip(axes,r['partial_five_case_evidence']):
    s=next(v for v in r['case_summaries'] if v['seed']==e['seed'] and v.get('status')=='completed')
    raster,count=row; t=e['trigger_step']; tl=[v for v in timeline if int(v['seed'])==e['seed']]
    for name,y,color in [('actual',1,'#74388a'),('reference',0,'#008393')]:
        raster.vlines(np.array(s[name+'_b3_steps'])/1000,y-.2,y+.2,color=color,lw=2,label=name)
    raster.axvspan((t-200)/1000,t/1000,color='#d08000',alpha=.13)
    raster.set_yticks([0,1],['reference','actual']); raster.set_ylim(-.5,1.5)
    raster.set_title(f'{e["seed"]}: exact left b3 spikes',fontsize=11)
    times=[int(v['end_step'])/1000 for v in tl]
    count.step(times,[int(v['b3_rolling']) for v in tl],where='post',label='200 ms rolling increment',color='#74388a')
    count.step(times,[int(v['b3_cumulative']) for v in tl],where='post',label='cumulative increment',color='#008393',ls='--')
    count.axhline(0,color='gray',lw=.6); count.set_title(f'Original gate {t} ms: -1; final cumulative +{e["final_cumulative_b3"]}',fontsize=11)
    for ax in row:
        ax.axvline(t/1000,color='#d08000',ls=':'); ax.grid(alpha=.15); ax.set_xlim(.5,1.5)
        ax.set_xlabel('Recorded seconds')
axes[0,0].legend(fontsize=8); axes[0,1].legend(fontsize=8)
fig.suptitle('Partial evidence: four of five required deficit-only cases completed\nOriginal selections retained. Overall hypothesis unresolved; no replacement readout evaluated.',fontsize=12)
fig.savefig(OUT/'four-deficit-timelines.png',dpi=160); plt.close(fig)
print('Two complete 1.50-second recorded timelines exported; no simulation.')
