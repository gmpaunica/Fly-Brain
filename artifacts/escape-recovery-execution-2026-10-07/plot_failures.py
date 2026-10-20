"""Saved score/streak timelines: show why the fixed candidate missed two cases."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ART=Path('artifacts/escape-recovery-execution-2026-10-07')
with (ART/'failure-timelines.csv').open(newline='') as f:rows=list(csv.DictReader(f))
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for axis,seed in zip(axes,(1100018,1100021)):
    selected=[r for r in rows if int(r['seed'])==seed];times=[float(r['seconds']) for r in selected]
    a,b=axis
    a.step(times,[int(r['left']) for r in selected],where='post',label='cumulative left',color='#703b89')
    a.step(times,[int(r['right']) for r in selected],where='post',label='cumulative right',color='#078090')
    b.step(times,[int(r['streak']) for r in selected],where='post',color='#703b89')
    a.set_title(f'{seed}: recorded cumulative scores');b.set_title('Recorded agreement streak')
    onset=1.3 if seed==1100018 else .86
    for ax in axis:ax.axvline(onset,color='#c77700',ls='--');ax.grid(alpha=.2);ax.set_xlabel('Recorded seconds')
    if seed==1100021:
        for ax in axis:ax.set_xlim(.8,.94);ax.axvline(.9,color='red',ls=':')
    b.axhline(2,color='gray',ls=':',label='two-sample requirement')
axes[0,0].legend();axes[0,1].legend()
fig.suptitle('Implemented fixed candidate: two regressions at 86.25 degrees\nOrange: first strong threat; red: wrong-side commitment. No alternate readout evaluated.',fontsize=12)
fig.savefig(ART/'failure-timelines.png',dpi=160);plt.close(fig)
