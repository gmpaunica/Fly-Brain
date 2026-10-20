"""Saved trajectories and physical hover bands; never advance a simulation."""
import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flybrain.escape_edge.validation import ART, BASE
from flybrain.flight3d.anatomy import digest
from flybrain.escape_diagnostic_recovery.runner import write

fig,axs = plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
colors = dict(idle='gray',turning='orange',fleeing='blue',edge_braking='purple',edge_hover='green')
for ax,index in zip(axs.flat,(0,4,12,26,28,36)):
    with gzip.open(BASE/'development/traces'/f'{index:03}.jsonl.gz','rt') as stream:
        rows = [json.loads(line) for line in stream]
    for state,color in colors.items():
        points = np.array([r['position'][:2] for r in rows if r['escape_state']==state])
        if len(points):
            ax.scatter(points[:,0],points[:,1],s=5,color=color,label=state.replace('_',' '))
    predator = rows[0]['predator_position']
    ax.scatter(predator[0],predator[1],marker='x',color='red',s=40,label='predator')
    ax.plot([.015,.585,.585,.015,.015],[.015,.015,.585,.585,.015],'k--',lw=.8)
    ax.set(xlim=(0,.6),ylim=(0,.6),aspect='equal',xlabel='X (m)',ylabel='Y (m)',
           title=f'{rows[0]["azimuth"]} deg / {rows[0]["state"]}')
axs.flat[0].legend(fontsize=7,loc='upper right')
fig.suptitle('Measured sustained edge escape — historical development cases')
fig.savefig(ART/'trajectories.png',dpi=150)
plt.close(fig)

fig,axs = plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
for phase in ('development','robustness'):
    for index in range(0,48,2):
        with gzip.open(BASE/phase/'traces'/f'{index:03}.jsonl.gz','rt') as stream:
            rows = [json.loads(line) for line in stream]
        pos = np.array([r['position'] for r in rows])
        velocity = np.array([r['velocity'] for r in rows])
        times = [r['end_seconds'] for r in rows]
        axs[0].plot(times,1000*np.minimum(pos[:,:2],.6-pos[:,:2]).min(axis=1),alpha=.3,lw=.7)
        axs[1].plot(times,np.linalg.norm(velocity[:,:2],axis=1),alpha=.3,lw=.7)
        axs[2].plot(times,pos[:,2],alpha=.3,lw=.7)
axs[0].axhspan(5,25,color='green',alpha=.15)
axs[0].set(ylabel='Nearest wall (mm)')
axs[1].axhspan(0,.01,color='green',alpha=.15)
axs[1].set(ylabel='Horizontal speed (m/s)')
axs[2].axhspan(.085,.115,color='green',alpha=.15)
axs[2].set(ylabel='Altitude (m)',xlabel='Simulated time (s)')
for ax in axs:
    ax.axvline(10,color='black',ls='--',lw=.8)
    ax.grid(alpha=.2)
fig.suptitle('All 48 intact cases — arrival by 10 s, stable edge hover through 12 s')
fig.savefig(ART/'hover.png',dpi=150)
plt.close(fig)
write(ART/'plots.json',dict(status='generated',neural_steps_added=0,
    inputs={phase:digest(BASE/phase/'receipt.json') for phase in ('development','robustness')},
    outputs={n:digest(ART/n) for n in ('trajectories.png','hover.png')}))
print('Saved-only trajectory and hover figures generated.')
