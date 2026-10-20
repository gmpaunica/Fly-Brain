"""Offline saved-flight plots; plotting never advances world or neural state."""
import gzip
import json
from pathlib import Path
import numpy as np


def render(directory:Path,output:Path) -> list[Path]:
    """Plot measured paths plus yaw/height/separation of representative cases."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups={}
    with gzip.open(directory/'traces.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);groups.setdefault((r['azimuth'],r['state'],r['intervention']),[]).append(r)
    result=json.loads((directory/'results.json').read_text());output.mkdir(parents=True,exist_ok=True);paths=[]
    for state in ('resting','hovering','cruising'):
        fig=plt.figure(figsize=(18,9),layout='constrained')
        for i,azimuth in enumerate(range(0,360,45)):
            ax=fig.add_subplot(2,4,i+1,projection='3d')
            for condition,color in (('intact','#177db0'),('visual-silent','#dd7a22')):
                records=groups[(azimuth,state,condition)];positions=np.array([r['position'] for r in records])*1000
                ax.plot(*positions.T,color=color,label=condition,lw=1.5)
                ax.scatter(*positions[0],color=color,marker='o',s=14);ax.scatter(*positions[-1],color=color,marker='x',s=25)
            predator=np.array(records[0]['predator_position'])*1000;ax.scatter(*predator,color='crimson',marker='^',s=45)
            row=next(r for r in result['closed_loop'] if r['azimuth']==azimuth and r['state']==state and r['intervention']=='intact')
            ax.set_title(f'{azimuth}° | intact {"success" if row["success"] else "failure"}',fontsize=11)
            ax.set(xlabel='X mm',ylabel='Y mm',zlabel='height mm',zlim=(0,150))
            if i==0:ax.legend(fontsize=8)
        fig.suptitle(f'Frozen controller: {state} starts — saved six-second paths\n'+result['status'],fontsize=15)
        path=output/f'flight-{state}.png';fig.savefig(path,dpi=140);plt.close(fig);paths.append(path)
    fig,axes=plt.subplots(4,3,figsize=(15,11),sharex='col',layout='constrained')
    for j,state in enumerate(('resting','hovering','cruising')):
        failed=next((r for r in result['closed_loop'] if r['state']==state and r['intervention']=='intact' and not r['success']),None)
        row=failed or next(r for r in result['closed_loop'] if r['state']==state and r['intervention']=='intact')
        records=groups[(row['azimuth'],state,'intact')];t=np.array([r['end_seconds'] for r in records])
        pos=np.array([r['position'] for r in records]);velocity=np.array([r['velocity'] for r in records])
        anchor=row['pre_command_yaw'];yaw=np.array([r['orientation'][2] for r in records])
        axes[0,j].plot(t,pos[:,2]*1000);axes[0,j].axhspan(75,125,color='green',alpha=.1)
        axes[0,j].set_title(f'{state}, {row["azimuth"]}°, side {row["side"] or "none"}')
        axes[1,j].plot(t,np.degrees(yaw-(anchor or 0.)));axes[1,j].axhline(90 if row['side']=='right' else -90,color='grey',ls=':')
        axes[2,j].plot(t,np.linalg.norm(velocity[:,:2],axis=1)*1000,label='horizontal')
        axes[2,j].plot(t,np.abs(velocity[:,2])*1000,label='vertical');axes[2,j].axhline(20,color='grey',ls=':');axes[2,j].legend(fontsize=8)
        separation=np.linalg.norm(pos-np.array([r['predator_position'] for r in records]),axis=1)
        axes[3,j].plot(t,(separation-(row['initial_predator_separation_m'] or separation[0]))*1000)
        axes[3,j].axhline(20,color='grey',ls=':');axes[3,j].set_xlabel('seconds')
        for ax in axes[:,j]:
            for key,color in (('trigger_seconds','purple'),('first_flee_seconds','blue'),('weak_seconds','orange'),('braking_seconds','green')):
                if row[key] is not None:ax.axvline(row[key],color=color,alpha=.5,lw=.8)
            ax.grid(alpha=.2)
    for ax,label in zip(axes[:,0],('Height (mm)','Actual yaw change (degrees)','Speed (mm/s)','Separation gain (mm)')):ax.set_ylabel(label)
    fig.suptitle('Saved physical measurements — purple trigger, blue flee, orange weak, green brake\nFirst failed intact case in each state, when available',fontsize=13)
    path=output/'physical-timelines.png';fig.savefig(path,dpi=140);plt.close(fig);paths.append(path)
    return paths
