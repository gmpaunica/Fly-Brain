"""Audit recorded FB-10 results and plot them without simulation or optimization."""

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import torch


ROOT=Path(__file__).resolve().parents[1]


def audit(output: Path) -> dict:
    """Recompute checkpoint errors/gates, CSV agreement and preserved file hashes."""
    torch.set_num_threads(1)
    from flybrain.training.banc_readiness import save_new_json,verify_provenance
    from flybrain.training.banc_training import training_gates
    from flybrain.training.banc_coverage import coverage_gates
    from flybrain.training.evaluation import EpisodeResult,summarize
    from flybrain.training import banc_protocol,banc_conditioning,banc_memory,banc_coverage,banc_capacity,banc_projection_fit
    from flybrain.neural import banc_readout,banc_conditioning as conditioned,banc_memory as memory,banc_capacity as capacity,banc_projection as projection
    configs=[('Original',banc_protocol.DEFAULT_OUTPUT,banc_protocol,banc_readout),
        ('Conditioned',banc_conditioning.DEFAULT_OUTPUT,banc_conditioning,conditioned),
        ('Fast+slow',banc_memory.DEFAULT_OUTPUT,banc_memory,memory),
        ('Covered',banc_coverage.DEFAULT_OUTPUT,banc_coverage,memory),
        ('64 hidden',banc_capacity.DEFAULT_OUTPUT,banc_capacity,capacity),
        ('PN inputs',banc_projection_fit.DEFAULT_OUTPUT,banc_projection_fit,projection)]
    if output.exists():raise FileExistsError('audit output already exists')
    output.mkdir(parents=True);fits=[];checks={};original_data=banc_memory.load_training_data()
    for label,directory,runner,neural in configs:
        directory=(ROOT/directory).resolve()
        loader=(runner.load_training_protocol if runner in (banc_coverage,banc_projection_fit) else runner.load_protocol)
        p,digest,data=loader(directory/'protocol.json');r=json.loads((directory/'results.json').read_text(encoding='utf-8'))
        mean=torch.tensor(p['normalization']['mean']);scale=torch.tensor(p['normalization']['scale'])
        models={name:neural.load_checkpoint(directory/(name+'.pt'),protocol_sha256=digest,mean=mean,scale=scale) for name in ('trained','untrained')}
        with torch.no_grad():error=(models['trained'](data.features).double()-data.targets).square()
        mse=error.mean(0).tolist()
        if runner in (banc_coverage,banc_capacity):
            with torch.no_grad():original_mse=(models['trained'](original_data.features).double()-original_data.targets).square().mean(0).tolist()
        elif runner is banc_projection_fit:original_mse=error[:1008].mean(0).tolist()
        else:original_mse=mse
        changed=sum(int(torch.count_nonzero(v-models['untrained'].state_dict()[key])) for key,v in models['trained'].named_parameters())
        summary={name:summarize([EpisodeResult(**row) for row in rows]) for name,rows in r['episodes'].items()}
        original=runner in (banc_protocol,banc_conditioning,banc_memory)
        gates=(training_gates(mse[1],p['train_only_baseline']['constant_mse'][1],summary,r['checks']) if original
               else coverage_gates(mse[1],original_mse[1],p['train_only_baseline']['constant_mse'][1],summary,r['checks']))
        with (directory/'curve.csv').open(newline='',encoding='utf-8') as stream:curve=list(csv.DictReader(stream))
        csv_curves=len(curve)==len(r['curve'])==501 and all(all(a[k]==('' if v is None else str(v)) for k,v in b.items()) for a,b in zip(curve,r['curve']))
        expected=[dict(condition=name,**row) for name,rows in r['episodes'].items() for row in rows]+[dict(condition='trained_replay',**r['replay_episode'])]
        with (directory/'episodes.csv').open(newline='',encoding='utf-8') as stream:episodes=list(csv.DictReader(stream))
        csv_episodes=len(episodes)==len(expected)==25 and all(all(a[k]==('' if v is None else str(v)) for k,v in b.items()) for a,b in zip(episodes,expected))
        local={'exact_checkpoint_mse':mse==r['per_output_mse'],'exact_original_mse':original or original_mse==r['original_per_output_mse'],
            'parameter_change_count_exact':changed==r['parameter_changed_scalars'],'unchanged_normalization':torch.equal(models['trained'].mean,models['untrained'].mean) and torch.equal(models['trained'].scale,models['untrained'].scale),
            'summaries_exact':summary==r['summaries'],'gates_exact':gates==r['gates'],'curve_csv_exact':csv_curves,
            'episodes_csv_exact':csv_episodes,'500_updates':r['optimizer_steps']==500,'actual_steps_match':r['actual_neural_steps']==20*r['actual_world_ticks'],
            'provenance_preserved':verify_provenance(p['provenance']),'checkpoint_sha256_exact':hashlib.sha256((directory/'trained.pt').read_bytes()).hexdigest()==r['checkpoint_sha256']}
        checks[label]=local
        fits.append({'label':label,'directory':str(directory.relative_to(ROOT)),'per_output_mse':mse,
            'original_per_output_mse':original_mse,'training_gate_passed':r['training_gate_passed'],
            'arrivals':{name:s['successes'] for name,s in summary.items()},'optimizer_steps':r['optimizer_steps'],
            'world_ticks':r['actual_world_ticks'],'neural_steps':r['actual_neural_steps'],'changed_parameters':changed,'curve':r['curve']})
    snapshot=json.loads((ROOT/'.fb10-recovery-session-before.json').read_text(encoding='utf-8'))
    preservation={name:Path(ROOT/name).is_file() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in snapshot.items()}
    fresh=json.loads((ROOT/'experiments/fb10-banc-evaluation/results.json').read_text(encoding='utf-8'))
    fresh_rows={n:[EpisodeResult(**r) for r in group] for n,group in fresh['results']['validation'].items()}
    from flybrain.training.banc_memory_evaluation import assess_split
    fresh_summaries={n:summarize(group) for n,group in fresh_rows.items()}
    fresh_gates=assess_split('validation',{n:[r.outcome=='success' for r in group] for n,group in fresh_rows.items()},fresh_summaries)
    checks['first_fresh_assay']={'summaries_exact':fresh_summaries==fresh['summaries']['validation'],
        'gates_exact':fresh_gates==fresh['gates']['validation'],'final_not_stepped':fresh['final_cases_evaluated'] is False,
        'zero_updates':fresh['optimizer_steps']==0,'steps_match':fresh['actual_neural_steps']==20*fresh['actual_world_ticks']}
    r={'experiment':'fb10_recovery_offline_audit','created_utc':datetime.now(timezone.utc).isoformat(),
        'actual_world_ticks':0,'actual_neural_steps':0,'optimizer_steps':0,'fits':fits,'checks':checks,
        'original_files_preserved':all(preservation.values()),'original_files_checked':len(preservation),
        'original_sources_checked':sum(name.startswith('src') for name in snapshot),'original_experiments_checked':sum(name.startswith('experiments') for name in snapshot),
        'first_fresh_arrivals':fresh_gates['successes'],'first_fresh_gate_passed':fresh_gates['passed'],
        'scientific_failures_preserved':True,'verified':all(preservation.values()) and all(v for group in checks.values() for v in group.values())}
    save_new_json(output/'verification.json',r)
    plot(fits,fresh_gates,output)
    return r


def plot(fits,fresh,output):
    """Static scientific figures from stored numbers; never rerun a case."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    colors=['#64748b','#8b5cf6','#059669','#ea580c','#b45309','#0284c7']
    for row,color in zip(fits,colors):
        axis=axes[0,0] if row['label'] in ('Original','Conditioned','Fast+slow') else axes[0,1]
        axis.plot([r['update'] for r in row['curve']],[r['turn_mse'] for r in row['curve']],label=row['label'],color=color)
    for axis,title in zip(axes[0],('Original 1,008 training rows','Covered 15,864 training rows')):
        axis.axhline(.02225273146561865,color='#be123c',ls='--',label='Absolute gate 0.02225')
        axis.set(title=title,xlabel='Optimizer update',ylabel='Turn MSE',ylim=(0,.12));axis.legend(fontsize=9);axis.grid(alpha=.2)
    locations=np.arange(len(fits));axis=axes[1,0]
    for offset,name,color in ((-.24,'trained','#0284c7'),(0,'untrained','#94a3b8'),(.24,'fixed','#f59e0b')):
        values=[row['arrivals'][name] for row in fits];axis.bar(locations+offset,values,.23,label=name,color=color)
    axis.axhline(7,color='#be123c',ls='--');axis.set(title='Historical training arrivals (8 cases)',ylabel='Arrivals',ylim=(0,9))
    axis.set_xticks(locations,[r['label'] for r in fits],rotation=20);axis.legend(fontsize=9)
    axis=axes[1,1];labels=['trained','untrained','fixed'];values=[fresh['successes'][n] for n in labels]
    bars=axis.bar(labels,values,color=['#0284c7','#94a3b8','#f59e0b']);axis.bar_label(bars,labels=[f'{n}/16' for n in values],padding=4)
    axis.axhline(13,color='#be123c',ls='--',label='Validation minimum 13/16');axis.set(title='First fresh assay: closed at validation',ylabel='Arrivals',ylim=(0,17));axis.legend(fontsize=9)
    fig.suptitle('FB-10 recovery evidence: failures retained; circuit weights and neurons fixed',fontsize=14)
    fig.savefig(output/'recovery_summary.png',dpi=160);fig.savefig(output/'recovery_summary.svg');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'experiments/fb10-recovery-review');a=parser.parse_args()
    r=audit(a.output.resolve());print(json.dumps({k:r[k] for k in ('verified','original_files_checked','actual_world_ticks','optimizer_steps')},indent=2))
    raise SystemExit(0 if r['verified'] else 1)
