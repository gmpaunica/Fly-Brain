"""Independent saved-only arithmetic audit; no brain or renderer invocation."""
import argparse
from collections import deque
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np


def sha(path: Path) -> str:
    """Streaming saved-byte digest, independent of the runner."""
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def exact(a,b,label: str) -> None:
    """Fail closed on a missing/changed shape or numerical observation."""
    if not np.array_equal(a,b): raise ValueError(f'saved measurement mismatch: {label}')


def check_chunk(a: dict, graph: dict, first_step: int) -> dict:
    """Reconstruct original equations from saved tensors; no simulated steps."""
    pre,post,w=graph['pre'],graph['post'],graph['weights']
    inputs,live=graph['inputs'],graph['live']; effective=(~inputs[post])&live[post]
    steps=len(a['step_index']); n=len(inputs)
    exact(a['step_index'],np.arange(first_step,first_step+steps),'step IDs')
    if a['rgb'].shape!=(2,24,48,3): raise ValueError('missing RGB')
    results={}
    for b in ('actual','reference'):
        v=a[b+'_initial_voltage'].copy(); prev=a[b+'_initial_previous'].copy(); phase=a[b+'_initial_phase'].copy()
        rates=a[b+'_rates']; counts=np.zeros(n,dtype=np.int64); delivered=np.zeros(len(w))
        sums=np.zeros(n); peaks=np.zeros(n); discarded=np.zeros(n)
        pv=np.full(n,-np.inf); pp=np.zeros(n); first=np.full(n,-1)
        for k in range(steps):
            c=w*prev[pre]; exact(a[b+'_contribution'][k],c,b+' edge delivery')
            v=v*.9900498337491681; np.add.at(v,post,c)
            exact(a[b+'_pre_voltage'][k],v,b+' pre-reset voltage')
            delivered=delivered+c*effective
            phase=phase+rates*.001; exact(a[b+'_pre_phase'][k],phase,b+' pre phase')
            incoming=np.zeros(n); np.add.at(incoming,post,c*effective)
            lost=np.zeros(n); np.add.at(lost,post,c*(~effective))
            sums=sums+incoming; peaks=np.maximum(peaks,incoming); discarded=discarded+lost
            pv=np.maximum(pv,np.where(inputs|~live,0.,v)); pp=np.maximum(pp,np.where(inputs&live,phase,0.))
            forced=phase>=1.; phase=np.where(forced,phase-1.,phase)
            spikes=np.where(inputs,forced,v>=1.)&live
            exact(a[b+'_spikes'][k],spikes,b+' spikes')
            v=np.where(inputs|~live,0.,v-spikes.astype(np.float64))
            exact(a[b+'_post_voltage'][k],v,b+' post-reset voltage')
            exact(a[b+'_post_phase'][k],phase,b+' post phase')
            counts+=spikes.astype(np.int64); first=np.where(spikes&(first<0),k+1,first)
            prev=spikes.astype(np.float64)
        results[b]=dict(counts=counts,delivered=delivered,current_sum=sums,current_peak=peaks,
                        discarded_current=discarded,peak_voltage=pv,peak_phase=pp,first=first)
    if any(not np.isfinite(v).all() for v in a.values()): raise ValueError('nonfinite saved chunk')
    return results


def csv_write(path: Path, rows: list[dict]) -> None:
    """Fresh reproducible table; JSON-encode arrays rather than lossy display."""
    with path.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def audit(directory: Path, output: Path | None=None) -> dict:
    """Validate seals, every cell/edge/step, continuity, historical parity/windows."""
    receipt=json.loads((directory/'receipt.json').read_text()); config=json.loads((directory/'config.json').read_text())
    seal=json.loads((directory.parent/'escape-diagnostic-seal.json').read_text())
    if seal['receipt_sha256']!=sha(directory/'receipt.json'): raise ValueError('receipt seal mismatch')
    for name,h in receipt['outputs'].items():
        if sha(directory/name)!=h: raise ValueError(f'output hash mismatch: {name}')
    expected_files=set(receipt['outputs'])|{'receipt.json'}
    if {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}!=expected_files:
        raise ValueError('unbound output file')
    if receipt['status']!='completed': return dict(status='closed-incomplete',receipt=receipt,audit_kernel_steps=0)
    if (receipt['completed_cases'],receipt['recorded_intervals'],receipt['reserved_kernel_steps'])!=(18,1350,54000):
        raise ValueError('budget mismatch')
    for name,h in {**dict(config['response']['source_sha256']),**config['added_sources'],**config['test_sources']}.items():
        if sha(directory/'sources'/name)!=h: raise ValueError('archived source mismatch')
    manifest=json.loads((directory/'circuit.json').read_text())
    if sha(directory/'circuit.json')!=config['response']['circuit_sha256']: raise ValueError('circuit archive binding mismatch')
    graph=dict(np.load(directory/'graph.npz',allow_pickle=False))
    nodes=manifest['nodes']; ids=[n['id'] for n in nodes]; index={s:i for i,s in enumerate(ids)}
    triples=[e for e in manifest['edges'] if e['pre']!=e['post']]
    exact(graph['pre'],[index[e['pre']] for e in triples],'all graph sources')
    exact(graph['post'],[index[e['post']] for e in triples],'all graph destinations')
    totals=np.zeros(589)
    for e in triples: totals[index[e['post']]]+=e['count']
    inc=set(map(tuple,config['response']['incoming_edges'])); out=set(map(tuple,config['response']['outgoing_edges']))
    weights=[]
    for e in triples:
        w=.9*e['count']/totals[index[e['post']]]
        if (e['pre'],e['post']) in inc: w*=4
        if (e['pre'],e['post']) in out: w*=2
        weights.append(w)
    exact(graph['weights'],weights,'frozen effective weights')
    exact(graph['effective'],(~graph['inputs'][graph['post']])&graph['live'][graph['post']],'edge effective status')
    historical={c['seed']:[] for c in config['cases']}
    with gzip.open('experiments/turnflee-2026-10-06/traces.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['phase']=='validation' and r['candidate']==1 and r['seed'] in historical: historical[r['seed']].append(r)
    six=sorted([i for i,n in enumerate(nodes) if n['metadata'].get('cell_type') in ('i1','b3','b1')],
               key=lambda i: (nodes[i]['metadata']['side']=='right',('i1','b3','b1').index(nodes[i]['metadata']['cell_type'])))
    b3=six[1]; dn=[i for i,n in enumerate(nodes) if n['metadata'].get('cell_type') in ('DNp03','DNp26') and n['metadata'].get('side')=='right']
    dn_edges=np.flatnonzero(np.isin(graph['pre'],dn)&(graph['post']==b3)&graph['effective'])
    summaries=[]; timelines=[]; events=[]; intervals_total=0; all_spikes=0
    for ordinal,case in enumerate(config['cases']):
        case_dir=directory/f'{ordinal:02d}-{case["seed"]}'
        with gzip.open(case_dir/'intervals.jsonl.gz','rt',encoding='utf-8') as f: records=[json.loads(line) for line in f]
        if len(records)!=75: raise ValueError('missing intervals')
        window=deque(maxlen=10); count_window=deque(maxlen=10); cumulative=np.zeros(589,dtype=np.int64)
        first={b:[None]*589 for b in ('actual','reference')}; previous={}; history={b:np.zeros((2,589)) for b in first}
        sizes=deque([0.]*5,maxlen=5); growths=deque([0.]*5,maxlen=5)
        last_raw=0.; absent=5; appearance=False; weak=0; previous_salience=None
        selected=None; trigger=None; strong=None; pending=-1; max_hz=0.; rng=random.Random(case['seed'])
        spikes_b3={b:[] for b in first}; deliveries=[]; earliest_diff=None; trigger_state=None
        for k,r in enumerate(records):
            old=r['original']
            if old!=historical[case['seed']][k]: raise ValueError('historical interval mismatch')
            p=case_dir/f'{k:03d}.npz'
            if sha(p)!=r['chunk_sha256']: raise ValueError('chunk receipt mismatch')
            a=dict(np.load(p,allow_pickle=False)); measured=check_chunk(a,graph,k*20+1)
            if hashlib.sha256(a['rgb'].tobytes()).hexdigest()!=old['eyes_sha256']: raise ValueError('RGB hash mismatch')
            contrast=np.minimum(a['rgb'][...,1],a['rgb'][...,2])-a['rgb'][...,0]
            salience=np.where(contrast>.05,contrast,0.)
            growth=np.zeros_like(salience) if previous_salience is None else np.maximum(salience-previous_salience,0.)
            exact(a['salience'],salience,'salience'); exact(a['growth'],growth,'growth'); previous_salience=salience
            rates=np.zeros(589); patch=np.zeros((589,3))
            for side,eye in (('left',0),('right',1)):
                visual=[i for i,n in enumerate(nodes) if 'vision' in n['roles'] and n['metadata'].get('side')==side]
                for rank,i in enumerate(visual):
                    row=(rank%6)*4; col=(rank//6%8)*6
                    v=salience[eye,row:row+4,col:col+6]; g=growth[eye,row:row+4,col:col+6]
                    patch[i]=[float(v.max()),float(v.mean()),float(g.max())]
                    rates[i]=240*min(1.,.75*float(v.max())+.25*float(v.mean())+.5*float(g.max()))
            motion=np.zeros(589)
            for side in ('left','right'):
                group=[i for i,n in enumerate(nodes) if 'motion' in n['roles'] and n['metadata'].get('side')==side]
                for rank,i in enumerate(group):
                    signal=(0.,0.,0.,0.,0.,1.)[(rank//2)%6]*(1 if rank%2 else -1)
                    motion[i]=120*(1+math.tanh(signal))
            exact(a['patch'],patch,'LPLC4 patch statistics')
            exact(a['actual_rates'],rates+motion,'unchanged actual rates'); exact(a['reference_rates'],motion,'motion-only rates')
            fraction=np.clip(salience/.9,0,1)
            elevation=np.linspace(math.radians(80),math.radians(-80),24)
            area=(math.pi/48)*math.radians(160)/23*np.cos(elevation)[None,:,None]
            solid=float(np.sum(fraction*area)); raw=math.degrees(2*math.acos(float(np.clip(1-solid/(2*math.pi),-1,1))))
            if raw==0:
                absent+=1
                if absent>=5: appearance=False
            else:
                if absent>=5: appearance=raw>=20
                absent=0
            sizes.append(raw); growths.append(max(0.,(raw-last_raw)/.02)); last_raw=raw
            size=float(np.mean(sizes)); gr=float(np.mean(growths)); isstrong=size>=20 and (appearance or gr>=10)
            if isstrong: appearance=False
            weak=weak+1 if size<=10 and gr<=5 else 0
            if old['threat']!=dict(size_degrees=size,positive_growth_degrees_s=gr,strong=isstrong,weak_continuous=weak>=15):
                raise ValueError('threat reconstruction mismatch')
            for b,diag in [('actual',old['pathway']),('reference',r['reference_pathway'])]:
                m=measured[b]; exact(m['counts'],old[b+'_counts'],b+' counts')
                for key in ('current_sum','current_peak','discarded_current','peak_voltage','peak_phase'):
                    exact(m[key],diag[key],b+' interval '+key)
                times=[(k*20+int(t))*.001 if t>=0 else None for t in m['first']]
                if times!=diag['first_spike_seconds']: raise ValueError('first spike time mismatch')
                for i,t in enumerate(times):
                    if first[b][i] is None: first[b][i]=t
                if first[b]!=diag['episode_first_spike_seconds']: raise ValueError('episode spike time mismatch')
                if [round(t*1000) if t is not None else None for t in first[b]]!=r['first_spike_step_'+b]: raise ValueError('integer spike time mismatch')
                if k:
                    for state in ('voltage','previous','phase'): exact(a[b+'_initial_'+state],previous[b][state],b+' continuity '+state)
                else:
                    for state in ('voltage','previous','phase'): exact(a[b+'_initial_'+state],np.zeros(589),b+' reset '+state)
                previous[b]=dict(voltage=a[b+'_post_voltage'][-1],previous=a[b+'_spikes'][-1].astype(np.float64),phase=a[b+'_post_phase'][-1])
                for j,tau in enumerate((.1,.5)):
                    alpha=-math.expm1(-.02/tau); history[b][j]=(1-alpha)*history[b][j]+alpha*m['counts'].astype(float)/.02
                exact(a[b+'_history'],history[b],b+' rate history')
                steps=a['step_index'][a[b+'_spikes'][:,b3]].tolist(); spikes_b3[b].extend(steps)
                all_spikes+=int(a[b+'_spikes'].sum())
            if earliest_diff is None:
                diff=np.flatnonzero(a['actual_spikes'][:,b3]!=a['reference_spikes'][:,b3])
                if len(diff): earliest_diff=int(a['step_index'][diff[0]])
            for t in np.flatnonzero(np.any(a['actual_contribution'][:,dn_edges]!=a['reference_contribution'][:,dn_edges],axis=1)):
                deliveries.append(int(a['step_index'][t]))
                events.append(dict(seed=case['seed'],kind='right-DN to b3 delivery difference',step=int(a['step_index'][t]),
                    actual=float(a['actual_contribution'][t,dn_edges].sum()),reference=float(a['reference_contribution'][t,dn_edges].sum())))
            inc=measured['actual']['counts']-measured['reference']['counts']; exp=window[0] if len(window)==10 else np.zeros(589,dtype=np.int64)
            window.append(inc); cumulative+=inc; roll=np.sum(window,axis=0)
            exact(r['entered_increment'],inc,'window entry'); exact(r['expired_increment'],exp,'window expiry')
            if r['expired_interval']!=(k-10 if k>=10 else None): raise ValueError('expiry ID mismatch')
            exact(r['rolling_all'],roll,'rolling counts'); exact(r['cumulative_increment'],cumulative,'cumulative counts')
            exact(old['wing_vector'],roll[graph['wing_indices']],'wing vector')
            v=roll[six]; left=float(v[0]+v[1]+v[5]); right=float(v[3]+v[4]+v[2]); norm=math.sqrt(float(np.sum(v*v)))
            bilateral=bool(np.any(v[:3]) and np.any(v[3:])); delta=right-left
            if old['wing_scores']!=dict(left=left,right=right,strength=norm,bilateral=bilateral): raise ValueError('score mismatch')
            if isstrong:
                pending=k+1+25
                if strong is None: strong=(k+1)*20
            if selected is None and k+1<=pending and norm>=1 and (delta!=0 or bilateral):
                selected=('right' if delta>0 else 'left') if delta else rng.choice(('left','right'))
                trigger=(k+1)*20; pending=-1
                trigger_state=dict(interval=k,b3_rolling=int(v[1]),b3_entered=int(inc[b3]),b3_expired=int(exp[b3]),six=v.tolist())
            if old['selected_side']!=selected or old['triggers']!=int(selected is not None): raise ValueError('selection mismatch')
            count_window.append(np.stack((measured['actual']['counts'],measured['reference']['counts'])))
            if len(count_window)==10: max_hz=max(max_hz,float(np.max(np.sum(count_window,axis=0)[:,~graph['inputs']]))/.2)
            if old['max_downstream_hz']!=max_hz or max_hz>240: raise ValueError('rolling rate mismatch')
            timelines.append(dict(seed=case['seed'],end_step=(k+1)*20,actual_b3_spikes=a['step_index'][a['actual_spikes'][:,b3]].tolist(),
                reference_b3_spikes=a['step_index'][a['reference_spikes'][:,b3]].tolist(),
                b3_entered=int(inc[b3]),b3_expired=int(exp[b3]),b3_rolling=int(roll[b3]),b3_cumulative=int(cumulative[b3]),
                six=v.tolist(),left=left,right=right,delta=delta,strong=isstrong,selected=selected))
            intervals_total+=1
        for b in first:
            for step in spikes_b3[b]: events.append(dict(seed=case['seed'],kind=b+' b3 spike',step=step,actual=int(b=='actual'),reference=int(b=='reference')))
        before_a=[s for s in spikes_b3['actual'] if s<=trigger]; before_r=[s for s in spikes_b3['reference'] if s<=trigger]
        diff_before=sorted(set(before_a)^set(before_r))
        summaries.append(dict(**case,comparison_label='failure' if ordinal%2==0 else 'successful context',
            first_strong_step=strong,trigger_step=trigger,selected=selected,trigger=trigger_state,
            final_cumulative_b3=int(cumulative[b3]),first_b3_timing_difference_step=earliest_diff,
            first_differential_right_DN_delivery_step=min(deliveries) if deliveries else None,
            differential_DN_delivery_precedes_b3=bool(deliveries and earliest_diff and min(deliveries)<earliest_diff),
            b3_actual_spike_steps=spikes_b3['actual'],b3_reference_spike_steps=spikes_b3['reference'],
            b3_redistribution_before_trigger=bool(set(before_a)-set(before_r) and set(before_r)-set(before_a)),
            b3_timing_differences_before_trigger=diff_before,max_downstream_hz=max_hz))
        print(f'offline audit {ordinal+1}/18 seed {case["seed"]}: all cells/edges/steps verified',flush=True)
    deficits={1100000,1100001,1100012,1100013,1100036}
    evidence=[s for s in summaries if s['seed'] in deficits]
    # Expiration can create the deficit earlier than the gate; check the exact
    # transition to -1 and its expired +1 interval, not merely the gate tick.
    for s in evidence:
        t=s['trigger_step']; rows=[r for r in timelines if r['seed']==s['seed'] and r['end_step']<=t]
        transitions=[r for j,r in enumerate(rows) if r['b3_rolling']==-1 and (j==0 or rows[j-1]['b3_rolling']>=0)
                     and r['b3_expired']>0 and r['b3_entered']==0]
        s['deficit_expiration_steps']=[r['end_step'] for r in transitions]
        s['support_all_four']=bool(s['trigger']['b3_rolling']==-1 and s['b3_redistribution_before_trigger']
            and transitions and s['final_cumulative_b3']>0 and s['differential_DN_delivery_precedes_b3'])
    classification='supported' if len(evidence)==5 and all(s['support_all_four'] for s in evidence) else 'unsupported'
    result=dict(status='audit-passed',classification=classification,cases=18,intervals=intervals_total,
                recorded_kernel_steps=54000,audit_kernel_steps=0,total_recorded_spikes=all_spikes,
                all_five_criteria=evidence,case_summaries=summaries,
                limitation='Temporal edge delivery supports an observed route, not causal necessity; comparisons differ in elevation/profile.')
    if output:
        output.mkdir(parents=True,exist_ok=False)
        (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        csv_write(output/'cases.csv',summaries); csv_write(output/'timelines.csv',timelines)
        csv_write(output/'events.csv',sorted(events,key=lambda e:(e['seed'],e['step'],e['kind'])))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('directory',type=Path); p.add_argument('--output',type=Path); args=p.parse_args()
    r=audit(args.directory,args.output)
    print(json.dumps({k:r[k] for k in ('status','classification','cases','intervals','recorded_kernel_steps','audit_kernel_steps') if k in r},indent=2))
