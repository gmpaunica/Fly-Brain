"""Verify all partial tensors/records and predeclared timing evidence offline."""
from collections import Counter,deque
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zlib
import numpy as np

ROOT=Path('experiments/escape-diagnostic-remaining-2026-10-07')
OUT=Path('artifacts/escape-diagnostic-remaining-2026-10-07')
calls=Counter()


def guard(frame,event,arg):
    if event!='call': return
    name=frame.f_code.co_filename.replace('\\','/')
    if '/src/flybrain/' not in name: return
    key=name.split('/src/flybrain/')[1]+':'+frame.f_code.co_name; calls[key]+=1
    if not name.endswith(('/escape_diagnostic_remaining/audit.py','/escape_diagnostic_remaining/contract.py')) and frame.f_code.co_name!='<module>':
        raise RuntimeError('Forbidden simulation/sensory call: '+key)


def write_table(name,rows):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader()
        for r in rows: writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def main():
    """Incomplete gzip is read as flushed saved bytes, never silently repaired."""
    sys.setprofile(guard)
    from flybrain.escape_diagnostic_remaining.audit import audit,check_chunk,exact
    sealed=audit(ROOT); assert sealed['status']=='closed-incomplete'
    config=json.loads((ROOT/'config.json').read_text()); manifest=json.loads((ROOT/'circuit.json').read_text())
    graph=dict(np.load(ROOT/'graph.npz',allow_pickle=False)); nodes=manifest['nodes']
    b3=next(i for i,n in enumerate(nodes) if n['metadata'].get('cell_type')=='b3' and n['metadata'].get('side')=='left')
    dn=[i for i,n in enumerate(nodes) if n['metadata'].get('cell_type') in ('DNp03','DNp26') and n['metadata'].get('side')=='right']
    de=np.flatnonzero(np.isin(graph['pre'],dn)&(graph['post']==b3)&graph['effective'])
    saved={}; compressed={}
    for d in sorted(ROOT.glob('[0-9][0-9]-*')):
        decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
        raw=decoder.decompress((d/'intervals.jsonl.gz').read_bytes())
        saved[int(d.name.split('-')[1])]=[json.loads(line) for line in raw.splitlines()]
        compressed[d.name]=decoder.eof
    expected={seed:[] for seed in saved}
    with gzip.open('experiments/turnflee-2026-10-06/traces.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['phase']=='validation' and r['candidate']==1 and r['seed'] in expected: expected[r['seed']].append(r)
    summaries=[]; timeline=[]; complete_chunks=0; full_records=0
    for ordinal,c in enumerate(config['cases']):
        d=ROOT/f'{ordinal:02d}-{c["seed"]}'
        if not d.exists():
            summaries.append(dict(seed=c['seed'],case_order=ordinal+1,status='not started',chunks=0,records=0))
            continue
        rows=saved[c['seed']]; rolling=deque(maxlen=10); previous={}; cumulative=0
        spikes={b:[] for b in ('actual','reference')}; deliveries=[]; first_diff=None; trigger=None
        expiration=[]; old_roll=0; trigger_actual=None; trigger_reference=None
        for k,p in enumerate(sorted(d.glob('*.npz'))):
            arrays=dict(np.load(p,allow_pickle=False)); measured=check_chunk(arrays,graph,k*20+1)
            for b in spikes:
                if k:
                    for field in ('voltage','previous','phase'): exact(arrays[b+'_initial_'+field],previous[b][field],b+' episode continuity')
                else:
                    for field in ('voltage','previous','phase'): exact(arrays[b+'_initial_'+field],np.zeros(589),b+' episode reset')
                previous[b]=dict(voltage=arrays[b+'_post_voltage'][-1],previous=arrays[b+'_spikes'][-1].astype(float),phase=arrays[b+'_post_phase'][-1])
                spikes[b].extend(arrays['step_index'][arrays[b+'_spikes'][:,b3]].tolist())
            if first_diff is None:
                where=np.flatnonzero(arrays['actual_spikes'][:,b3]!=arrays['reference_spikes'][:,b3])
                if len(where): first_diff=int(arrays['step_index'][where[0]])
            where=np.flatnonzero(np.any(arrays['actual_contribution'][:,de]!=arrays['reference_contribution'][:,de],axis=1))
            deliveries.extend(arrays['step_index'][where].tolist())
            inc=measured['actual']['counts']-measured['reference']['counts']
            expired=rolling[0] if len(rolling)==10 else np.zeros(589,dtype=np.int64)
            rolling.append(inc); roll=np.sum(rolling,axis=0); cumulative+=int(inc[b3])
            if int(roll[b3])==-1 and old_roll>=0 and int(expired[b3])>0 and int(inc[b3])==0:
                expiration.append((k+1)*20)
            old_roll=int(roll[b3])
            if k<len(rows):
                r=rows[k]; old=r['original']; assert old==expected[c['seed']][k]
                assert hashlib.sha256(arrays['rgb'].tobytes()).hexdigest()==old['eyes_sha256']
                for b,diag in [('actual',old['pathway']),('reference',r['reference_pathway'])]:
                    exact(measured[b]['counts'],old[b+'_counts'],b+' counts')
                    for field in ('current_sum','current_peak','discarded_current','peak_voltage','peak_phase'):
                        exact(measured[b][field],diag[field],b+' '+field)
                exact(roll,r['rolling_all'],'rolling all'); exact(expired,r['expired_increment'],'expiry')
                exact(roll[graph['wing_indices']],old['wing_vector'],'original wing vector')
                assert int(r['cumulative_increment'][b3])==cumulative
                if trigger is None and old['selected_side'] is not None:
                    trigger=(k+1)*20; trigger_actual=[s for s in spikes['actual'] if trigger-200<s<=trigger]
                    trigger_reference=[s for s in spikes['reference'] if trigger-200<s<=trigger]
                full_records+=1
            timeline.append(dict(seed=c['seed'],end_step=(k+1)*20,actual_b3_spikes=arrays['step_index'][arrays['actual_spikes'][:,b3]].tolist(),
                reference_b3_spikes=arrays['step_index'][arrays['reference_spikes'][:,b3]].tolist(),
                b3_entered=int(inc[b3]),b3_expired=int(expired[b3]),b3_rolling=int(roll[b3]),b3_cumulative=cumulative,
                full_interval_record=k<len(rows)))
            complete_chunks+=1
        status='completed' if len(rows)==len(list(d.glob('*.npz')))==75 and compressed[d.name] else 'interrupted'
        summaries.append(dict(seed=c['seed'],case_order=ordinal+1,status=status,chunks=len(list(d.glob('*.npz'))),records=len(rows),
            trigger_step=trigger,selected=rows[-1]['original']['selected_side'],final_cumulative_b3=cumulative,
            first_b3_difference_step=first_diff,first_differential_right_DN_delivery_step=min(deliveries) if deliveries else None,
            trigger_actual_b3_steps=trigger_actual,trigger_reference_b3_steps=trigger_reference,
            deficit_expiration_steps=[s for s in expiration if trigger is not None and s<=trigger],
            actual_b3_steps=spikes['actual'],reference_b3_steps=spikes['reference']))
        print(f'partial audit seed {c["seed"]}: {len(rows)} full records, {len(list(d.glob("*.npz")))} exact chunks',flush=True)
    assert complete_chunks==505 and full_records==505
    evidence=[]
    for s in summaries:
        if s['seed'] not in (1100012,1100013): continue
        t=s['trigger_step']; a=[v for v in s['actual_b3_steps'] if v<=t]; b=[v for v in s['reference_b3_steps'] if v<=t]
        supported=(s['status']=='completed' and len(s['trigger_actual_b3_steps'])-len(s['trigger_reference_b3_steps'])==-1
            and bool(set(a)-set(b)) and bool(set(b)-set(a)) and bool(s['deficit_expiration_steps'])
            and s['final_cumulative_b3']>0 and s['first_differential_right_DN_delivery_step']<s['first_b3_difference_step'])
        evidence.append(dict(seed=s['seed'],four_conditions_observed=bool(supported),trigger_step=t,
            actual_window_steps=s['trigger_actual_b3_steps'],reference_window_steps=s['trigger_reference_b3_steps'],
            expiration_steps=s['deficit_expiration_steps'],final_cumulative_b3=s['final_cumulative_b3'],
            first_right_DN_delivery=s['first_differential_right_DN_delivery_step'],first_b3_difference=s['first_b3_difference_step']))
    sys.setprofile(None)
    result=dict(status='failure-audit-passed',hypothesis='unresolved: four of five deficit-only failures completed across runs',
        completed_cases=6,started_cases=7,full_historical_records=full_records,complete_neural_chunks=complete_chunks,
        individual_kernel_steps=20200,simulated_seconds=10.1,cumulative_diagnostic_steps=42360,cumulative_diagnostic_seconds=21.18,
        attempted_reservation_steps=20240,cumulative_attempted_reservation_steps=42400,unexecuted_attempted_steps=40,
        added_audit_kernel_steps=0,brain_sensor_simulation_calls=0,all_records_equal_original=True,
        all_recorded_cell_edge_steps_exact=True,unlogged_last_interval='None. Next reservation failed before either kernel; 40 in-memory reserved steps were not executed.',
        partial_five_case_evidence=evidence,case_summaries=summaries,project_calls=dict(calls))
    (OUT/'failure-audit-final.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    write_table('final-partial-timelines.csv',timeline)
    write_table('final-partial-case-status.csv',[{k:s.get(k) for k in ('seed','case_order','status','chunks','records','trigger_step','selected','final_cumulative_b3','first_b3_difference_step','first_differential_right_DN_delivery_step')} for s in summaries])
    write_table('final-partial-deficit-evidence.csv',evidence)
    print(json.dumps({k:v for k,v in result.items() if k not in ('project_calls','case_summaries')},indent=2))


if __name__=='__main__': main()
