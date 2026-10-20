"""One capped propagation/causality assay of the fixed range encoder repair."""

import csv
from datetime import datetime,timezone
import json

import torch

from flybrain.neural.banc_range import make_range_controller,range_contract
from .banc_recovery import (pulse_cases,measure_pulse,steering_delays,contrast_gate,sensory_counts_match)
from .banc_protocol import sha256,software_record
from .banc_readiness import PROJECT_ROOT,capture_provenance,verify_provenance,save_new_json
from .banc_coverage import _write_protocol
from .records import ensure_new_run_directory


OUTPUT=PROJECT_ROOT/'experiments/fb10-range-readiness'


def run_readiness(output_dir=OUTPUT):
    """Predeclare 26 pulses and four cuts; no fitting/world/candidate selection."""
    out=output_dir.resolve();ensure_new_run_directory(out)
    cases=pulse_cases()
    for center in (1.,2.):
        for pattern,left,right in (('left',center,0.),('right',0.,center),('equal',center,center),
            ('right_stronger',.8*center,1.2*center),('left_stronger',1.2*center,.8*center)):
            cases.append({'name':f'{center:g}-{pattern}','center':center,'pattern':pattern,'left':left,'right':right})
    c=make_range_controller();brain=c.network;delays=steering_delays(brain.topology)
    p={'experiment':'fb10_range_readiness','prepared_utc':datetime.now(timezone.utc).isoformat(),
        'hypothesis':'The fixed sqrt/log encoder preserves transient bounded directional and sourced causal propagation over the arena odor range.',
        'contract':range_contract(),'cases':cases,'endpoint_window_delays':delays,'software':software_record(),
        'max_neural_steps':45000,'max_world_ticks':0,'optimizer_steps':0,
        'gates':'Original propagation, finite, silence, washout and <=250Hz; aligned >=1Hz/2% directional contrasts; original four causal cuts',
        'stop_condition':'One fixed encoder,26pulses,four conditional cuts. Close on failed gates; no slope/range sweep or fit.',
        'output_dir':str(out),'provenance':capture_provenance(PROJECT_ROOT,out)}
    path=_write_protocol(out,p);digest=sha256(path);save_new_json(out/'execution.json',{'status':'consumed','protocol_sha256':digest})
    previous=torch.get_num_threads();torch.set_num_threads(1)
    rows=[];contrasts={};controls=[];steps=0
    try:
        for case in cases:
            rows.append(measure_pulse(brain,case,endpoint_delays=delays));steps+=1500
        for center in (.025,.1,.25,1.,2.):
            rates={r['case']['pattern']:r['steering_rates_hz'] for r in rows if r['case']['center']==center}
            contrasts[str(center)]=contrast_gate(rates['equal'],rates['right_stronger'],rates['left_stronger'])
        dynamics=all(r['passed'] for r in rows) and all(r['passed'] for r in contrasts.values())
        if dynamics:
            case=next(c for c in cases if c['center']==.1 and c['pattern']=='equal');intact=next(r for r in rows if r['case']==case)
            for condition in ('synapses_off','downstream_synapses_off','left_dna02_input_off','right_dna02_input_off'):
                brain=make_range_controller().network;w=brain.population.weights
                if condition=='synapses_off':w.zero_()
                elif condition=='downstream_synapses_off':w[[i for i,g in enumerate(brain.topology.groups) if not g.startswith('sensory_')],:]=0
                else:w[list(brain.topology.indices('steering_'+condition.split('_')[0])),:]=0
                brain.population=brain.population_for_weights(w)
                r=measure_pulse(brain,case,endpoint_delays=delays);steps+=1500
                groups=([g for g in r['pulse_counts'] if not g.startswith('sensory_')] if 'synapses_off' in condition
                    else ['steering_'+condition.split('_')[0],'motor_'+condition.split('_')[0]])
                checks={'external_encoding_exact':r['external_drive_sha256']==intact['external_drive_sha256'],
                    'sensory_counts_preserved':sensory_counts_match(brain.topology,intact,r,allow_sensory_edge_receivers=condition=='synapses_off'),
                    'cut_targets_silent':all(r['pulse_counts'][g]==0 for g in groups),
                    'stability_and_washout':all(v for k,v in r['gates'].items() if k!='propagation')}
                controls.append({'condition':condition,'checks':checks,'passed':all(checks.values()),'measurement':r})
        checks={'provenance_preserved':verify_provenance(p['provenance']),'protocol_unchanged':sha256(path)==digest,'budget_passed':steps<=45000}
        result={'experiment':p['experiment'],'protocol_sha256':digest,'software':software_record(),'cases':rows,'contrasts':contrasts,
            'causal_controls':controls,'checks':checks,'actual_neural_steps':steps,'actual_world_ticks':0,'optimizer_steps':0,
            'dynamics_gate_passed':dynamics,'causal_gate_passed':len(controls)==4 and all(r['passed'] for r in controls),
            'readiness_passed':dynamics and len(controls)==4 and all(r['passed'] for r in controls) and all(checks.values())}
        save_new_json(out/'results.json',result)
        with (out/'summary.csv').open('x',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['case','left_steering_hz','right_steering_hz','passed'])
            for r in rows:writer.writerow([r['case']['name'],*r['steering_rates_hz'],r['passed']])
        return result
    finally:torch.set_num_threads(previous)


if __name__=='__main__':
    r=run_readiness();print(json.dumps({k:r[k] for k in ('readiness_passed','dynamics_gate_passed','causal_gate_passed','actual_neural_steps')},indent=2))
    raise SystemExit(0 if r['readiness_passed'] else 1)
