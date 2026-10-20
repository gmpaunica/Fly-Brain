"""One fresh, zero-update matched assay of the frozen projection readout."""

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random

import torch

from flybrain.neural.banc_projection import load_checkpoint
from .banc_projection import DEFAULT_OUTPUT as CANDIDATE,evaluate_group,coverage_gates
from .banc_projection_fit import load_training_protocol
from .banc_memory_evaluation import assess_split
from .banc_protocol import sha256,software_record
from .banc_readiness import PROJECT_ROOT,capture_provenance,verify_provenance,save_new_json
from .banc_training import _write_csv
from .evaluation import EpisodeCase,EvaluationConfig,summarize,_git_commit
from .records import ensure_new_run_directory


DEFAULT_OUTPUT=PROJECT_ROOT/'experiments/fb10-projection-evaluation'
CENTRAL_RECEIPT=CANDIDATE/'fresh-evaluation.execution.json'
CONTROLLERS=('trained','untrained','fixed')


def historical_seeds(exclude: Path | None=None) -> set[int]:
    """Scan every prior experiment identity; exclude only this fixed assay's files."""
    seeds=set()
    def visit(v):
        if isinstance(v,dict):
            for key,item in v.items():
                if key.endswith('seed') and type(item) is int:seeds.add(item)
                if key=='seeds' and isinstance(item,list):seeds.update(x for x in item if type(x) is int)
                visit(item)
        elif isinstance(v,list):
            for item in v:visit(item)
    for path in (PROJECT_ROOT/'experiments').rglob('*.json'):
        if path.resolve().is_relative_to(DEFAULT_OUTPUT.resolve()):continue
        if exclude is not None and path.resolve().is_relative_to(Path(exclude).resolve()):continue
        visit(json.loads(path.read_text(encoding='utf-8-sig')))
    return seeds


def fresh_cases() -> dict:
    """Fixed fresh 720000/721000 starts; same annulus and independent headings."""
    groups={}
    for split,start,count in (('validation',720000,16),('final',721000,32)):
        rows=[]
        for seed in range(start,start+count):
            rng=random.Random(seed);angle=rng.uniform(0,math.tau)
            radius=math.sqrt(rng.uniform(60**2,180**2));heading=rng.uniform(0,math.tau)
            rows.append(EpisodeCase(f'fb10-projection-{split}-{seed}',split,seed,
                600+radius*math.cos(angle),300+radius*math.sin(angle),heading))
        groups[split]=tuple(rows)
    return groups


def training_evidence():
    """Recompute both error gates; verify trained/untrained strict checkpoints."""
    p,digest,data=load_training_protocol(CANDIDATE/'protocol.json')
    r=json.loads((CANDIDATE/'results.json').read_text(encoding='utf-8'))
    if (r.get('training_gate_passed') is not True or r.get('failure') is not None
            or r.get('optimizer_steps')!=500 or r.get('protocol_sha256')!=digest
            or r.get('held_out_evaluated') is not False or not all(r['checks'].values())
            or sha256(CANDIDATE/'trained.pt')!=r['checkpoint_sha256']):
        raise ValueError('passing unchanged projection fit evidence required')
    mean=torch.tensor(p['normalization']['mean']);scale=torch.tensor(p['normalization']['scale'])
    models={name:load_checkpoint(CANDIDATE/(name+'.pt'),protocol_sha256=digest,mean=mean,scale=scale) for name in ('trained','untrained')}
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.no_grad():error=(models['trained'](data.features).double()-data.targets).square()
        aggregate=error.mean(0).tolist();original=error[:1008].mean(0).tolist()
    finally:torch.set_num_threads(previous)
    if (aggregate!=r['per_output_mse'] or original!=r['original_per_output_mse']
            or coverage_gates(aggregate[1],original[1],p['train_only_baseline']['constant_mse'][1],r['summaries'],r['checks'])!=r['gates']):
        raise ValueError('actual projection errors/gates differ')
    for m in models.values():m.requires_grad_(False)
    return p,digest,models


def protocol_definition() -> dict:
    p,digest,_=training_evidence()
    return {'schema':'fb10_fresh_projection_evaluation_v1','experiment':'fb10_fresh_projection_evaluation',
        'hypothesis':'The frozen projection-input and endpoint readout improves fresh food arrival over its matched untrained initialization and fixed BANC.',
        'training_protocol_sha256':digest,'candidate_directory':str(CANDIDATE),
        'candidate_sha256':{n:sha256(CANDIDATE/n) for n in ('protocol.json','results.json','trained.pt','untrained.pt')},
        'cases':{s:[asdict(c) for c in group] for s,group in fresh_cases().items()},'config':p['world_config'],
        'software':software_record(),'network_size':83,'trainable_parameters':370,'optimizer_steps':0,'learning_rate':None,
        'objective':'Fresh food arrival with paired matched controls; progress includes every failure',
        'budget':{'optimizer_steps':0,'validation_episodes':48,'final_episodes_if_validation_passes':96,
            'maximum_episodes':144,'maximum_world_ticks':86400,'maximum_neural_steps':1728000},
        'gates':{'validation_arrivals_min':13,'validation_strict_matched_improvement':True,
            'final_arrivals_min':26,'final_paired_arrival_difference_min':8,'final_one_sided_p_max_per_control':.025,
            'strict_all_case_progress_improvement':True,'frozen_candidate_core_and_history':True},
        'stop_condition':'One central consumed receipt; stop before final if validation fails. Zero fitting/tuning/retries or historical-case reuse; preserve failures.'}


def prepare_protocol(output_dir: Path) -> Path:
    """Lock the new matched assay before any case outcome is observed."""
    out=Path(output_dir).resolve();ensure_new_run_directory(out);definition=protocol_definition()
    seeds={c['seed'] for group in definition['cases'].values() for c in group}
    if seeds.intersection(historical_seeds(out)):raise ValueError('fresh seeds overlap prior records')
    p={**definition,'output_dir':str(out),'prepared_utc':datetime.now(timezone.utc).isoformat(),
        'git_commit':_git_commit(),'provenance':capture_provenance(PROJECT_ROOT,out)}
    out.mkdir(parents=True);path=out/'protocol.json';save_new_json(path,p)
    with path.with_suffix('.sha256').open('x',encoding='utf-8') as stream:stream.write(sha256(path)+'\n')
    return path


def load_protocol(path: Path):
    path=Path(path).resolve();digest=sha256(path)
    if path.with_suffix('.sha256').read_text(encoding='utf-8').strip()!=digest:raise ValueError('fresh projection hash differs')
    p=json.loads(path.read_text(encoding='utf-8'))
    if path!=Path(p['output_dir']).resolve()/'protocol.json':raise ValueError('evaluation destination differs')
    if any(p.get(k)!=v for k,v in protocol_definition().items()):raise ValueError('frozen candidate/cases/settings differ')
    if not verify_provenance(p['provenance']):raise ValueError('evaluation core/history differs')
    seeds={c['seed'] for group in p['cases'].values() for c in group}
    if seeds.intersection(historical_seeds(path.parent)):raise ValueError('fresh cases overlap prior records')
    return p,digest


def run_evaluation(path: Path) -> dict:
    """Separate validation and final stepping, without modifying the frozen head."""
    path=Path(path).resolve();out=path.parent
    if CENTRAL_RECEIPT.exists() or any((out/n).exists() for n in ('execution.json','results.json','episodes.jsonl','aborted.json')):
        raise FileExistsError('projection fresh assay already consumed')
    p,digest=load_protocol(path);_,_,models=training_evidence()
    initial={name:{k:v.clone() for k,v in m.state_dict().items()} for name,m in models.items()}
    receipt={'status':'consumed','protocol_sha256':digest,'output_dir':str(out),'started_utc':datetime.now(timezone.utc).isoformat()}
    save_new_json(CENTRAL_RECEIPT,receipt);save_new_json(out/'execution.json',receipt)
    previous=torch.get_num_threads();torch.set_num_threads(1)
    completed=0;ticks=0;steps=0;results={};summaries={};gates={};weights_unchanged=True
    try:
        with (out/'episodes.jsonl').open('x',encoding='utf-8') as journal:
            for split in ('validation','final'):
                if split=='final' and not gates['validation']['passed']:break
                load_protocol(path);cases=tuple(EpisodeCase(**c) for c in p['cases'][split]);counts={n:0 for n in CONTROLLERS}
                def record(name,row):
                    nonlocal completed,ticks
                    completed+=1;ticks+=row.ticks;counts[name]+=1
                    journal.write(json.dumps({'condition':name,'result':asdict(row)},allow_nan=False)+'\n');journal.flush()
                    if completed>144 or ticks>86400:raise RuntimeError('fresh assay budget exceeded')
                    if counts[name]%4==0:print(f'{split}/{name}: {counts[name]}/{len(cases)}',flush=True)
                rows,account=evaluate_group(cases,models,EvaluationConfig(**p['config']),record)
                steps+=account['neural_steps'];weights_unchanged &= account['effective_weights_unchanged']
                summaries[split]={n:summarize(group) for n,group in rows.items()}
                gates[split]=assess_split(split,{n:[r.outcome=='success' for r in group] for n,group in rows.items()},summaries[split])
                results[split]={n:[asdict(r) for r in group] for n,group in rows.items()}
                if not weights_unchanged or any(not torch.equal(v,initial[n][k]) for n,m in models.items() for k,v in m.state_dict().items()):
                    raise RuntimeError('frozen readout/core changed')
                _write_csv(out/(split+'.csv'),[dict(condition=n,**r) for n,group in results[split].items() for r in group])
                save_new_json(out/(split+'.json'),{'gates':gates[split],'summaries':summaries[split],'results':results[split]})
                print(f"{split}: arrivals={gates[split]['successes']}; passed={gates[split]['passed']}",flush=True)
        load_protocol(path)
        if steps!=ticks*20 or steps>1728000:raise RuntimeError('fresh neural budget mismatch')
        r={'experiment':p['experiment'],'protocol_sha256':digest,'software':software_record(),'git_commit':p['git_commit'],
            'network_size':83,'objective':p['objective'],'optimizer_steps':0,'learning_rate':None,
            'checkpoint':str(CANDIDATE/'trained.pt'),'checkpoint_sha256':sha256(CANDIDATE/'trained.pt'),
            'actual_episodes':completed,'actual_world_ticks':ticks,'actual_neural_steps':steps,
            'results':results,'summaries':summaries,'gates':gates,'final_cases_evaluated':'final' in results,
            'frozen_parameters_unchanged':True,'effective_weights_unchanged':weights_unchanged,
            'integrity_checks_passed':True,'fb10_gate_passed':gates.get('final',{}).get('passed',False),
            'completed_utc':datetime.now(timezone.utc).isoformat()}
        save_new_json(out/'results.json',r);return r
    except Exception as e:
        save_new_json(out/'aborted.json',{'type':type(e).__name__,'error':str(e),'completed_episodes':completed,
            'actual_world_ticks':ticks,'protocol_sha256':digest,'instruction':'Preserve partial consumed assay; no automatic retry.'});raise
    finally:torch.set_num_threads(previous)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT);parser.add_argument('--protocol',type=Path,default=DEFAULT_OUTPUT/'protocol.json');a=parser.parse_args()
    if a.prepare:
        path=prepare_protocol(a.output_dir);print(json.dumps({'protocol':str(path),'sha256':sha256(path),'updates':0,'ticks':0}));return 0
    r=run_evaluation(a.protocol);print(json.dumps({k:r[k] for k in ('fb10_gate_passed','actual_episodes','actual_world_ticks','actual_neural_steps')},indent=2))
    return 0 if r['fb10_gate_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
