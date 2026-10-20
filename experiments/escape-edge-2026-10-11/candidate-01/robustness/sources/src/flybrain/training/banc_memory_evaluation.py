"""Frozen fresh matched FB-10 evaluation; never fits or changes the candidate."""

import argparse
from collections.abc import Callable, Sequence
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import random

import torch

from flybrain.neural.banc_memory import make_memory_controller, load_checkpoint
from flybrain.neural.banc_motor import make_banc_controller
from flybrain.neural.banc_network import load_verified_banc_config
from .banc_memory import DEFAULT_OUTPUT as CANDIDATE, load_protocol as load_training_protocol
from .banc_protocol import sha256, software_record
from .banc_readiness import (PROJECT_ROOT,capture_provenance,verify_provenance,
                             save_new_json,weights_sha256)
from .banc_training import training_gates, _write_csv
from .charge_evaluation import assess_split as arrival_gate
from .evaluation import EpisodeCase,EpisodeResult,EvaluationConfig,run_episode,summarize,_git_commit
from .records import ensure_new_run_directory


DEFAULT_OUTPUT=PROJECT_ROOT/'experiments/fb10-banc-evaluation'
CANDIDATE=(PROJECT_ROOT/CANDIDATE).resolve()
CENTRAL_RECEIPT=CANDIDATE/'fresh-evaluation.execution.json'
CONTROLLERS=('trained','untrained','fixed')


def historical_seeds(exclude: Path | None = None) -> set[int]:
    """Collect earlier experiment seeds, excluding this evaluation's own record.

    These are the histories preceding this fixed protocol. Its own cases become
    historical after execution and cannot be reopened because of its receipt.
    """
    seeds=set()
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key.endswith('seed') and type(item) is int:seeds.add(item)
                if key=='seeds' and isinstance(item,list):seeds.update(x for x in item if type(x) is int)
                visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    for path in (PROJECT_ROOT/'experiments').rglob('*.json'):
        if path.resolve().is_relative_to(DEFAULT_OUTPUT.resolve()):continue
        if exclude is not None and path.resolve().is_relative_to(Path(exclude).resolve()):continue
        visit(json.loads(path.read_text(encoding='utf-8-sig')))
    return seeds


def fresh_cases() -> dict[str,tuple[EpisodeCase,...]]:
    """The fixed, separately seeded 16/32-case protocol; no selection by outcome."""
    groups={}
    for split,start,count in (('validation',620000,16),('final',621000,32)):
        rows=[]
        for seed in range(start,start+count):
            rng=random.Random(seed)
            angle=rng.uniform(0,math.tau);radius=math.sqrt(rng.uniform(60**2,180**2))
            heading=rng.uniform(0,math.tau)
            rows.append(EpisodeCase(f'fb10-{split}-{seed}',split,seed,
                600+radius*math.cos(angle),300+radius*math.sin(angle),heading))
        groups[split]=tuple(rows)
    return groups


def assess_split(split: str,outcomes: dict,summaries: dict) -> dict:
    """Apply the original fresh arrival/effect-size gates plus all-case progress."""
    gate=arrival_gate(split,outcomes)
    progress=[summaries[name]['mean_progress_fraction'] for name in CONTROLLERS]
    valid=all(value is not None and math.isfinite(value) for value in progress)
    gate['matched_progress_passed']=valid and all(progress[0]>x for x in progress[1:])
    gate['passed']=gate['passed'] and gate['matched_progress_passed']
    return gate


def training_evidence() -> tuple[dict,str,dict]:
    """Verify actual passing fit/checkpoints before preparing or evaluating cases."""
    p,digest,data=load_training_protocol(CANDIDATE/'protocol.json')
    r=json.loads((CANDIDATE/'results.json').read_text(encoding='utf-8'))
    if (r.get('training_gate_passed') is not True or r.get('failure') is not None
            or r.get('optimizer_steps')!=500 or r.get('protocol_sha256')!=digest
            or r.get('held_out_evaluated') is not False or not all(r['checks'].values())):
        raise ValueError('actual passing training evidence required')
    if sha256(CANDIDATE/'trained.pt')!=r['checkpoint_sha256']:
        raise ValueError('trained checkpoint bytes differ')
    mean=torch.tensor(p['normalization']['mean']);scale=torch.tensor(p['normalization']['scale'])
    models={name:load_checkpoint(CANDIDATE/(name+'.pt'),protocol_sha256=digest,mean=mean,scale=scale)
            for name in ('trained','untrained')}
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.no_grad():errors=(models['trained'](data.features).double()-data.targets).square().mean(0).tolist()
    finally:torch.set_num_threads(previous_threads)
    if errors!=r['per_output_mse'] or training_gates(errors[1],p['train_only_baseline']['constant_mse'][1],r['summaries'],r['checks'])!=r['gates']:
        raise ValueError('recorded scientific training gates differ from candidate')
    for model in models.values():model.requires_grad_(False)
    return p,digest,models


def protocol_definition() -> dict:
    """Lock one held-out assay and explicit failure stops before any episodes."""
    p,digest,_=training_evidence()
    return {'schema':'fb10_fresh_memory_evaluation_v1','experiment':'fb10_fresh_memory_evaluation',
        'hypothesis':'The frozen learned temporal endpoint readout improves food arrival on fresh starts over its matched untrained initialization and fixed BANC.',
        'training_protocol_sha256':digest,'candidate_directory':str(CANDIDATE),
        'candidate_sha256':{name:sha256(CANDIDATE/name) for name in ('protocol.json','results.json','trained.pt','untrained.pt')},
        'cases':{split:[asdict(c) for c in group] for split,group in fresh_cases().items()},
        'config':p['world_config'],'software':software_record(),'network_size':83,
        'trainable_parameters':306,'optimizer_steps':0,'learning_rate':None,
        'objective':'Fresh food-arrival rate with paired trained/untrained/fixed comparisons; all-case progress includes failures',
        'budget':{'optimizer_steps':0,'validation_episodes':48,'final_episodes_if_validation_passes':96,
                  'maximum_episodes':144,'maximum_world_ticks':86400,'maximum_neural_steps':1728000},
        'gates':{'validation_arrivals_min':13,'validation_strict_matched_improvement':True,
            'final_arrivals_min':26,'final_paired_arrival_difference_min':8,'final_one_sided_p_max_per_control':.025,
            'strict_all_case_progress_improvement':True,'frozen_candidate_core_and_history':True},
        'stop_condition':'Consume once; stop before final if validation fails. No optimizer, tuning, retries, relabeling, topology intervention or fresh-case reuse. Preserve partial failures.'}


def prepare_protocol(output_dir: Path) -> Path:
    """Freeze fresh matched cases and candidate/data/source bytes with zero steps."""
    out=Path(output_dir).resolve();ensure_new_run_directory(out)
    definition=protocol_definition()
    seeds={row['seed'] for group in definition['cases'].values() for row in group}
    if seeds.intersection(historical_seeds(out)):raise ValueError('fresh seeds overlap prior experiments')
    p={**definition,'output_dir':str(out),'prepared_utc':datetime.now(timezone.utc).isoformat(),
       'git_commit':_git_commit(),'provenance':capture_provenance(PROJECT_ROOT,out)}
    out.mkdir(parents=True);path=out/'protocol.json';save_new_json(path,p)
    with path.with_suffix('.sha256').open('x',encoding='utf-8') as stream:stream.write(sha256(path)+'\n')
    return path


def load_protocol(path: Path) -> tuple[dict,str]:
    """Fail closed on copied, stale or resealed candidate/case/source settings."""
    path=Path(path).resolve();digest=sha256(path)
    if path.with_suffix('.sha256').read_text(encoding='utf-8').strip()!=digest:raise ValueError('evaluation hash differs')
    p=json.loads(path.read_text(encoding='utf-8'))
    if path!=Path(p['output_dir']).resolve()/'protocol.json':raise ValueError('evaluation destination differs')
    if any(p.get(k)!=v for k,v in protocol_definition().items()):raise ValueError('frozen candidate/cases/gates differ')
    if not verify_provenance(p['provenance']):raise ValueError('evaluation source/core/history differs')
    seeds={row['seed'] for group in p['cases'].values() for row in group}
    if seeds.intersection(historical_seeds(path.parent)):raise ValueError('case seeds overlap earlier histories')
    return p,digest


def evaluate_group(cases: Sequence[EpisodeCase],models: dict,config: EvaluationConfig,
                   on_episode: Callable[[str,EpisodeResult],None] | None = None) -> tuple[dict,dict]:
    """Measure three independent matched conditions; account actual neural steps."""
    if not cases or set(models)!={'trained','untrained'}:raise ValueError('expected cases and both candidate models')
    rows={};controllers=[];digests=[]
    for name in CONTROLLERS:
        group=[]
        for case in cases:
            core=(make_banc_controller(load_verified_banc_config()) if name=='fixed'
                  else make_memory_controller(models[name]))
            controllers.append(core);digests.append(weights_sha256(core))
            with torch.no_grad():row=run_episode(case,config,lambda:core)
            group.append(row)
            if on_episode is not None:on_episode(name,row)
        rows[name]=group
    ticks=sum(row.ticks for group in rows.values() for row in group)
    steps=sum(c.network.population.step_count for c in controllers)
    if steps!=20*ticks:raise RuntimeError('neural/world counter mismatch')
    return rows,{'episodes':len(controllers),'world_ticks':ticks,'neural_steps':steps,
        'effective_weights_unchanged':len(set(digests))==1 and all(weights_sha256(c)==h for c,h in zip(controllers,digests))}


def run_evaluation(path: Path) -> dict:
    """Consume the frozen assay globally; final stepping waits for validation."""
    path=Path(path).resolve();out=path.parent
    if CENTRAL_RECEIPT.exists() or any((out/name).exists() for name in ('execution.json','results.json','episodes.jsonl','aborted.json')):
        raise FileExistsError('fresh evaluation already consumed; no renamed retry')
    p,digest=load_protocol(path)
    _,_,models=training_evidence()
    initial={name:{k:v.clone() for k,v in model.state_dict().items()} for name,model in models.items()}
    receipt={'protocol_sha256':digest,'output_dir':str(out),'started_utc':datetime.now(timezone.utc).isoformat(),'status':'consumed'}
    save_new_json(CENTRAL_RECEIPT,receipt);save_new_json(out/'execution.json',receipt)
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    completed=0;ticks=0;steps=0;results={};summaries={};gates={};weights_unchanged=True
    try:
        with (out/'episodes.jsonl').open('x',encoding='utf-8') as journal:
            for split in ('validation','final'):
                if split=='final' and not gates['validation']['passed']:break
                load_protocol(path)
                cases=tuple(EpisodeCase(**row) for row in p['cases'][split]);counts={name:0 for name in CONTROLLERS}
                def record(name,row):
                    nonlocal completed,ticks
                    completed+=1;ticks+=row.ticks;counts[name]+=1
                    journal.write(json.dumps({'condition':name,'result':asdict(row)},allow_nan=False)+'\n');journal.flush()
                    if completed>p['budget']['maximum_episodes'] or ticks>p['budget']['maximum_world_ticks']:
                        raise RuntimeError('evaluation budget exceeded')
                    if counts[name]%4==0:print(f'{split}/{name}: {counts[name]}/{len(cases)}',flush=True)
                rows,account=evaluate_group(cases,models,EvaluationConfig(**p['config']),record)
                steps+=account['neural_steps'];weights_unchanged &= account['effective_weights_unchanged']
                summaries[split]={name:summarize(group) for name,group in rows.items()}
                gates[split]=assess_split(split,{name:[r.outcome=='success' for r in group] for name,group in rows.items()},summaries[split])
                results[split]={name:[asdict(r) for r in group] for name,group in rows.items()}
                if not weights_unchanged or any(not torch.equal(v,initial[name][k]) for name,model in models.items() for k,v in model.state_dict().items()):
                    raise RuntimeError('frozen core/readout changed during evaluation')
                _write_csv(out/(split+'.csv'),[dict(condition=name,**row) for name,group in results[split].items() for row in group])
                save_new_json(out/(split+'.json'),{'gate':gates[split],'summaries':summaries[split],'results':results[split]})
                print(f"{split}: arrivals={gates[split]['successes']}; gate passed={gates[split]['passed']}",flush=True)
        load_protocol(path)
        if steps!=ticks*20 or steps>p['budget']['maximum_neural_steps']:raise RuntimeError('neural-step budget mismatch')
        r={'experiment':p['experiment'],'protocol_sha256':digest,'software':software_record(),'git_commit':p['git_commit'],
            'network_size':83,'objective':p['objective'],'optimizer_steps':0,'learning_rate':None,
            'checkpoint':str(CANDIDATE/'trained.pt'),'checkpoint_sha256':sha256(CANDIDATE/'trained.pt'),
            'actual_episodes':completed,'actual_world_ticks':ticks,'actual_neural_steps':steps,
            'results':results,'summaries':summaries,'gates':gates,'final_cases_evaluated':'final' in results,
            'frozen_parameters_unchanged':True,'effective_weights_unchanged':weights_unchanged,
            'integrity_checks_passed':True,'fb10_gate_passed':gates.get('final',{}).get('passed',False),
            'completed_utc':datetime.now(timezone.utc).isoformat()}
        save_new_json(out/'results.json',r);return r
    except Exception as error:
        save_new_json(out/'aborted.json',{'type':type(error).__name__,'error':str(error),'completed_episodes':completed,
            'actual_world_ticks':ticks,'protocol_sha256':digest,'instruction':'preserve partial attempt; no automatic retry'})
        raise
    finally:torch.set_num_threads(previous_threads)


def main() -> int:
    """Prepare the fresh locked protocol or execute it once without any training."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--protocol',type=Path,default=DEFAULT_OUTPUT/'protocol.json');args=parser.parse_args()
    if args.prepare:
        path=prepare_protocol(args.output_dir);print(json.dumps({'protocol':str(path),'sha256':sha256(path),'updates':0,'ticks':0}));return 0
    r=run_evaluation(args.protocol);print(json.dumps({k:r[k] for k in ('fb10_gate_passed','actual_episodes','actual_world_ticks','optimizer_steps')},indent=2))
    return 0 if r['fb10_gate_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
