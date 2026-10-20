"""One measured training-coverage repair: teacher labels on frozen student states."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

import torch
from torch import Tensor

from flybrain.neural.banc_memory import (MemoryBancReadout,make_memory_controller,
    memory_contract,normalization,save_checkpoint,load_checkpoint)
from flybrain.neural.controller import NeuralController,NeuralDecision
from flybrain.world.fly import MotorCommand
from flybrain.world.sensing import SensoryObservation
from .banc_memory import load_training_data as original_data,evaluate_training_models
from .banc_memory_evaluation import training_evidence
from .banc_protocol import sha256,software_record
from .banc_readiness import (PROJECT_ROOT,capture_provenance,verify_provenance,
                             save_new_json,weights_sha256)
from .banc_training import fit_readout,training_gates,_write_csv
from .evaluation import EpisodeCase,EvaluationConfig,run_episode,summarize,locked_cases,_git_commit
from .readout_experiment import teacher_command,TrainingDataset
from .records import ensure_new_run_directory


DEFAULT_COLLECTION=PROJECT_ROOT/'experiments/fb10-coverage-data'
DEFAULT_OUTPUT=PROJECT_ROOT/'experiments/fb10-coverage-readout'
ORIGINAL_TURN_CEILING=.02225273146561865


def coverage_cases() -> tuple[EpisodeCase,...]:
    """32 predeclared training starts; no validation/outcome-based selection."""
    cases=[]
    for seed in range(700000,700032):
        rng=random.Random(seed);angle=rng.uniform(0,math.tau)
        radius=math.sqrt(rng.uniform(60**2,180**2));heading=rng.uniform(0,math.tau)
        cases.append(EpisodeCase(f'coverage-train-{seed}','train',seed,
            600+radius*math.cos(angle),300+radius*math.sin(angle),heading))
    return tuple(cases)


class CoverageRecorder(NeuralController):
    """Teacher labels only; student mode retains the exact frozen body command."""

    def __init__(self,model: MemoryBancReadout,case_id: str,trajectory: str) -> None:
        if trajectory not in ('teacher','student'):raise ValueError('expected teacher/student trajectory')
        c=make_memory_controller(model);super().__init__(c.network,c.decoder)
        self.case_id=case_id;self.trajectory=trajectory;self.rows=[];self.tick=0

    def reset(self) -> None:
        """Clear both neural/EMA histories and tick labels at each episode reset."""
        super().reset();self.tick=0

    def decide(self,observation: SensoryObservation) -> NeuralDecision:
        """Record actual spike-derived features and a separate sensory-only label."""
        decision=super().decide(observation);target=teacher_command(observation)
        self.rows.append({'case_id':self.case_id,'trajectory':self.trajectory,'tick':self.tick,
            'rates':self.decoder.feature_rates.tolist(),'target_forward':float(target[0]),'target_turn':float(target[1])})
        self.tick+=1
        if self.trajectory=='teacher':
            return NeuralDecision(decision.activity,MotorCommand(float(target[0]),float(target[1])))
        return decision


def target_baseline(targets: Tensor) -> dict:
    """Train-only mean-command predictor; include all signed labels and failures."""
    if (not isinstance(targets,Tensor) or targets.ndim!=2 or targets.shape[1]!=2
            or not len(targets) or not torch.isfinite(targets).all()):raise ValueError('expected finite [N,2] targets')
    values=targets.detach().double();mean=values.mean(0)
    return {'samples':len(values),'mean_command':mean.tolist(),'constant_mse':(values-mean).square().mean(0).tolist(),
            'strong_left_labels':int((values[:,1]<-.8).sum()),'strong_right_labels':int((values[:,1]>.8).sum())}


def _write_protocol(out: Path,payload: dict) -> Path:
    out.mkdir(parents=True);path=out/'protocol.json';save_new_json(path,payload)
    with path.with_suffix('.sha256').open('x',encoding='utf-8') as stream:stream.write(sha256(path)+'\n')
    return path


def _read_protocol(path: Path) -> tuple[dict,str]:
    path=Path(path).resolve();digest=sha256(path)
    if path.with_suffix('.sha256').read_text(encoding='utf-8').strip()!=digest:raise ValueError('coverage protocol hash differs')
    p=json.loads(path.read_text(encoding='utf-8'))
    if path!=Path(p['output_dir']).resolve()/'protocol.json':raise ValueError('coverage destination differs')
    if not verify_provenance(p['provenance']):raise ValueError('coverage source/core/history differs')
    return p,digest


def collection_definition() -> dict:
    """One capped collection, with teacher readiness and no model changes."""
    p,digest,_=training_evidence()
    return {'schema':'fb10_coverage_collection_v1','experiment':'fb10_coverage_collection',
        'hypothesis':'Predetermined teacher and frozen-student trajectories cover the strong-left correction states absent from original teacher data.',
        'frozen_student_protocol_sha256':digest,'frozen_student_checkpoint_sha256':sha256(PROJECT_ROOT/'experiments/fb10-memory-readout/trained.pt'),
        'contract':memory_contract(),'software':software_record(),'world_config':p['world_config'],
        'cases':[asdict(c) for c in coverage_cases()],'trajectory_order':['teacher','student'],
        'max_episodes':64,'max_world_ticks':38400,'max_neural_steps':768000,'optimizer_steps':0,
        'gates':['teacher 32/32 arrivals','all endpoint-only rates and sensory labels finite','frozen student/core/history'],
        'stop_condition':'One teacher and one frozen-student episode per 32 fixed training starts; retain failures, no optimizer or case selection.'}


def prepare_collection(output_dir: Path) -> Path:
    """Freeze new training collection before any neural/world stepping."""
    out=Path(output_dir).resolve();ensure_new_run_directory(out)
    return _write_protocol(out,{**collection_definition(),'output_dir':str(out),
        'prepared_utc':datetime.now(timezone.utc).isoformat(),'provenance':capture_provenance(PROJECT_ROOT,out)})


def load_collection(path: Path) -> tuple[dict,str]:
    """Reject relocated or modified collection scope, cases, model or source."""
    p,digest=_read_protocol(path)
    if any(p.get(k)!=v for k,v in collection_definition().items()):raise ValueError('collection settings differ')
    return p,digest


def collect(path: Path) -> dict:
    """Consume collection once; teacher labels never enter the student inference."""
    path=Path(path).resolve();out=path.parent
    if any((out/name).exists() for name in ('execution.json','results.json','samples.csv','episodes.jsonl')):
        raise FileExistsError('coverage collection consumed')
    p,digest=load_collection(path);_,_,models=training_evidence();model=models['trained']
    initial={k:v.clone() for k,v in model.state_dict().items()}
    save_new_json(out/'execution.json',{'protocol_sha256':digest,'status':'consumed','started_utc':datetime.now(timezone.utc).isoformat()})
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    all_rows=[];episodes={};ticks=0;steps=0;weights=[];weights_after=[]
    try:
        with (out/'episodes.jsonl').open('x',encoding='utf-8') as journal:
            for trajectory in p['trajectory_order']:
                group=[]
                for i,case in enumerate(coverage_cases(),1):
                    c=CoverageRecorder(model,case.case_id,trajectory);weights.append(weights_sha256(c))
                    with torch.no_grad():row=run_episode(case,EvaluationConfig(**p['world_config']),lambda:c)
                    group.append(row);all_rows.extend(c.rows);ticks+=row.ticks;steps+=c.network.population.step_count
                    weights_after.append(weights_sha256(c))
                    journal.write(json.dumps({'trajectory':trajectory,'result':asdict(row)})+'\n');journal.flush()
                    if ticks>p['max_world_ticks'] or steps>p['max_neural_steps']:raise RuntimeError('coverage collection budget exceeded')
                    if i%8==0:print(f'{trajectory}: {i}/32',flush=True)
                episodes[trajectory]=group
        fields=['case_id','trajectory','tick',*[f'feature_{i}' for i in range(16)],'target_forward','target_turn']
        with (out/'samples.csv').open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in all_rows:writer.writerow({k:v for k,v in row.items() if k!='rates'}|{f'feature_{i}':v for i,v in enumerate(row['rates'])})
        x=torch.tensor([row['rates'] for row in all_rows],dtype=torch.float64)
        y=torch.tensor([[row['target_forward'],row['target_turn']] for row in all_rows],dtype=torch.float64)
        checks={'finite_endpoint_features_and_labels':bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            'student_parameters_unchanged':all(torch.equal(v,initial[k]) for k,v in model.state_dict().items()),
            'effective_weights_unchanged':len(set(weights))==1 and weights==weights_after,
            'provenance_preserved':verify_provenance(p['provenance']),'protocol_unchanged':sha256(path)==digest,
            'budget_and_samples':len(weights)==64 and len(all_rows)==ticks and steps==ticks*20 and ticks<=38400}
        summaries={name:summarize(rows) for name,rows in episodes.items()}
        r={'experiment':p['experiment'],'protocol_sha256':digest,'software':software_record(),
            'actual_episodes':len(weights),'actual_world_ticks':ticks,'actual_neural_steps':steps,'optimizer_steps':0,
            'sample_count':len(all_rows),'samples_sha256':sha256(out/'samples.csv'),'summaries':summaries,
            'episodes':{name:[asdict(row) for row in rows] for name,rows in episodes.items()},
            'new_training_baseline':target_baseline(y),'checks':checks,
            'readiness_passed':summaries['teacher']['successes']==32 and all(checks.values())}
        save_new_json(out/'results.json',r);return r
    except Exception as error:
        save_new_json(out/'aborted.json',{'type':type(error).__name__,'error':str(error),'actual_world_ticks':ticks,'actual_neural_steps':steps})
        raise
    finally:torch.set_num_threads(previous_threads)


def load_training_data(collection_dir: Path=DEFAULT_COLLECTION) -> TrainingDataset:
    """Combine all validated original and newly declared training rows only."""
    directory=Path(collection_dir);p,digest=load_collection(directory/'protocol.json')
    r=json.loads((directory/'results.json').read_text(encoding='utf-8'))
    if r.get('readiness_passed') is not True or r['protocol_sha256']!=digest or sha256(directory/'samples.csv')!=r['samples_sha256']:
        raise ValueError('coverage readiness did not pass or its bytes differ')
    with (directory/'samples.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    counts={};allowed={c.case_id for c in coverage_cases()}
    for row in rows:
        if row['case_id'] not in allowed or row['trajectory'] not in ('teacher','student'):raise ValueError('nontraining coverage row')
        key=row['case_id']+':'+row['trajectory'];tick=counts.get(key,0)
        if int(row['tick'])!=tick:raise ValueError('noncontiguous coverage ticks')
        counts[key]=tick+1
    expected={episode['case_id']+':'+name:episode['ticks'] for name,group in r['episodes'].items() for episode in group}
    if counts!=expected or len(rows)!=r['sample_count']:raise ValueError('coverage samples/episode counts differ')
    x=torch.tensor([[float(row[f'feature_{i}']) for i in range(16)] for row in rows],dtype=torch.float64)
    y=torch.tensor([[float(row['target_forward']),float(row['target_turn'])] for row in rows],dtype=torch.float64)
    if not torch.isfinite(x).all() or torch.any(x<0) or torch.any(x>1000) or not torch.isfinite(y).all() or torch.any(y[:,0]<0) or torch.any(y[:,0]>1) or torch.any(y[:,1].abs()>1):
        raise ValueError('invalid coverage rates or sensory labels')
    original=original_data();counts.update({key+':original-teacher':value for key,value in original.case_counts.items()})
    return TrainingDataset(torch.cat((original.features,x)),torch.cat((original.targets,y)),counts)


def coverage_gates(aggregate_turn_mse: float,original_turn_mse: float,constant_mse: float,
                   summaries: dict,checks: dict) -> dict:
    """Retain absolute original ceilings as well as the aggregate constant gate."""
    gates=training_gates(aggregate_turn_mse,constant_mse,summaries,checks)
    gates['aggregate_absolute_turn_fit']=math.isfinite(aggregate_turn_mse) and aggregate_turn_mse<=ORIGINAL_TURN_CEILING
    gates['original_training_turn_fit']=math.isfinite(original_turn_mse) and original_turn_mse<=ORIGINAL_TURN_CEILING
    gates['passed']=all(value for key,value in gates.items() if key!='passed')
    return gates


def training_definition(data: TrainingDataset) -> dict:
    """One covered-state candidate with frozen normalization and original gates."""
    mean,scale=normalization(data.features)
    return {'schema':'fb10_coverage_training_v1','experiment':'fb10_coverage_readout',
        'hypothesis':'Training on predetermined teacher and frozen-student states fills the measured strong-left coverage gap while preserving original navigation gates.',
        'contract':memory_contract(),'software':software_record(),'seed':1010,'network_size':83,
        'trainable_parameters':306,'sample_count':len(data.features),'case_counts':data.case_counts,
        'normalization':{'mean':mean.tolist(),'scale':scale.tolist()},'train_only_baseline':target_baseline(data.targets),
        'optimizer':'Adam full batch','learning_rate':.01,'optimizer_step_ceiling':500,
        'objective':'forward MSE + turn MSE on training-only covered states; no spike/filter gradients',
        'cases':[asdict(c) for c in locked_cases()['train']],'world_config':asdict(EvaluationConfig()),
        'max_episodes':25,'max_world_ticks':15000,'max_neural_steps':300000,
        'gates':{'aggregate_absolute_turn_mse_max':ORIGINAL_TURN_CEILING,'original_turn_mse_max':ORIGINAL_TURN_CEILING,
            'aggregate_constant_mse_fraction_max':.25,'original_training_arrivals_min':7,
            'strict_matched_arrival_and_progress':True,'finite_gradients_updates_frozen_core_and_exact_replay':True},
        'stop_condition':'One 500-update covered-state candidate; stop on nonfinite numerics. Record original matched training episodes after finite fitting, close on failed gates; no sweep/retry or held-out stepping.'}


def prepare_training(output_dir: Path,collection_dir: Path=DEFAULT_COLLECTION) -> Path:
    """Freeze covered training samples and baseline before any optimizer update."""
    out=Path(output_dir).resolve();ensure_new_run_directory(out);collection=Path(collection_dir).resolve()
    data=load_training_data(collection)
    return _write_protocol(out,{**training_definition(data),'output_dir':str(out),'collection_dir':str(collection),
        'collection_sha256':{name:sha256(collection/name) for name in ('protocol.json','results.json','samples.csv')},
        'prepared_utc':datetime.now(timezone.utc).isoformat(),'git_commit':_git_commit(),'provenance':capture_provenance(PROJECT_ROOT,out)})


def load_training_protocol(path: Path):
    """Reject changed data/normalization/settings/history or copied fit protocols."""
    p,digest=_read_protocol(path);collection=Path(p['collection_dir'])
    if any(sha256(collection/name)!=h for name,h in p['collection_sha256'].items()):raise ValueError('coverage collection bytes differ')
    data=load_training_data(collection)
    if any(p.get(k)!=v for k,v in training_definition(data).items()):raise ValueError('coverage candidate/settings differ')
    return p,digest,data


def run_training(path: Path) -> dict:
    """Consume one covered-state fit and record unchanged original behavior gates."""
    path=Path(path).resolve();out=path.parent
    if any((out/name).exists() for name in ('execution.json','results.json','trained.pt','untrained.pt','curve.csv','episodes.csv')):
        raise FileExistsError('coverage fit consumed; no retry')
    p,digest,data=load_training_protocol(path)
    save_new_json(out/'execution.json',{'protocol_sha256':digest,'status':'consumed','started_utc':datetime.now(timezone.utc).isoformat()})
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:return _execute(path,p,digest,data)
    except Exception as error:
        save_new_json(out/'aborted.json',{'type':type(error).__name__,'error':str(error),'protocol_sha256':digest});raise
    finally:torch.set_num_threads(previous_threads)


def _execute(path: Path,p: dict,digest: str,data: TrainingDataset) -> dict:
    out=path.parent;mean=torch.tensor(p['normalization']['mean']);scale=torch.tensor(p['normalization']['scale'])
    model,untrained=[MemoryBancReadout(mean,scale,seed=1010) for _ in range(2)]
    save_checkpoint(out/'untrained.pt',untrained,protocol_sha256=digest)
    fit=fit_readout(model,data.features,data.targets)
    r={'experiment':p['experiment'],'protocol_sha256':digest,'software':software_record(),'git_commit':p['git_commit'],
        'seed':1010,'network_size':83,'learning_rate':.01,'objective':p['objective'],'sample_count':len(data.features),
        **fit,'training_gate_passed':False,'held_out_evaluated':False,'checkpoint':None,
        'actual_world_ticks':0,'actual_neural_steps':0,'actual_episodes':0}
    if fit['curve']:_write_csv(out/'curve.csv',fit['curve'])
    if fit['failure'] is None:
        save_checkpoint(out/'trained.pt',model,protocol_sha256=digest)
        loaded=load_checkpoint(out/'trained.pt',protocol_sha256=digest,mean=mean,scale=scale)
        original=original_data()
        with torch.no_grad():
            errors=(model(data.features).double()-data.targets).square().mean(0)
            original_errors=(model(original.features).double()-original.targets).square().mean(0)
            exact=torch.equal(model(data.features),loaded(data.features))
        b=evaluate_training_models(model,loaded,untrained,locked_cases()['train'],EvaluationConfig());r.update(b)
        checks={'checkpoint_outputs_exact':exact,'checkpoint_replay_exact':b['checkpoint_replay_exact'],
            'effective_weights_unchanged':b['effective_weights_unchanged'],'execution_budget_passed':b['execution_budget_passed']
                and b['actual_world_ticks']<=15000 and b['actual_neural_steps']<=300000 and b['actual_episodes']==25,
            'optimizer_budget_passed':fit['optimizer_steps']==500,'gradients_and_updates_observed':fit['gradient_nonzero_scalars']>0 and fit['parameter_changed_scalars']>0,
            'normalization_unchanged':torch.equal(model.mean,mean) and torch.equal(model.scale,scale),
            'provenance_preserved':verify_provenance(p['provenance']),'protocol_unchanged':sha256(path)==digest}
        gates=coverage_gates(float(errors[1]),float(original_errors[1]),p['train_only_baseline']['constant_mse'][1],b['summaries'],checks)
        r.update({'checks':checks,'gates':gates,'per_output_mse':errors.tolist(),'original_per_output_mse':original_errors.tolist(),
            'constant_command_mse':p['train_only_baseline']['constant_mse'],'training_gate_passed':gates['passed'],
            'checkpoint':'trained.pt','checkpoint_sha256':sha256(out/'trained.pt')})
        _write_csv(out/'episodes.csv',[dict(condition=name,**row) for name,rows in b['episodes'].items() for row in rows]+[dict(condition='trained_replay',**b['replay_episode'])])
    else:r['checks']={'provenance_preserved':verify_provenance(p['provenance']),'protocol_unchanged':sha256(path)==digest}
    save_new_json(out/'results.json',r);return r


def main() -> int:
    """Separately prepare/consume collection and the bounded fit."""
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare-collection','collect','prepare-training','fit'):modes.add_argument('--'+mode,action='store_true')
    parser.add_argument('--protocol',type=Path);parser.add_argument('--output-dir',type=Path);args=parser.parse_args()
    if args.prepare_collection:
        path=prepare_collection(args.output_dir or DEFAULT_COLLECTION);print(json.dumps({'protocol':str(path),'sha256':sha256(path),'updates':0,'ticks':0}));return 0
    if args.prepare_training:
        path=prepare_training(args.output_dir or DEFAULT_OUTPUT);print(json.dumps({'protocol':str(path),'sha256':sha256(path),'updates':0,'ticks':0}));return 0
    if args.collect:
        r=collect(args.protocol or DEFAULT_COLLECTION/'protocol.json');print(json.dumps({k:r[k] for k in ('readiness_passed','sample_count','actual_world_ticks','optimizer_steps')},indent=2));return 0 if r['readiness_passed'] else 1
    r=run_training(args.protocol or DEFAULT_OUTPUT/'protocol.json');print(json.dumps({k:r[k] for k in ('training_gate_passed','optimizer_steps','actual_world_ticks')},indent=2));return 0 if r['training_gate_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
