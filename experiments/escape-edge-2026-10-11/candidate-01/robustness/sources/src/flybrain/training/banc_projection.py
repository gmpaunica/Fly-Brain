"""One precision repair: replay covered data, fit only a small neural readout."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path

import torch

from flybrain.neural.banc_projection import (ProjectionReadout,make_projection_controller,
    projection_contract,normalization,save_checkpoint,load_checkpoint)
from flybrain.neural.banc_motor import make_banc_controller
from flybrain.neural.banc_network import load_verified_banc_config
from flybrain.neural.controller import NeuralController,NeuralDecision
from flybrain.world.fly import MotorCommand
from flybrain.world.sensing import SensoryObservation
from .banc_coverage import (coverage_cases,DEFAULT_COLLECTION as OLD_COLLECTION,
    load_training_data as old_data,training_definition as old_definition,coverage_gates,
    _write_protocol,_read_protocol)
from .banc_memory_evaluation import training_evidence as old_student
from .banc_protocol import DEFAULT_READINESS,sha256,software_record
from .banc_readiness import PROJECT_ROOT,capture_provenance,verify_provenance,save_new_json,weights_sha256
from .banc_training import fit_readout,_write_csv
from .evaluation import EpisodeCase,EvaluationConfig,run_episode,summarize,locked_cases,_git_commit
from .readout_experiment import TrainingDataset,teacher_command
from .records import ensure_new_run_directory


DEFAULT_COLLECTION=PROJECT_ROOT/'experiments/fb10-projection-data'
DEFAULT_OUTPUT=PROJECT_ROOT/'experiments/fb10-projection-readout'


class ProjectionRecorder(NeuralController):
    """Log new spike features while preserving the old frozen body policy exactly."""

    def __init__(self,old_model,case_id: str,trajectory: str) -> None:
        if trajectory not in ('teacher','student'):raise ValueError('teacher/student required')
        shadow=ProjectionReadout(torch.zeros(20),torch.ones(20));shadow.requires_grad_(False)
        c=make_projection_controller(shadow);super().__init__(c.network,c.decoder)
        self.old_model=old_model;self.case_id=case_id;self.trajectory=trajectory;self.rows=[];self.tick=0

    def reset(self) -> None:
        super().reset();self.rows=[];self.tick=0

    def decide(self,observation: SensoryObservation) -> NeuralDecision:
        decision=super().decide(observation);target=teacher_command(observation)
        rates=self.decoder.feature_rates
        self.rows.append({'case_id':self.case_id,'trajectory':self.trajectory,'tick':self.tick,
            'rates':rates.tolist(),'target_forward':float(target[0]),'target_turn':float(target[1])})
        self.tick+=1
        with torch.no_grad():body=target if self.trajectory=='teacher' else self.old_model(rates[:16])
        return NeuralDecision(decision.activity,MotorCommand(float(body[0]),float(body[1])))


def collection_definition() -> dict:
    """Pin identical original/coverage data and exact body replays, zero fitting."""
    p,digest,_=old_student();old_data()
    return {'schema':'fb10_projection_collection_v1','experiment':'fb10_projection_collection',
        'hypothesis':'Actual PN incoming spikes preserve steering precision lost by endpoint spike counting, without changing any training trajectory or the core.',
        'frozen_student_protocol_sha256':digest,'contract':projection_contract(),'software':software_record(),
        'world_config':p['world_config'],'cases':[asdict(c) for c in coverage_cases()],
        'trajectory_order':['teacher','student'],'expected_sample_count':15864,
        'original_neural_intervals':1008,'max_episodes':64,'max_world_ticks':14856,'max_neural_steps':317280,'optimizer_steps':0,
        'old_collection_sha256':{name:sha256(OLD_COLLECTION/name) for name in ('protocol.json','results.json','samples.csv')},
        'gates':['bit-exact old endpoint features and sensory labels','identical all 64 body outcomes','unchanged effective circuit, frozen model and history'],
        'stop_condition':'One exact replay of predetermined old training data; no different starts, selections, optimizer updates or held-out episodes.'}


def prepare_collection(output_dir: Path) -> Path:
    """Freeze replay scope before any world or neural stepping."""
    out=Path(output_dir).resolve();ensure_new_run_directory(out)
    return _write_protocol(out,{**collection_definition(),'output_dir':str(out),
        'prepared_utc':datetime.now(timezone.utc).isoformat(),'provenance':capture_provenance(PROJECT_ROOT,out)})


def load_collection(path: Path):
    p,digest=_read_protocol(path)
    if any(p.get(k)!=v for k,v in collection_definition().items()):raise ValueError('projection collection settings differ')
    return p,digest


def collect(path: Path) -> dict:
    """Consume replay once; assert every historical training datum/outcome exact."""
    path=Path(path).resolve();out=path.parent
    if any((out/n).exists() for n in ('execution.json','results.json','samples.csv','episodes.jsonl')):
        raise FileExistsError('projection collection consumed')
    p,digest=load_collection(path);_,_,models=old_student();model=models['trained']
    initial={k:v.clone() for k,v in model.state_dict().items()}
    save_new_json(out/'execution.json',{'status':'consumed','protocol_sha256':digest,'started_utc':datetime.now(timezone.utc).isoformat()})
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    rows=[];episodes={};ticks=0;steps=0;controllers=[];weights=[]
    try:
        c=make_projection_controller(ProjectionReadout(torch.zeros(20),torch.ones(20)))
        weights.append(weights_sha256(c));controllers.append(c);current=None
        with (DEFAULT_READINESS/'samples.csv').open(newline='',encoding='utf-8') as stream:
            for record in csv.DictReader(stream):
                if record['case_id']!=current:c.reset();current=record['case_id']
                c.decide(SensoryObservation(float(record['left_odor']),float(record['right_odor'])))
                rows.append({'case_id':current,'trajectory':'original-teacher','tick':int(record['tick']),
                    'rates':c.decoder.feature_rates.tolist(),'target_forward':float(record['target_forward']),
                    'target_turn':float(record['target_turn'])})
                steps+=20
        with (out/'episodes.jsonl').open('x',encoding='utf-8') as journal:
            for trajectory in p['trajectory_order']:
                group=[]
                for i,case in enumerate(coverage_cases(),1):
                    c=ProjectionRecorder(model,case.case_id,trajectory);controllers.append(c);weights.append(weights_sha256(c))
                    with torch.no_grad():result=run_episode(case,EvaluationConfig(**p['world_config']),lambda:c)
                    group.append(asdict(result));rows.extend(c.rows);ticks+=result.ticks;steps+=c.network.population.step_count
                    journal.write(json.dumps({'trajectory':trajectory,'result':asdict(result)})+'\n');journal.flush()
                    if ticks>p['max_world_ticks'] or steps>p['max_neural_steps']:raise RuntimeError('projection replay budget exceeded')
                    if i%8==0:print(f'{trajectory}: {i}/32',flush=True)
                episodes[trajectory]=group
        x=torch.tensor([r['rates'] for r in rows],dtype=torch.float64)
        y=torch.tensor([[r['target_forward'],r['target_turn']] for r in rows],dtype=torch.float64)
        old=old_data();old_results=json.loads((OLD_COLLECTION/'results.json').read_text(encoding='utf-8'))
        with (OLD_COLLECTION/'samples.csv').open(newline='',encoding='utf-8') as stream:old_rows=list(csv.DictReader(stream))
        identities_exact=all((r['case_id'],r['trajectory'],r['tick'])==(b['case_id'],b['trajectory'],int(b['tick'])) for r,b in zip(rows[1008:],old_rows))
        checks={'old_features_exact':torch.equal(x[:,:16],old.features),'all_labels_exact':torch.equal(y,old.targets),
            'all_64_outcomes_exact':episodes==old_results['episodes'],'all_row_identities_exact':identities_exact,
            'fixed_student_parameters':all(torch.equal(v,initial[k]) for k,v in model.state_dict().items()),
            'effective_weights_unchanged':len(set(weights))==1 and all(weights_sha256(c)==h for c,h in zip(controllers,weights)),
            'budget_and_samples':len(rows)==15864 and ticks==14856 and steps==317280 and len(controllers)==65,
            'finite_neural_features':bool(torch.isfinite(x).all() and torch.all(x>=0) and torch.all(x<=1000)),
            'provenance_preserved':verify_provenance(p['provenance']),'protocol_unchanged':sha256(path)==digest}
        fields=['case_id','trajectory','tick',*[f'feature_{i}' for i in range(20)],'target_forward','target_turn']
        with (out/'samples.csv').open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for r in rows:writer.writerow({k:v for k,v in r.items() if k!='rates'}|{f'feature_{i}':v for i,v in enumerate(r['rates'])})
        r={'experiment':p['experiment'],'protocol_sha256':digest,'software':software_record(),
            'optimizer_steps':0,'actual_episodes':64,'actual_world_ticks':ticks,'actual_neural_steps':steps,
            'file_only_neural_steps':20160,'sample_count':len(rows),'samples_sha256':sha256(out/'samples.csv'),
            'episodes':episodes,'checks':checks,'readiness_passed':all(checks.values())}
        save_new_json(out/'results.json',r);return r
    except Exception as e:
        save_new_json(out/'aborted.json',{'type':type(e).__name__,'error':str(e),'actual_world_ticks':ticks,'actual_neural_steps':steps});raise
    finally:torch.set_num_threads(previous_threads)


def load_training_data() -> TrainingDataset:
    """Validated [15864,20] spike-derived data, with exact earlier rows/labels."""
    p,digest=load_collection(DEFAULT_COLLECTION/'protocol.json')
    r=json.loads((DEFAULT_COLLECTION/'results.json').read_text(encoding='utf-8'))
    if r.get('readiness_passed') is not True or r['protocol_sha256']!=digest or sha256(DEFAULT_COLLECTION/'samples.csv')!=r['samples_sha256']:
        raise ValueError('projection readiness missing/changed')
    with (DEFAULT_COLLECTION/'samples.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    old=old_data();counts={}
    for row in rows:
        key=row['case_id']+':'+row['trajectory'];tick=counts.get(key,0)
        if int(row['tick'])!=tick:raise ValueError('projection sample ticks differ')
        counts[key]=tick+1
    x=torch.tensor([[float(row[f'feature_{i}']) for i in range(20)] for row in rows],dtype=torch.float64)
    y=torch.tensor([[float(row['target_forward']),float(row['target_turn'])] for row in rows],dtype=torch.float64)
    if counts!=old.case_counts or not torch.equal(x[:,:16],old.features) or not torch.equal(y,old.targets):
        raise ValueError('projection samples differ from all declared old data')
    normalization(x)
    return TrainingDataset(x,y,counts)


def training_definition(data) -> dict:
    """Retain strict covered/original gates; change only observed neural precision."""
    base=old_definition(old_data());mean,scale=normalization(data.features)
    return {**base,'schema':'fb10_projection_training_v1','experiment':'fb10_projection_readout',
        'hypothesis':'A small head observing precise internal PN spike events plus endpoint histories passes unchanged error and navigation gates on covered training states.',
        'contract':projection_contract(),'trainable_parameters':370,
        'normalization':{'mean':mean.tolist(),'scale':scale.tolist()}}


def prepare_training(output_dir: Path) -> Path:
    out=Path(output_dir).resolve();ensure_new_run_directory(out);data=load_training_data()
    return _write_protocol(out,{**training_definition(data),'output_dir':str(out),'prepared_utc':datetime.now(timezone.utc).isoformat(),
        'git_commit':_git_commit(),'provenance':capture_provenance(PROJECT_ROOT,out),
        'collection_sha256':{n:sha256(DEFAULT_COLLECTION/n) for n in ('protocol.json','results.json','samples.csv')}})


def load_training_protocol(path: Path):
    p,digest=_read_protocol(path)
    if any(sha256(DEFAULT_COLLECTION/n)!=h for n,h in p['collection_sha256'].items()):raise ValueError('projection data bytes differ')
    data=load_training_data()
    if any(p.get(k)!=v for k,v in training_definition(data).items()):raise ValueError('projection fit settings differ')
    return p,digest,data


def evaluate_group(cases,models,config,on_episode=None):
    """Measure three independently reset matched controls, never optimizer steps."""
    if not cases or set(models)!={'trained','untrained'}:raise ValueError('cases and matched models required')
    episodes={};controllers=[];weights=[]
    with torch.no_grad():
        for name in ('trained','untrained','fixed'):
            group=[]
            for case in cases:
                c=make_banc_controller(load_verified_banc_config()) if name=='fixed' else make_projection_controller(models[name])
                controllers.append(c);weights.append(weights_sha256(c));row=run_episode(case,config,lambda:c)
                group.append(row)
                if on_episode is not None:on_episode(name,row)
            episodes[name]=group
    ticks=sum(r.ticks for group in episodes.values() for r in group);steps=sum(c.network.population.step_count for c in controllers)
    return episodes,{'episodes':len(controllers),'world_ticks':ticks,'neural_steps':steps,
        'effective_weights_unchanged':len(set(weights))==1 and all(weights_sha256(c)==h for c,h in zip(controllers,weights))}


def evaluate_models(trained,reloaded,untrained,cases,config) -> dict:
    """Original eight training starts plus one exact saved-model replay."""
    if not cases or len(set(cases))!=len(cases) or any(c not in locked_cases()['train'] for c in cases):raise ValueError('original training cases required')
    episodes,a=evaluate_group(cases,{'trained':reloaded,'untrained':untrained},config)
    c=make_projection_controller(trained);weight=weights_sha256(c)
    with torch.no_grad():replay=run_episode(cases[0],config,lambda:c)
    ticks=a['world_ticks']+replay.ticks;steps=a['neural_steps']+c.network.population.step_count
    return {'episodes':{n:[asdict(r) for r in group] for n,group in episodes.items()},
        'summaries':{n:summarize(group) for n,group in episodes.items()},'replay_episode':asdict(replay),
        'actual_episodes':a['episodes']+1,'actual_world_ticks':ticks,'actual_neural_steps':steps,
        'checkpoint_replay_exact':replay==episodes['trained'][0],
        'effective_weights_unchanged':a['effective_weights_unchanged'] and weights_sha256(c)==weight,
        'execution_budget_passed':a['episodes']+1==3*len(cases)+1 and ticks<=(3*len(cases)+1)*config.horizon_ticks and steps==20*ticks}


def run_training(path: Path) -> dict:
    """Consume exactly one 500-update head fit; preserve negative/partial evidence."""
    path=Path(path).resolve();out=path.parent
    if any((out/n).exists() for n in ('execution.json','results.json','trained.pt','untrained.pt','curve.csv','episodes.csv')):
        raise FileExistsError('projection fit consumed')
    p,digest,data=load_training_protocol(path)
    save_new_json(out/'execution.json',{'status':'consumed','protocol_sha256':digest,'started_utc':datetime.now(timezone.utc).isoformat()})
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        mean=torch.tensor(p['normalization']['mean']);scale=torch.tensor(p['normalization']['scale'])
        trained,untrained=[ProjectionReadout(mean,scale) for _ in range(2)]
        save_checkpoint(out/'untrained.pt',untrained,protocol_sha256=digest)
        fit=fit_readout(trained,data.features,data.targets)
        r={'experiment':p['experiment'],'protocol_sha256':digest,'software':software_record(),'git_commit':p['git_commit'],
            'seed':1010,'network_size':83,'learning_rate':.01,'objective':p['objective'],'sample_count':len(data.features),
            **fit,'training_gate_passed':False,'held_out_evaluated':False,'checkpoint':None,
            'actual_world_ticks':0,'actual_neural_steps':0,'actual_episodes':0}
        if fit['curve']:_write_csv(out/'curve.csv',fit['curve'])
        if fit['failure'] is None:
            save_checkpoint(out/'trained.pt',trained,protocol_sha256=digest)
            loaded=load_checkpoint(out/'trained.pt',protocol_sha256=digest,mean=mean,scale=scale)
            with torch.no_grad():
                error=(trained(data.features).double()-data.targets).square()
                errors=error.mean(0);original_errors=error[:1008].mean(0)
                exact=torch.equal(trained(data.features),loaded(data.features))
            b=evaluate_models(trained,loaded,untrained,locked_cases()['train'],EvaluationConfig());r.update(b)
            checks={'checkpoint_outputs_exact':exact,'checkpoint_replay_exact':b['checkpoint_replay_exact'],
                'effective_weights_unchanged':b['effective_weights_unchanged'],'execution_budget_passed':b['execution_budget_passed'] and b['actual_episodes']==25 and b['actual_world_ticks']<=15000 and b['actual_neural_steps']<=300000,
                'optimizer_budget_passed':fit['optimizer_steps']==500,
                'gradients_and_updates_observed':fit['gradient_nonzero_scalars']>0 and fit['parameter_changed_scalars']>0,
                'normalization_unchanged':torch.equal(trained.mean,mean) and torch.equal(trained.scale,scale),
                'provenance_preserved':verify_provenance(p['provenance']),'protocol_unchanged':sha256(path)==digest}
            gates=coverage_gates(float(errors[1]),float(original_errors[1]),p['train_only_baseline']['constant_mse'][1],b['summaries'],checks)
            r.update({'checks':checks,'gates':gates,'per_output_mse':errors.tolist(),'original_per_output_mse':original_errors.tolist(),
                'constant_command_mse':p['train_only_baseline']['constant_mse'],'training_gate_passed':gates['passed'],
                'checkpoint':'trained.pt','checkpoint_sha256':sha256(out/'trained.pt')})
            _write_csv(out/'episodes.csv',[dict(condition=n,**row) for n,group in b['episodes'].items() for row in group]+[dict(condition='trained_replay',**b['replay_episode'])])
        save_new_json(out/'results.json',r);return r
    except Exception as e:
        save_new_json(out/'aborted.json',{'type':type(e).__name__,'error':str(e),'protocol_sha256':digest});raise
    finally:torch.set_num_threads(previous_threads)


def main() -> int:
    """Preparation and scientific execution are separate, each consumed once."""
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare-collection','collect','prepare-training','fit'):modes.add_argument('--'+mode,action='store_true')
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--protocol',type=Path);a=parser.parse_args()
    if a.prepare_collection or a.prepare_training:
        path=prepare_collection(a.output_dir or DEFAULT_COLLECTION) if a.prepare_collection else prepare_training(a.output_dir or DEFAULT_OUTPUT)
        print(json.dumps({'protocol':str(path),'sha256':sha256(path),'updates':0,'ticks':0}));return 0
    r=collect(a.protocol or DEFAULT_COLLECTION/'protocol.json') if a.collect else run_training(a.protocol or DEFAULT_OUTPUT/'protocol.json')
    gate='readiness_passed' if a.collect else 'training_gate_passed'
    print(json.dumps({k:r[k] for k in (gate,'optimizer_steps','actual_world_ticks','actual_neural_steps')},indent=2));return 0 if r[gate] else 1


if __name__=='__main__':raise SystemExit(main())
