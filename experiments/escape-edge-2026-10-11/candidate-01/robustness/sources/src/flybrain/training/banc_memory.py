"""One fixed dual-timescale repair fit, separate from fresh evaluation."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json

import torch

from flybrain.neural.banc_memory import (MemoryBancReadout, memory_contract,
    normalization, refilter_rates, make_memory_controller, save_checkpoint, load_checkpoint)
from flybrain.neural.banc_motor import make_banc_controller
from flybrain.neural.banc_network import load_verified_banc_config
from .banc_conditioning import protocol_definition as previous_definition
from .banc_protocol import (DEFAULT_READINESS,SEED,MAX_UPDATES,LEARNING_RATE,
                            load_readiness,sha256,software_record)
from .banc_readiness import (PROJECT_ROOT,capture_provenance,verify_provenance,
                             save_new_json,weights_sha256)
from .banc_training import fit_readout,training_gates,_write_csv
from .evaluation import EpisodeCase,EvaluationConfig,locked_cases,run_episode,summarize,_git_commit
from .readout_experiment import TrainingDataset
from .records import ensure_new_run_directory


DEFAULT_OUTPUT = Path('experiments/fb10-memory-readout')
MEMORY_READINESS = PROJECT_ROOT/'experiments/fb10-memory-readiness'


def load_training_data() -> TrainingDataset:
    """Reconstruct the declared slow history from the original training counts."""
    original = load_readiness(DEFAULT_READINESS)
    with (DEFAULT_READINESS/'samples.csv').open(newline='',encoding='utf-8') as stream:
        ids = [row['case_id'] for row in csv.DictReader(stream)]
    slow = refilter_rates(original.features,ids)
    with (MEMORY_READINESS/'samples.csv').open(newline='',encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    recorded = torch.tensor([[float(row['feature_'+str(i)]) for i in range(8)] for row in rows],dtype=torch.float64)
    targets = torch.tensor([[float(row['target_forward']),float(row['target_turn'])] for row in rows],dtype=torch.float64)
    if ([r['case_id'] for r in rows] != ids or not torch.equal(recorded,slow)
            or not torch.equal(targets,original.targets)):
        raise ValueError('temporal readiness differs from original counted-spike reconstruction')
    return TrainingDataset(torch.cat((original.features,slow),dim=1),original.targets,original.case_counts)


def protocol_definition(data: TrainingDataset) -> dict:
    """One measured repair; retain all original turn/behavior/integrity gates."""
    base = previous_definition(load_readiness(DEFAULT_READINESS))
    mean,scale = normalization(data.features)
    return {**base,'schema':'fb10_memory_training_v1','experiment':'fb10_memory_readout',
        'hypothesis':'Fast and slow actual endpoint spike histories reduce counting noise while retaining response speed, allowing the readout to pass the original scientific gates.',
        'contract':memory_contract(),'trainable_parameters':306,
        'normalization':{'mean':mean.tolist(),'scale':scale.tolist()},
        'objective':'forward MSE + turn MSE; fixed spike circuit and fixed causal endpoint filters'}


def prepare_protocol(output_dir: Path) -> Path:
    """Freeze source/data/normalization/cases/budgets before any fitting."""
    out = Path(output_dir).resolve();ensure_new_run_directory(out)
    data = load_training_data()
    p = {**protocol_definition(data),'output_dir':str(out),'git_commit':_git_commit(),
        'prepared_utc':datetime.now(timezone.utc).isoformat(),
        'provenance':capture_provenance(PROJECT_ROOT,out),
        'data_sha256':{str(directory/name):sha256(directory/name)
            for directory in (DEFAULT_READINESS,MEMORY_READINESS)
            for name in ('protocol.json','samples.csv','results.json')}}
    out.mkdir(parents=True);path=out/'protocol.json';save_new_json(path,p)
    with path.with_suffix('.sha256').open('x',encoding='utf-8') as stream:stream.write(sha256(path)+'\n')
    return path


def load_protocol(path: Path):
    """Reject changed settings, destination, data, core, history or environment."""
    path=Path(path).resolve();digest=sha256(path)
    if path.with_suffix('.sha256').read_text(encoding='utf-8').strip()!=digest:
        raise ValueError('memory protocol hash differs')
    p=json.loads(path.read_text(encoding='utf-8'))
    if path!=Path(p['output_dir']).resolve()/'protocol.json':raise ValueError('protocol destination differs')
    if any(sha256(Path(name))!=h for name,h in p['data_sha256'].items()):raise ValueError('training data differs')
    data=load_training_data()
    if any(p.get(k)!=v for k,v in protocol_definition(data).items()):raise ValueError('candidate/settings differ')
    if not verify_provenance(p['provenance']):raise ValueError('source/core/history differs')
    return p,digest,data


def evaluate_training_models(trained, reloaded, untrained, cases, config) -> dict:
    """Three matched training controls plus one exact checkpoint replay."""
    allowed=set(locked_cases()['train'])
    if not cases or len(set(cases))!=len(cases) or any(c not in allowed for c in cases):
        raise ValueError('expected original training cases')
    controllers=[];digests=[]
    def factory(model):
        def create():
            c=make_memory_controller(model) if model is not None else make_banc_controller(load_verified_banc_config())
            controllers.append(c);digests.append(weights_sha256(c));return c
        return create
    with torch.no_grad():
        episodes={name:[run_episode(c,config,factory(m)) for c in cases]
                  for name,m in (('trained',reloaded),('untrained',untrained),('fixed',None))}
        replay=run_episode(cases[0],config,factory(trained))
    ticks=sum(row.ticks for rows in episodes.values() for row in rows)+replay.ticks
    steps=sum(c.network.population.step_count for c in controllers)
    return {'episodes':{name:[asdict(row) for row in rows] for name,rows in episodes.items()},
        'summaries':{name:summarize(rows) for name,rows in episodes.items()},
        'replay_episode':asdict(replay),'actual_world_ticks':ticks,'actual_neural_steps':steps,
        'actual_episodes':len(controllers),'checkpoint_replay_exact':replay==episodes['trained'][0],
        'effective_weights_unchanged':len(set(digests))==1 and all(weights_sha256(c)==h for c,h in zip(controllers,digests)),
        'execution_budget_passed':len(controllers)==3*len(cases)+1 and ticks<=(3*len(cases)+1)*config.horizon_ticks and steps==ticks*20}


def run_training(path: Path) -> dict:
    """Consume one repair protocol and preserve partial/completed attempts."""
    path=Path(path).resolve();out=path.parent
    if any((out/name).exists() for name in ('execution.json','results.json','trained.pt','untrained.pt','curve.csv','episodes.csv')):
        raise FileExistsError('memory attempt consumed; no retry')
    p,digest,data=load_protocol(path)
    save_new_json(out/'execution.json',{'protocol_sha256':digest,'status':'consumed',
        'started_utc':datetime.now(timezone.utc).isoformat(),'stop_rule':p['stop_condition']})
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:return _execute(path,p,digest,data)
    except Exception as error:
        save_new_json(out/'aborted.json',{'type':type(error).__name__,'error':str(error),'protocol_sha256':digest})
        raise
    finally:torch.set_num_threads(previous_threads)


def _execute(path: Path,p: dict,digest: str,data: TrainingDataset) -> dict:
    out=path.parent;mean=torch.tensor(p['normalization']['mean']);scale=torch.tensor(p['normalization']['scale'])
    model,untrained=[MemoryBancReadout(mean,scale,seed=SEED) for _ in range(2)]
    save_checkpoint(out/'untrained.pt',untrained,protocol_sha256=digest)
    fit=fit_readout(model,data.features,data.targets)
    r={'experiment':p['experiment'],'protocol_sha256':digest,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'software':software_record(),'git_commit':p['git_commit'],'network_size':83,'seed':SEED,
        'objective':p['objective'],'learning_rate':LEARNING_RATE,'sample_count':len(data.features),
        **fit,'training_gate_passed':False,'held_out_evaluated':False,'checkpoint':None,
        'actual_world_ticks':0,'actual_neural_steps':0,'actual_episodes':0}
    if fit['curve']:_write_csv(out/'curve.csv',fit['curve'])
    if fit['failure'] is None:
        save_checkpoint(out/'trained.pt',model,protocol_sha256=digest)
        loaded=load_checkpoint(out/'trained.pt',protocol_sha256=digest,mean=mean,scale=scale)
        with torch.no_grad():
            errors=(model(data.features).double()-data.targets).square().mean(0)
            exact=torch.equal(model(data.features),loaded(data.features))
        b=evaluate_training_models(model,loaded,untrained,tuple(EpisodeCase(**c) for c in p['cases']),EvaluationConfig(**p['world_config']))
        r.update(b)
        checks={'checkpoint_outputs_exact':exact,'checkpoint_replay_exact':b['checkpoint_replay_exact'],
            'effective_weights_unchanged':b['effective_weights_unchanged'],
            'execution_budget_passed':b['execution_budget_passed'] and b['actual_world_ticks']<=p['max_world_ticks'] and b['actual_neural_steps']<=p['max_neural_steps'] and b['actual_episodes']==p['max_episodes'],
            'optimizer_budget_passed':fit['optimizer_steps']==MAX_UPDATES,
            'gradients_and_updates_observed':fit['gradient_nonzero_scalars']>0 and fit['parameter_changed_scalars']>0,
            'normalization_unchanged':torch.equal(model.mean,mean) and torch.equal(model.scale,scale),
            'provenance_preserved':verify_provenance(p['provenance']),'protocol_unchanged':sha256(path)==digest}
        gates=training_gates(float(errors[1]),p['train_only_baseline']['constant_mse'][1],b['summaries'],checks)
        r.update({'checks':checks,'gates':gates,'per_output_mse':errors.tolist(),
            'constant_command_mse':p['train_only_baseline']['constant_mse'],'training_gate_passed':gates['passed'],
            'checkpoint':'trained.pt','checkpoint_sha256':sha256(out/'trained.pt')})
        _write_csv(out/'episodes.csv',[dict(condition=name,**row) for name,rows in b['episodes'].items() for row in rows]+[dict(condition='trained_replay',**b['replay_episode'])])
    else:r['checks']={'provenance_preserved':verify_provenance(p['provenance']),'protocol_unchanged':sha256(path)==digest}
    save_new_json(out/'results.json',r);return r


def main() -> int:
    """Prepare with no scientific stepping, or execute the one consumed fit."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--protocol',type=Path,default=DEFAULT_OUTPUT/'protocol.json');args=parser.parse_args()
    if args.prepare:
        path=prepare_protocol(args.output_dir);print(json.dumps({'protocol':str(path),'sha256':sha256(path),'updates':0,'ticks':0}));return 0
    r=run_training(args.protocol);print(json.dumps({k:r[k] for k in ('optimizer_steps','failure','training_gate_passed','actual_world_ticks')},indent=2))
    return 0 if r['training_gate_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
