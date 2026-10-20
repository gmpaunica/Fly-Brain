"""One 64-unit head fit on the unchanged, predeclared covered training data."""

import argparse
from datetime import datetime,timezone
from pathlib import Path
import json

import torch

from flybrain.neural.banc_capacity import WideMemoryReadout,capacity_contract,save_checkpoint,load_checkpoint
from .banc_coverage import (DEFAULT_COLLECTION,load_training_data,training_definition as coverage_definition,
    coverage_gates,_read_protocol,_write_protocol)
from .banc_memory import load_training_data as original_data,evaluate_training_models
from .banc_protocol import sha256,software_record
from .banc_readiness import PROJECT_ROOT,capture_provenance,verify_provenance,save_new_json
from .banc_training import fit_readout,_write_csv
from .evaluation import locked_cases,EvaluationConfig,_git_commit
from .records import ensure_new_run_directory


DEFAULT_OUTPUT=PROJECT_ROOT/'experiments/fb10-capacity-readout'


def protocol_definition(data) -> dict:
    """Change only capacity; retain all covered-data budgets and scientific gates."""
    return {**coverage_definition(data),'schema':'fb10_capacity_training_v1','experiment':'fb10_capacity_readout',
        'hypothesis':'A 64-unit readout has sufficient nonlinear capacity for the covered temporal states while preserving all original and aggregate gates.',
        'contract':capacity_contract(),'trainable_parameters':1218}


def prepare_protocol(output_dir: Path) -> Path:
    """Freeze the single capacity candidate before optimization or body stepping."""
    out=Path(output_dir).resolve();ensure_new_run_directory(out);data=load_training_data()
    return _write_protocol(out,{**protocol_definition(data),'output_dir':str(out),
        'collection_sha256':{name:sha256(DEFAULT_COLLECTION/name) for name in ('protocol.json','results.json','samples.csv')},
        'prepared_utc':datetime.now(timezone.utc).isoformat(),'git_commit':_git_commit(),'provenance':capture_provenance(PROJECT_ROOT,out)})


def load_protocol(path: Path):
    """Reject changed/resealed data, candidate, gates, source or destination."""
    p,digest=_read_protocol(path)
    if any(sha256(DEFAULT_COLLECTION/name)!=h for name,h in p['collection_sha256'].items()):raise ValueError('covered data bytes differ')
    data=load_training_data()
    if any(p.get(k)!=v for k,v in protocol_definition(data).items()):raise ValueError('capacity settings differ')
    return p,digest,data


def run_training(path: Path) -> dict:
    """Consume one attempt; preserve all partial and scientifically failed results."""
    path=Path(path).resolve();out=path.parent
    if any((out/name).exists() for name in ('execution.json','results.json','trained.pt','untrained.pt','curve.csv','episodes.csv')):
        raise FileExistsError('capacity attempt consumed; no retry')
    p,digest,data=load_protocol(path)
    save_new_json(out/'execution.json',{'protocol_sha256':digest,'status':'consumed','started_utc':datetime.now(timezone.utc).isoformat()})
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:return _execute(path,p,digest,data)
    except Exception as error:
        save_new_json(out/'aborted.json',{'type':type(error).__name__,'error':str(error),'protocol_sha256':digest});raise
    finally:torch.set_num_threads(previous_threads)


def _execute(path: Path,p: dict,digest: str,data) -> dict:
    out=path.parent;mean=torch.tensor(p['normalization']['mean']);scale=torch.tensor(p['normalization']['scale'])
    model,untrained=[WideMemoryReadout(mean,scale,seed=1010) for _ in range(2)]
    save_checkpoint(out/'untrained.pt',untrained,protocol_sha256=digest)
    fit=fit_readout(model,data.features,data.targets)
    r={'experiment':p['experiment'],'protocol_sha256':digest,'software':software_record(),'git_commit':p['git_commit'],
        'seed':1010,'network_size':83,'learning_rate':.01,'objective':p['objective'],'sample_count':len(data.features),
        **fit,'training_gate_passed':False,'held_out_evaluated':False,'checkpoint':None,
        'actual_world_ticks':0,'actual_neural_steps':0,'actual_episodes':0}
    if fit['curve']:_write_csv(out/'curve.csv',fit['curve'])
    if fit['failure'] is None:
        save_checkpoint(out/'trained.pt',model,protocol_sha256=digest)
        loaded=load_checkpoint(out/'trained.pt',protocol_sha256=digest,mean=mean,scale=scale);original=original_data()
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
    """Prepare or separately execute the single frozen head-capacity trial."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--protocol',type=Path,default=DEFAULT_OUTPUT/'protocol.json');args=parser.parse_args()
    if args.prepare:
        path=prepare_protocol(args.output_dir);print(json.dumps({'protocol':str(path),'sha256':sha256(path),'updates':0,'ticks':0}));return 0
    r=run_training(args.protocol);print(json.dumps({k:r[k] for k in ('training_gate_passed','optimizer_steps','actual_world_ticks')},indent=2));return 0 if r['training_gate_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
