"""Explicit zero-step rounding reconciliation and one bounded precision fit."""

import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

import torch

from flybrain.neural.banc_projection import ProjectionReadout,save_checkpoint,load_checkpoint,normalization
from .banc_projection import (DEFAULT_COLLECTION as RAW,DEFAULT_OUTPUT,load_collection,
    training_definition,evaluate_models,coverage_gates)
from .banc_coverage import load_training_data as old_data,_write_protocol,_read_protocol
from .banc_protocol import sha256,software_record
from .banc_readiness import PROJECT_ROOT,capture_provenance,verify_provenance,save_new_json
from .banc_training import fit_readout,_write_csv
from .evaluation import locked_cases,EvaluationConfig,_git_commit
from .readout_experiment import TrainingDataset
from .records import ensure_new_run_directory


CANONICAL=PROJECT_ROOT/'experiments/fb10-projection-canonical-data'


def canonical_features(raw,old):
    """Accept only isolated <=1e-12 original slow-history arithmetic differences."""
    if raw.ndim!=2 or old.shape!=(len(raw),16) or raw.shape[1]!=20 or not torch.isfinite(raw).all():
        raise ValueError('matching finite old/new matrices required')
    if not torch.equal(raw[:, :8],old[:,:8]) or not torch.equal(raw[1008:,:16],old[1008:]):
        raise ValueError('fast or coverage features changed')
    error=float((raw[:1008,8:16]-old[:1008,8:16]).abs().max())
    if error>1e-12:raise ValueError('original slow histories differ materially')
    return torch.cat((old,raw[:,16:]),dim=1),error


def data_definition():
    load_collection(RAW/'protocol.json')
    return {'schema':'fb10_projection_canonical_data_v1','experiment':'fb10_projection_canonical_data',
        'hypothesis':'The only raw replay mismatch is float64 exp versus 1-expm1 arithmetic in original slow histories, bounded by 1e-12 Hz.',
        'raw_collection_sha256':{n:sha256(RAW/n) for n in ('protocol.json','results.json','samples.csv')},
        'optimizer_steps':0,'max_world_ticks':0,'max_neural_steps':0,'original_slow_rounding_tolerance_hz':1e-12,
        'operation':'retain exact original 16 endpoint histories; append measured PN histories unchanged; no sample or target selection',
        'stop_condition':'One file-only reconciliation; reject any other difference or failed raw integrity check.'}


def _raw_data():
    result=json.loads((RAW/'results.json').read_text(encoding='utf-8'))
    checks=result['checks']
    if (set(k for k,v in checks.items() if v is not True)!={'old_features_exact'}
            or result['readiness_passed'] is not False or sha256(RAW/'samples.csv')!=result['samples_sha256']):
        raise ValueError('expected only the documented raw rounding failure')
    with (RAW/'samples.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    x=torch.tensor([[float(r[f'feature_{i}']) for i in range(20)] for r in rows],dtype=torch.float64)
    y=torch.tensor([[float(r['target_forward']),float(r['target_turn'])] for r in rows],dtype=torch.float64)
    old=old_data()
    if not torch.equal(y,old.targets):raise ValueError('canonical targets differ')
    x,error=canonical_features(x,old.features);normalization(x)
    return TrainingDataset(x,y,old.case_counts),error,rows


def prepare_data(output_dir: Path) -> Path:
    """Freeze and execute only an offline arithmetic conversion, zero simulation."""
    out=Path(output_dir).resolve();ensure_new_run_directory(out)
    definition=data_definition()
    p={**definition,'output_dir':str(out),'prepared_utc':datetime.now(timezone.utc).isoformat(),'provenance':capture_provenance(PROJECT_ROOT,out)}
    path=_write_protocol(out,p);data,error,rows=_raw_data()
    save_new_json(out/'execution.json',{'status':'consumed','protocol_sha256':sha256(path),'optimizer_steps':0})
    with (out/'samples.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        for i,r in enumerate(rows):writer.writerow({**r,**{f'feature_{j}':float(v) for j,v in enumerate(data.features[i])}})
    r={'experiment':p['experiment'],'protocol_sha256':sha256(path),'software':software_record(),
        'sample_count':len(rows),'case_counts':data.case_counts,'maximum_original_slow_rounding_error_hz':error,
        'actual_world_ticks':0,'actual_neural_steps':0,'actual_episodes':0,'optimizer_steps':0,
        'samples_sha256':sha256(out/'samples.csv'),'all_original_16_features_exact':True,
        'all_measured_pn_features_exact':True,'all_labels_exact':True,
        'provenance_preserved':verify_provenance(p['provenance']),'readiness_passed':True}
    if not r['provenance_preserved']:raise RuntimeError('canonicalization provenance changed')
    save_new_json(out/'results.json',r);return path


def load_data(directory: Path=CANONICAL):
    directory=Path(directory);p,digest=_read_protocol(directory/'protocol.json')
    if any(p.get(k)!=v for k,v in data_definition().items()):raise ValueError('canonical scope differs')
    r=json.loads((directory/'results.json').read_text(encoding='utf-8'))
    if r['readiness_passed'] is not True or r['protocol_sha256']!=digest or sha256(directory/'samples.csv')!=r['samples_sha256']:
        raise ValueError('canonical readiness differs')
    expected,error,raw_rows=_raw_data()
    with (directory/'samples.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    x=torch.tensor([[float(a[f'feature_{i}']) for i in range(20)] for a in rows],dtype=torch.float64)
    y=torch.tensor([[float(a['target_forward']),float(a['target_turn'])] for a in rows],dtype=torch.float64)
    if (not torch.equal(x,expected.features) or not torch.equal(y,expected.targets)
            or error!=r['maximum_original_slow_rounding_error_hz']
            or [(a['case_id'],a['trajectory'],a['tick']) for a in rows]!=[(a['case_id'],a['trajectory'],a['tick']) for a in raw_rows]):
        raise ValueError('canonical data values/identities differ')
    return expected


def prepare_training(output_dir: Path) -> Path:
    out=Path(output_dir).resolve();ensure_new_run_directory(out);data=load_data()
    return _write_protocol(out,{**training_definition(data),'output_dir':str(out),'prepared_utc':datetime.now(timezone.utc).isoformat(),
        'git_commit':_git_commit(),'provenance':capture_provenance(PROJECT_ROOT,out),
        'collection_sha256':{n:sha256(CANONICAL/n) for n in ('protocol.json','results.json','samples.csv')}})


def load_training_protocol(path: Path):
    p,digest=_read_protocol(path)
    if any(sha256(CANONICAL/n)!=h for n,h in p['collection_sha256'].items()):raise ValueError('canonical data bytes differ')
    data=load_data()
    if any(p.get(k)!=v for k,v in training_definition(data).items()):raise ValueError('precision fit settings differ')
    return p,digest,data


def run_training(path: Path) -> dict:
    """One consumed 500-update trial with unchanged original and covered gates."""
    path=Path(path).resolve();out=path.parent
    if any((out/n).exists() for n in ('execution.json','results.json','trained.pt','untrained.pt','curve.csv','episodes.csv')):
        raise FileExistsError('precision fit consumed')
    p,digest,data=load_training_protocol(path)
    save_new_json(out/'execution.json',{'status':'consumed','protocol_sha256':digest,'started_utc':datetime.now(timezone.utc).isoformat()})
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        mean=torch.tensor(p['normalization']['mean']);scale=torch.tensor(p['normalization']['scale'])
        trained,untrained=[ProjectionReadout(mean,scale) for _ in range(2)]
        save_checkpoint(out/'untrained.pt',untrained,protocol_sha256=digest);fit=fit_readout(trained,data.features,data.targets)
        r={'experiment':p['experiment'],'protocol_sha256':digest,'software':software_record(),'git_commit':p['git_commit'],
            'seed':1010,'network_size':83,'learning_rate':.01,'objective':p['objective'],'sample_count':len(data.features),
            **fit,'training_gate_passed':False,'held_out_evaluated':False,'checkpoint':None,'actual_world_ticks':0,'actual_neural_steps':0,'actual_episodes':0}
        if fit['curve']:_write_csv(out/'curve.csv',fit['curve'])
        if fit['failure'] is None:
            save_checkpoint(out/'trained.pt',trained,protocol_sha256=digest)
            loaded=load_checkpoint(out/'trained.pt',protocol_sha256=digest,mean=mean,scale=scale)
            with torch.no_grad():
                error=(trained(data.features).double()-data.targets).square();errors=error.mean(0);original_errors=error[:1008].mean(0)
                exact=torch.equal(trained(data.features),loaded(data.features))
            b=evaluate_models(trained,loaded,untrained,locked_cases()['train'],EvaluationConfig());r.update(b)
            checks={'checkpoint_outputs_exact':exact,'checkpoint_replay_exact':b['checkpoint_replay_exact'],
                'effective_weights_unchanged':b['effective_weights_unchanged'],'execution_budget_passed':b['execution_budget_passed'] and b['actual_episodes']==25 and b['actual_world_ticks']<=15000 and b['actual_neural_steps']<=300000,
                'optimizer_budget_passed':fit['optimizer_steps']==500,'gradients_and_updates_observed':fit['gradient_nonzero_scalars']>0 and fit['parameter_changed_scalars']>0,
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
    finally:torch.set_num_threads(previous)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);m=parser.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare-data',action='store_true');m.add_argument('--prepare',action='store_true');m.add_argument('--fit',action='store_true')
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--protocol',type=Path,default=DEFAULT_OUTPUT/'protocol.json');a=parser.parse_args()
    if a.prepare_data or a.prepare:
        path=prepare_data(a.output_dir or CANONICAL) if a.prepare_data else prepare_training(a.output_dir or DEFAULT_OUTPUT)
        print(json.dumps({'protocol':str(path),'sha256':sha256(path),'optimizer_steps':0}));return 0
    r=run_training(a.protocol);print(json.dumps({k:r[k] for k in ('training_gate_passed','optimizer_steps','actual_world_ticks','actual_neural_steps')},indent=2))
    return 0 if r['training_gate_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
