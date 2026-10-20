"""One predeclared, gated flower-landing experiment. Never automatically retries.

Freeze with --prepare DIRECTORY, then run once with --run DIRECTORY.
All final cases are reserved until earlier gates pass. Development previews
must use other seeds. A failed gate writes a closed result and stops the run.
"""

from collections import Counter
from dataclasses import asdict
import argparse
import json
import math
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch

from .anatomy import CACHE, ROOT, digest, load_manifest
from .contracts import CONTROL_DT, TrialSpec, FlightCommand
from .neural import FlightBrain, MotorFeatures, FlightReadout, FlightController, save_readout, load_readout
from .teacher import FlightTeacher
from .world import FlightWorld, make_trial


def placements(split: str) -> list[TrialSpec]:
    """Balanced, disjoint seed ranges; geometry belongs only to the evaluator."""
    start,repeats={'train':(960000,2),'validation':(961000,1),'final':(962000,4)}[split]
    return [make_trial(start+r*24+h*8+s,sector=s,height=h)
            for r in range(repeats) for h in range(3) for s in range(8)]


def paired_p(trained,control) -> float:
    """Exact one-sided paired sign/McNemar test on discordant binary outcomes."""
    wins=sum(a and not b for a,b in zip(trained,control))
    losses=sum(b and not a for a,b in zip(trained,control)); n=wins+losses
    return sum(math.comb(n,k) for k in range(wins,n+1))/2**n if n else 1.


def accepted(rows: list[dict], minimum: int, per_height: int=0) -> bool:
    counts=Counter(r['height_group'] for r in rows if r['success'])
    return sum(counts.values())>=minimum and all(counts[h]>=per_height for h in ('lower','level','higher'))


def action_errors(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    """Separate MSE for power, pitch, roll, yaw; legs are a deployment channel."""
    return np.mean((predicted[:,:4]-target[:,:4])**2,axis=0).tolist()


def write_json(path: Path, value) -> None:
    from .protocol import durable_json
    durable_json(path,value)


def neural_prerequisites(manifest: dict) -> dict:
    """Frozen thresholds: signal distinction, forward propagation and causal cuts."""
    extractor=MotorFeatures(manifest); results={}; features={}
    for label,flower in {'above':(.3,.45,.18),'below':(.3,.45,.04),
                         'left':(.17,.42,.12),'right':(.43,.42,.12),'behind':(.3,.15,.12)}.items():
        obs=FlightWorld(TrialSpec(959000,flower)).observe(); brain=FlightBrain(manifest)
        rates=brain.sensory_rates(obs)
        for _ in range(100): brain.advance_rates(rates)
        features[label]=(rates,extractor(brain).numpy())
    for a,b in (('above','below'),('left','right'),('behind','right')):
        sensor=float(np.linalg.norm(features[a][0]-features[b][0]))
        motor=float(np.linalg.norm(features[a][1]-features[b][1]))
        results[a+'_'+b]={'sensory_difference':sensor,'motor_difference':motor,'pass':sensor>1e-6 and motor>1e-6}
    full=FlightBrain(manifest); rates=np.zeros(len(manifest['nodes']))
    rates[full.inputs.numpy()]=240
    for _ in range(100): full.advance_rates(rates)
    intact=extractor(full).numpy(); offset=0
    for role,size in zip(('flight_power_dn','flight_steering_dn','landing_dn'),extractor.sizes):
        cut=FlightBrain(manifest); cut.silenced={n['id'] for n in manifest['nodes'] if role in n['roles']}
        for _ in range(100): cut.advance_rates(rates)
        blocked=extractor(cut).numpy()[offset:offset+size]; endpoint=intact[offset:offset+size]
        delta=float(np.linalg.norm(endpoint-blocked))
        results[role]={'motor_norm':float(np.linalg.norm(endpoint)),'cut_difference':delta,
                       'pass':bool(np.any(endpoint>0) and delta>1e-6)}
        offset+=size
    model=FlightReadout(extractor.sizes)
    with torch.no_grad(): zero=model(torch.zeros(sum(extractor.sizes))).numpy()
    falling=FlightWorld(make_trial(959010)); rising=FlightWorld(make_trial(959010))
    for _ in range(10): falling.step(FlightCommand()); rising.step(FlightCommand(.7))
    results['motor_silence_and_mechanics']={'silent_output':zero.tolist(),
        'fall_z':float(falling.position[2]),'rise_z':float(rising.position[2]),
        'pass':bool(np.all(zero==0) and falling.position[2]<.1<rising.position[2])}
    return {'passed':all(r['pass'] for r in results.values()),'checks':results}


class Experiment:
    """Own evaluation geometry, bookkeeping and the strict stage transition gates."""

    def __init__(self,directory: Path):
        from .protocol import load_frozen
        self.directory=directory.resolve()
        self.protocol=load_frozen(self.directory)
        self.manifest=json.loads((self.directory/'circuit.json').read_text())
        self.sha=self.protocol['circuit_sha256']; self.seed=self.protocol['seed']
        self.started=time.monotonic(); self.rows=[]; self.updates=0; self.stage='prepared'
        self.model=None; self.before=None

    def cases(self,split: str) -> list[TrialSpec]:
        """Read exact frozen coordinates. Never call the historical generator."""
        return [TrialSpec(c['seed'],tuple(c['flower']),c['height_group'],c['sector'])
                for c in self.protocol['cases'][split]]

    def journal(self,event: str,**details) -> None:
        row={'event':event,'stage':self.stage,'actual_optimizer_updates':self.updates,
             'elapsed_seconds':time.monotonic()-self.started,**details}
        with (self.directory/'progress.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        write_json(self.directory/'progress.json',row)

    def enter(self,stage: str) -> None:
        self.stage=stage;self.journal('stage_started')
        write_json(self.directory/'results.json',{'status':'incomplete','stage':stage,
            'actual_optimizer_updates':self.updates,'learned_landing_accepted':False,
            'note':'Running; process interruption does not authorize a restart.'})

    def budget(self) -> None:
        if time.monotonic()-self.started_collection>self.protocol['collection_fit_seconds_limit']: raise TimeoutError('two-hour collection/fitting ceiling')

    def episode(self,case: TrialSpec,controller,label: str,collect: bool=False):
        self.journal('episode_started',label=label,seed=case.seed)
        world=FlightWorld(case); controller.reset()
        brain=FlightBrain(self.manifest) if collect else None
        extractor=MotorFeatures(self.manifest); x=[]; y=[]; trace=[]
        for tick in range(round(self.protocol['episode_seconds']/CONTROL_DT)):
            if tick%100==0 and collect: self.budget()
            obs=world.observe(); command=controller.command(obs)
            if brain is not None:
                brain.advance(obs); x.append(extractor(brain).numpy()); y.append(command.vector())
            snapshot=world.step(command)
            trace.append((*snapshot.position,*snapshot.velocity,*snapshot.orientation,*command.vector(),
                          snapshot.foot_contact,snapshot.food_contact,snapshot.crashed,snapshot.dwell))
        success=bool(world.food_contact and not world.crashed and world.dwell>=2 and np.linalg.norm(world.velocity)<1e-4)
        row={'label':label,'seed':case.seed,'height_group':case.height_group,'sector':case.sector,
             'success':success,'crashed':world.crashed,'dwell_seconds':world.dwell,
             'first_contact_seconds':world.first_contact,'impact_speed_m_s':world.impact_speed,
             'departures':world.departures,'ticks':round(self.protocol['episode_seconds']/CONTROL_DT)}
        folder=self.directory/'trajectories'; folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f'{label}-{case.seed}.npz',trace=np.asarray(trace,dtype=np.float32),
                            features=np.asarray(x),commands=np.asarray(y))
        with (self.directory/'episodes.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
        self.rows.append(row)
        self.journal('episode_completed',label=label,seed=case.seed,success=success)
        print(f'{label} {case.seed}: success={success} dwell={world.dwell:.3f}',flush=True)
        return row,x,y

    def close(self,status: str,stage: str,**details) -> dict:
        self.journal('stage_closed',status=status)
        result={'actual_optimizer_updates':self.updates,'status':status,'stage':stage,'learned_landing_accepted':status=='accepted',
                'elapsed_seconds':time.monotonic()-self.started,'episodes':self.rows,**details}
        write_json(self.directory/'results.json',result); print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)
        return result

    def run(self) -> dict:
        from .protocol import consume_authorization
        consume_authorization(self.directory,digest(self.directory/'protocol.json'))
        self.enter('neural_prerequisites')
        torch.set_num_threads(1); torch.manual_seed(self.seed)
        try: return self._run()
        except TimeoutError as exc: return self.close('incomplete','budget',reason=str(exc))
        except BaseException as exc:
            if self.model is not None: save_readout(self.directory/'interrupted.pt',self.model,self.sha)
            self.close('incomplete','execution_error',reason=f'{type(exc).__name__}: {exc}')
            raise

    def _run(self) -> dict:
        prereq=neural_prerequisites(self.manifest); write_json(self.directory/'prerequisites.json',prereq)
        if not prereq['passed']: return self.close('closed_failed_gate','neural_prerequisites',prerequisites=prereq)
        self.enter('teacher')
        self.started_collection=time.monotonic()
        xs=[]; ys=[]; teacher_rows=[]
        for case in self.cases('train'):
            row,x,y=self.episode(case,FlightTeacher(),'teacher-train',True)
            teacher_rows.append(row); xs.extend(x); ys.extend(y)
            if not row['success']: return self.close('closed_failed_gate','teacher',required='48/48')
        x=torch.tensor(np.asarray(xs),dtype=torch.float32); y=torch.tensor(np.asarray(ys),dtype=torch.float32)
        np.savez_compressed(self.directory/'training.npz',features=x.numpy(),commands=y.numpy())
        sizes=MotorFeatures(self.manifest).sizes; mean=x.mean(0); scale=x.std(0).clamp_min(1)
        model=FlightReadout(sizes,mean,scale,self.seed); self.model=model
        save_readout(self.directory/'untrained.pt',model,self.sha)
        before={k:v.detach().clone() for k,v in model.named_parameters()}
        self.before=before; self.enter('fit')
        optimizer=torch.optim.Adam(model.parameters(),lr=self.protocol['learning_rate']); generator=torch.Generator().manual_seed(self.seed)
        curves=[]; gradients={}
        for update in range(self.protocol['updates']):
            if update%50==0: self.budget()
            ix=torch.randint(len(x),(self.protocol['batch_size'],),generator=generator)
            optimizer.zero_grad(); prediction=model(x[ix]); loss=(prediction-y[ix]).square().mean(); loss.backward()
            norm=math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters()))
            if update in (0,self.protocol['updates']-1): gradients[str(update+1)]={n:p.grad.tolist() for n,p in model.named_parameters()}
            self.journal('optimizer_update_started',next_update=update+1)
            optimizer.step(); self.updates=update+1
            temporary=self.directory/'candidate.tmp'
            save_readout(temporary,model,self.sha); os.replace(temporary,self.directory/'candidate.pt')
            curves.append({'update':update+1,'loss':float(loss.detach()),'gradient_norm':norm})
            self.journal('optimizer_update_completed',loss=float(loss.detach()),gradient_norm=norm,
                         checkpoint_sha256=digest(self.directory/'candidate.pt'))
            if update in (0,self.protocol['updates']-1):write_json(self.directory/'gradients.json',gradients)
            write_json(self.directory/'parameter_changes.json',{n:float((p.detach()-before[n]).norm()) for n,p in model.named_parameters()})
        deltas={n:float((p.detach()-before[n]).norm()) for n,p in model.named_parameters()}
        write_json(self.directory/'learning_curves.json',curves); write_json(self.directory/'gradients.json',gradients)
        write_json(self.directory/'parameter_changes.json',deltas)
        save_readout(self.directory/'trained.pt',model,self.sha)
        with torch.no_grad(): before_reload=model(x[:512]).clone()
        model=load_readout(self.directory/'trained.pt',self.sha)
        reload_ok=torch.equal(before_reload,model(x[:512]))
        write_json(self.directory/'reload.json',{'exact_outputs':reload_ok,'frozen_parameters':all(not p.requires_grad for p in model.parameters())})
        if not reload_ok or not all(v>0 for v in deltas.values()) or not all(c['gradient_norm']>0 for c in curves):
            return self.close('closed_failed_gate','fit_integrity')
        self.enter('training')
        with torch.no_grad(): prediction=model(x).numpy()
        trained_mse=action_errors(prediction,y.numpy()); baseline=action_errors(np.broadcast_to(y.mean(0).numpy(),y.shape),y.numpy())
        errors={'trained_mse':trained_mse,'mean_command_mse':baseline,'pass':all(a<b for a,b in zip(trained_mse,baseline))}
        write_json(self.directory/'action_errors.json',errors)
        if not errors['pass']:
            return self.close('closed_failed_gate','training_command_error',action_errors=errors,
                              unexecuted=['training_behavior','validation','final'])
        trained=FlightController(self.manifest,model)
        rows=[self.episode(c,trained,'trained-train')[0] for c in self.cases('train')]
        if not accepted(rows,self.protocol['training_gate']) or not errors['pass']:
            return self.close('closed_failed_gate','training',successes=sum(r['success'] for r in rows),action_errors=errors)
        self.enter('validation')
        rows=[self.episode(c,trained,'trained-validation')[0] for c in self.cases('validation')]
        if not accepted(rows,self.protocol['validation_gate']): return self.close('closed_failed_gate','validation',successes=sum(r['success'] for r in rows))
        self.enter('final')
        outcomes={}
        controllers={'trained':trained,'untrained':FlightController(self.manifest,load_readout(self.directory/'untrained.pt',self.sha)),
                     'fixed':FlightController(self.manifest)}
        for name,controller in controllers.items():
            outcomes[name]=[self.episode(c,controller,name+'-final')[0] for c in self.cases('final')]
        p={name:paired_p([r['success'] for r in outcomes['trained']],[r['success'] for r in outcomes[name]]) for name in ('untrained','fixed')}
        passed=accepted(outcomes['trained'],self.protocol['final_gate'],self.protocol['final_per_height']) and all(v<=self.protocol['paired_p_limit'] for v in p.values())
        return self.close('accepted' if passed else 'closed_failed_gate','final',paired_p=p,
                          successes={k:sum(r['success'] for r in v) for k,v in outcomes.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',type=Path)
    group.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.prepare:
        from .protocol import freeze
        freeze(args.prepare)
    else: Experiment(args.run).run()
