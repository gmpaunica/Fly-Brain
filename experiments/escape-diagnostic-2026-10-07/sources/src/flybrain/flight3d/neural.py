"""Fixed sourced spiking circuit and trainable motor-endpoint readouts."""

import math
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .contracts import CONTROL_DT, FlightCommand, FlightObservation
from .anatomy import validate_manifest
from .sensors import flower_contrast


@torch.jit.script
def _advance(voltage: torch.Tensor, previous: torch.Tensor, phase: torch.Tensor,
             rates: torch.Tensor, pre: torch.Tensor, post: torch.Tensor,
             weights: torch.Tensor, inputs: torch.Tensor, live: torch.Tensor,
             steps: int):
    counts=torch.zeros_like(voltage,dtype=torch.int64)
    delivered=torch.zeros_like(weights)
    effective=(~inputs[post]) & live[post]
    for _ in range(steps):
        voltage=voltage*0.9900498337491681  # exp(-.001/.100)
        contribution=weights*previous[pre]
        voltage.index_add_(0,post,contribution)
        delivered=delivered+contribution*effective
        phase=phase+rates*.001
        forced=phase>=1.
        phase=torch.where(forced,phase-1.,phase)
        spikes=torch.where(inputs,forced,voltage>=1.) & live
        voltage=torch.where(inputs | ~live,torch.zeros_like(voltage),voltage-spikes.to(voltage.dtype))
        previous=spikes.to(voltage.dtype)
        counts=counts+spikes.to(torch.int64)
    return voltage,previous,phase,counts,delivered


class FlightBrain:
    """589-cell source slice (size comes from manifest), engineered dynamics.

    Non-input cells use 100 ms leak, threshold 1, subtractive reset, and 1 ms
    delayed source edges. Incoming count shares sum to .9 per destination,
    excluding self edges from dynamics but preserving them in source records.
    Sensory entry cells use deterministic 0..240 Hz phase encoding. Visual
    tuning is an explicit assumed input model, not source-measured physiology.
    """

    def __init__(self,manifest: dict):
        validate_manifest(manifest)
        self.manifest=manifest; self.nodes=manifest['nodes']
        self.ids=tuple(n['id'] for n in self.nodes); self.index={v:i for i,v in enumerate(self.ids)}
        self.roles={role:np.array([i for i,n in enumerate(self.nodes) if role in n['roles']],dtype=int)
                    for role in {r for n in self.nodes for r in n['roles']}}
        count=np.zeros(len(self.ids))
        triples=[(self.index[e['pre']],self.index[e['post']],e['count']) for e in manifest['edges'] if e['pre']!=e['post']]
        for pre,post,n in triples: count[post]+=n
        self.pre=torch.tensor([e[0] for e in triples],dtype=torch.int64)
        self.post=torch.tensor([e[1] for e in triples],dtype=torch.int64)
        self.weights=torch.tensor([.9*n/count[post] for pre,post,n in triples],dtype=torch.float64)
        self.edge_order=tuple((e['pre'],e['post']) for e in manifest['edges'])
        self.dynamic_edges=np.array([i for i,e in enumerate(manifest['edges']) if e['pre']!=e['post']],dtype=int)
        self.inputs=torch.tensor([bool(set(n['roles'])&{'odor','vision','optic_flow','motion','taste','touch'}) for n in self.nodes])
        self.silenced:set[str]=set()
        self.reset()

    def reset(self) -> None:
        """Clear every episode state; preserve graph and selected intervention."""
        self.voltage=torch.zeros(len(self.ids),dtype=torch.float64)
        self.previous=torch.zeros_like(self.voltage); self.phase=torch.zeros_like(self.voltage)
        self.history=torch.zeros((2,len(self.ids)),dtype=torch.float64)
        self.steps=0; self.previous_eyes=None
        self.last_counts=np.zeros(len(self.ids),dtype=np.int64)
        self.edge_transmissions=(0.,)*len(self.edge_order)

    def sensory_rates(self,observation: FlightObservation) -> np.ndarray:
        """Map local sensory samples to declared source entry cells only."""
        rates=np.zeros(len(self.ids))
        eyes=observation.eyes
        salience=np.clip(flower_contrast(eyes)*3,0,1)
        delta=np.zeros_like(salience) if self.previous_eyes is None else salience-self.previous_eyes
        self.previous_eyes=salience.copy()
        for role,indices in self.roles.items():
            if role not in ('odor','vision','optic_flow','motion','taste','touch'): continue
            for side in ('left','right'):
                group=[int(i) for i in indices if self.nodes[i]['metadata'].get('side')==side]
                eye=0 if side=='left' else 1
                for rank,i in enumerate(group):
                    if role=='odor':
                        odor=observation.odor_left if eye==0 else observation.odor_right
                        rates[i]=240*odor/(odor+.12)
                    elif role=='vision':
                        # Assigned 8x6 visual fields within each hemifield. Source
                        # IDs retain anatomy; ID order does NOT imply retinotopy.
                        row=(rank%6)*4; column=(rank//6%8)*6
                        patch=salience[eye,row:row+4,column:column+6]
                        loom=np.maximum(delta[eye,row:row+4,column:column+6],0)
                        rates[i]=240*min(1,float(patch.max())+.5*float(loom.max()))
                    elif role=='optic_flow':
                        # Coarse motion-response proxy from image changes.
                        value=float(np.mean(np.abs(delta[eye])))
                        rates[i]=min(240,30+1200*value)
                    elif role=='motion':
                        motion=(*observation.angular_velocity,*(v/9.81 for v in observation.specific_force))
                        signal=motion[(rank//2)%6]*(1 if rank%2 else -1)
                        rates[i]=120*(1+math.tanh(signal))
                    elif role=='taste': rates[i]=240*observation.food_contact
                    elif role=='touch': rates[i]=240*observation.foot_contact
        return rates

    @torch.no_grad()
    def advance_rates(self,rates: np.ndarray,steps: int=20) -> np.ndarray:
        """Advance actual spikes with explicit delayed source connectivity."""
        values=np.asarray(rates,dtype=float)
        if values.shape!=(len(self.ids),) or not np.isfinite(values).all() or np.any(values<0) or np.any(values>240):
            raise ValueError('one finite 0..240 Hz input rate per neuron required')
        live=torch.tensor([i not in self.silenced for i in self.ids])
        self.voltage,self.previous,self.phase,counts,delivered=_advance(self.voltage,self.previous,self.phase,
            torch.from_numpy(values),self.pre,self.post,self.weights,self.inputs,live,steps)
        self.steps+=steps
        elapsed=steps*.001
        for k,tau in enumerate((.1,.5)):
            alpha=-math.expm1(-elapsed/tau)
            self.history[k]=(1-alpha)*self.history[k]+alpha*counts.double()/elapsed
            self.history[k,~live]=0.
        self.last_counts=counts.numpy().copy()
        totals=np.zeros(len(self.edge_order))
        totals[self.dynamic_edges]=delivered.numpy()
        self.edge_transmissions=tuple(float(v) for v in totals)
        return self.last_counts.copy()

    def advance(self,observation: FlightObservation) -> np.ndarray:
        return self.advance_rates(self.sensory_rates(observation))


class MotorFeatures:
    """Only named motor endpoints reach learned actuator heads."""

    def __init__(self,manifest: dict):
        self.groups=[[i for i,n in enumerate(manifest['nodes']) if role in n['roles']]
                     for role in ('wing_power','wing_steering','landing_motor')]
        if any(not g for g in self.groups): raise ValueError('all three motor roles required')
        self.sizes=tuple(2*len(g) for g in self.groups)

    def __call__(self,brain: FlightBrain) -> torch.Tensor:
        return torch.cat([brain.history[:,g].flatten() for g in self.groups]).float()


class FlightReadout(nn.Module):
    """Three independent heads: power, pitch/roll/yaw, and leg deployment.

    An output cannot read a different group's features. Zero motor histories
    give zero actuator drive, including when learned biases are nonzero.
    """

    def __init__(self,sizes,mean=None,scale=None,seed: int=3030):
        super().__init__(); self.sizes=tuple(sizes); self.seed=seed
        total=sum(sizes)
        self.register_buffer('mean',torch.zeros(total) if mean is None else mean.clone())
        self.register_buffer('scale',torch.ones(total)*100 if scale is None else scale.clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.heads=nn.ModuleList([nn.Sequential(nn.Linear(n,32),nn.Tanh(),nn.Linear(32,out))
                                      for n,out in zip(sizes,(1,3,1))])

    def forward(self,rates: torch.Tensor) -> torch.Tensor:
        parts=torch.split((rates-self.mean)/self.scale,self.sizes,dim=-1)
        masks=[(r.abs().sum(-1,keepdim=True)>0).to(r.dtype) for r in torch.split(rates,self.sizes,dim=-1)]
        output=[head(p) for head,p in zip(self.heads,parts)]
        return torch.cat((torch.sigmoid(output[0])*masks[0],torch.tanh(output[1])*masks[1],
                          torch.sigmoid(output[2])*masks[2]),dim=-1)


def save_readout(path: Path,model: FlightReadout,manifest_sha256: str) -> None:
    """Save learned values and source-identity binding, never recurrent state."""
    torch.save({'schema':'flight3d-readout-v1','sizes':model.sizes,'seed':model.seed,
                'manifest_sha256':manifest_sha256,'state':model.state_dict()},path)


def load_readout(path: Path,manifest_sha256: str) -> FlightReadout:
    """Reload an identity-bound readout frozen for inference."""
    record=torch.load(path,map_location='cpu',weights_only=True)
    if record['schema']!='flight3d-readout-v1' or record['manifest_sha256']!=manifest_sha256:
        raise ValueError('checkpoint circuit binding mismatch')
    model=FlightReadout(record['sizes'],seed=record['seed']); model.load_state_dict(record['state'])
    model.eval().requires_grad_(False)
    return model


class FlightController:
    """Local observation -> sourced spikes -> motor endpoint histories -> actuators."""

    def __init__(self,manifest: dict,model: FlightReadout | None=None):
        self.brain=FlightBrain(manifest); self.features=MotorFeatures(manifest); self.model=model

    def reset(self) -> None: self.brain.reset()

    @torch.no_grad()
    def command(self,observation: FlightObservation) -> FlightCommand:
        self.brain.advance(observation)
        rates=self.features(self.brain)
        if self.model is not None: values=self.model(rates).numpy()
        else:
            p,s,l=torch.split(rates,self.features.sizes)
            # Fixed baseline chosen before behavioral outcomes.
            values=np.array((min(float(p.mean())/160,1),0.,0.,
                float(torch.tanh((s[0]-s[len(s)//2])/100)),min(float(l.mean())/80,1)))
        return FlightCommand(*(float(x) for x in values))
