"""Observe real PN synaptic spike events without changing the sourced circuit."""

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import Tensor,nn

from flybrain.data.banc import load_banc_subset
from flybrain.world.fly import MotorCommand
from .banc_network import BancNetwork,load_verified_banc_config
from .banc_memory import memory_features,memory_contract,FAST_TAU,SLOW_TAU
from .banc_motor import BancMotorDecoder
from .banc_readout import _valid_digest
from .controller import NeuralController
from .network import NeuralActivity


def incoming_event_rate(weighted_events: Tensor,elapsed_seconds: float) -> Tensor:
    """Convert two summed PN input events to budget-normalized equivalent Hz.

    W is [postsynaptic,presynaptic]. [1,2,0] @ [1,0,1] =1 incoming
    weighted event; over 20ms with fixed budget3 this is 1/(3*.02) Hz.
    This observes the real previous-step spikes, never external sensory drive.
    """
    if (not isinstance(weighted_events,Tensor) or weighted_events.shape!=(2,)
            or not torch.isfinite(weighted_events).all() or torch.any(weighted_events<0)
            or not math.isfinite(elapsed_seconds) or elapsed_seconds<=0):
        raise ValueError('expected two nonnegative event sums and positive seconds')
    return weighted_events.double()/(3.*elapsed_seconds)


@dataclass(frozen=True)
class ProjectionActivity(NeuralActivity):
    """Original full-interval counts plus two detached internal spike-event rates."""

    projection_event_rates: Tensor | None = None


class ProjectionNetwork(BancNetwork):
    """Unmodified BANC steps with a read-only incoming-event accumulator."""

    def advance(self,observation,substeps: int=20) -> ProjectionActivity:
        """Account every actual 1ms step once; observe the weights used by neurons."""
        if type(substeps) is not int or substeps<=0:raise ValueError('positive integer substeps required')
        indices=self.topology.indices('projection_left')+self.topology.indices('projection_right')
        if len(indices)!=2:raise ValueError('exactly one projection neuron per side required')
        rows=self.population.weights[list(indices)].double()
        summed=torch.zeros(2,dtype=torch.float64)
        counts=torch.zeros(self.population.size,dtype=torch.int64)
        start=self.population.step_count
        for _ in range(substeps):
            summed+=rows@self.population.spikes.double()
            activity=super().advance(observation,1)
            counts+=activity.spike_counts
        elapsed=(self.population.step_count-start)*self.population.config.dt
        return ProjectionActivity(start,self.population.step_count,elapsed,counts,self.snapshot(),
                                  projection_event_rates=incoming_event_rate(summed,elapsed))


def projection_features(rates: Tensor) -> Tensor:
    """20 raw neural rates ->16 endpoint contrasts plus fast/slow PN mean/diff."""
    if (not isinstance(rates,Tensor) or rates.ndim not in (1,2) or rates.shape[-1]!=20
            or rates.numel()==0 or rates.device.type!='cpu' or not rates.is_floating_point()
            or not torch.isfinite(rates).all() or torch.any(rates<0) or torch.any(rates>1000)):
        raise ValueError('expected finite nonnegative [20] or [N,20] neural rates <=1000 Hz')
    pn=rates[...,16:]
    return torch.cat((memory_features(rates[...,:16]),
        torch.stack(((pn[...,0]+pn[...,1])/2,pn[...,1]-pn[...,0],
                     (pn[...,2]+pn[...,3])/2,pn[...,3]-pn[...,2]),dim=-1)),dim=-1)


def normalization(rates: Tensor) -> tuple[Tensor,Tensor]:
    """Freeze training-only conditioned population means/stds, floor 1Hz."""
    if not isinstance(rates,Tensor) or rates.ndim!=2:raise ValueError('training matrix required')
    values=projection_features(rates.double())
    return values.mean(0).float(),values.std(0,correction=0).clamp_min(1).float()


class ProjectionReadout(nn.Module):
    """20->16->2 motor readout; its 370 scalars are the only trainable state."""

    def __init__(self,mean: Tensor,scale: Tensor,*,seed: int=1010) -> None:
        super().__init__()
        if (type(seed) is not int or seed<0
                or any(not isinstance(t,Tensor) or t.shape!=(20,) or t.device.type!='cpu'
                       or not t.is_floating_point() or not torch.isfinite(t).all() for t in (mean,scale))
                or torch.any(mean.abs()>1000) or torch.any(mean[[0,1,2,3,8,9,10,11,16,18]]<0)
                or torch.any(scale<1)):
            raise ValueError('expected 20 finite conditioned means/scales >=1 Hz')
        self.seed=seed
        self.register_buffer('mean',mean.detach().float().clone());self.register_buffer('scale',scale.detach().float().clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden=nn.Linear(20,16);self.output=nn.Linear(16,2)

    def forward(self,rates: Tensor) -> Tensor:
        """Forward tensor calculations; autograd is used only by the fit runner."""
        values=(projection_features(rates).float()-self.mean)/self.scale
        raw=self.output(torch.tanh(self.hidden(values)))
        return torch.stack((torch.sigmoid(raw[...,0]),torch.tanh(raw[...,1])),dim=-1)


class ProjectionDecoder:
    """Two endpoint EMAs and two PN incoming-event EMAs, reset per episode."""

    def __init__(self,topology,model: ProjectionReadout) -> None:
        self.fast=BancMotorDecoder(topology);self.slow=BancMotorDecoder(topology)
        self.slow.filter_tau_seconds=SLOW_TAU;self.model=model
        self.feature_names=tuple(projection_contract()['feature_names']);self.reset()

    @property
    def feature_rates(self) -> Tensor:
        """Detached snapshot; reading it cannot step neurons or alter filters."""
        return torch.cat((self.fast.feature_rates,self.slow.feature_rates,self.pn_fast,self.pn_slow))

    def reset(self) -> None:
        self.fast.reset();self.slow.reset()
        self.pn_fast=torch.zeros(2,dtype=torch.float64);self.pn_slow=self.pn_fast.clone()

    def decode(self,activity: ProjectionActivity) -> MotorCommand:
        """Observe only counted endpoint spikes and measured internal events."""
        if not isinstance(activity,ProjectionActivity):raise ValueError('measured PN activity required')
        events=activity.projection_event_rates
        if events is None or events.shape!=(2,) or not torch.isfinite(events).all() or torch.any(events<0):
            raise ValueError('two finite PN event rates required')
        self.fast.decode(activity);self.slow.decode(activity)
        for name,tau in (('pn_fast',FAST_TAU),('pn_slow',SLOW_TAU)):
            alpha=-math.expm1(-activity.elapsed_seconds/tau)
            setattr(self,name,(1-alpha)*getattr(self,name)+alpha*events)
        with torch.no_grad():output=self.model(self.feature_rates)
        return MotorCommand(float(output[0]),float(output[1]))


def make_projection_controller(model: ProjectionReadout) -> NeuralController:
    """Independent fixed circuit and neural feature state, sharing only the head."""
    network=ProjectionNetwork(load_banc_subset(connectome_gain=1.),dynamics_config=load_verified_banc_config())
    return NeuralController(network,ProjectionDecoder(network.topology,model))


def projection_contract() -> dict:
    """Pin sourced PN identity, event semantics, normalization and feature order."""
    original=memory_contract();topology=load_banc_subset(connectome_gain=1.)
    indices=topology.indices('projection_left')+topology.indices('projection_right')
    names=[f'{topology.groups[i]}:{topology.root_ids[i]}' for i in indices]
    return {**original,'schema':'banc_projection_readout_v1',
        'architecture':{**original['architecture'],'inputs':20},
        'projection_source_names':names,
        'projection_signal':'sum W[PN,pre] @ previous actual spikes / (fixed budget 3 * elapsed seconds)',
        'feature_names':original['feature_names']+[f'{tag}:incoming:{name}' for tag in ('fast100ms','slow500ms') for name in names],
        'conditioning':{**original['conditioning'],'projection_pairs':'fast then slow mean and right-minus-left'}}


def save_checkpoint(path: Path,model: ProjectionReadout,*,protocol_sha256: str) -> None:
    """Exclusive finite checkpoint with strict circuit/feature/protocol binding."""
    if not _valid_digest(protocol_sha256) or any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('finite state and SHA256 required')
    with Path(path).open('xb') as stream:
        torch.save({'contract':projection_contract(),'seed':model.seed,'protocol_sha256':protocol_sha256,'state_dict':model.state_dict()},stream)


def load_checkpoint(path: Path,*,protocol_sha256: str,mean: Tensor,scale: Tensor) -> ProjectionReadout:
    """Reject any changed feature/core schema, normalization, seed or bad tensors."""
    p=torch.load(Path(path),map_location='cpu',weights_only=True)
    if (not isinstance(p,dict) or p.get('contract')!=projection_contract()
            or not _valid_digest(protocol_sha256) or p.get('protocol_sha256')!=protocol_sha256):
        raise ValueError('projection checkpoint contract differs')
    try:
        m=ProjectionReadout(mean,scale,seed=p['seed']);m.load_state_dict(p['state_dict'],strict=True)
        if (not torch.equal(m.mean,mean.float()) or not torch.equal(m.scale,scale.float())
                or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in p['state_dict'].values())):
            raise ValueError('checkpoint normalization/numerics differ')
    except (KeyError,TypeError,RuntimeError) as e:raise ValueError('invalid projection checkpoint state') from e
    m.eval();return m
