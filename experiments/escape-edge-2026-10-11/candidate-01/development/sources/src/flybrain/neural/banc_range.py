"""Fixed arena-wide sensory range, with unchanged BANC neurons and wiring."""

import math
from pathlib import Path

import torch

from flybrain.data.banc import load_banc_subset
from .banc_network import load_verified_banc_config
from .banc_projection import ProjectionNetwork,ProjectionDecoder,ProjectionReadout,projection_contract
from .banc_motor import BancMotorDecoder
from .banc_readout import _valid_digest
from .controller import NeuralController


JOIN=.03
MAX_ODOR=2.5
JOIN_RATE=176*math.sqrt(JOIN)
LOG_GAIN=(96-JOIN_RATE)/math.log(MAX_ODOR/JOIN)


def sensory_rate(concentration: float) -> float:
    """Engineered monotone sqrt/log map, fixed from low band and world range.

    Preserve 176*sqrt(c) through .03, then logarithmically cover [.03,2.5]
    up to96Hz. This is fixed scaffolding, not a fitted/biological receptor.
    """
    if not math.isfinite(concentration) or concentration<0:raise ValueError('finite nonnegative odor required')
    if concentration<=JOIN:return 176*math.sqrt(concentration)
    if concentration>=MAX_ODOR:return 96.
    return JOIN_RATE+LOG_GAIN*math.log(concentration/JOIN)


class RangeNetwork(ProjectionNetwork):
    """Change only sensory scheduling rates; reuse all original actual LIF steps."""

    def _sensory_rates(self,observation):
        rates=torch.zeros(self.population.size,dtype=torch.float64,device='cpu')
        for side,value in zip(('left','right'),(observation.odor_left,observation.odor_right)):
            rates[list(self._scales[side][0])]=sensory_rate(value)
        return rates


def make_range_controller(model: ProjectionReadout | None=None) -> NeuralController:
    """Same revised encoder/core for trained, matched untrained and fixed controls."""
    brain=RangeNetwork(load_banc_subset(connectome_gain=1.),dynamics_config=load_verified_banc_config())
    return NeuralController(brain,BancMotorDecoder(brain.topology) if model is None else ProjectionDecoder(brain.topology,model))


def range_contract() -> dict:
    """Bind revised encoder explicitly; the old dynamics receipt covers the core."""
    return {**projection_contract(),'schema':'banc_range_readout_v1',
        'encoder_revision':{'rule':'176*sqrt(c) through .03, then r0+(96-r0)*log(c/.03)/log(2.5/.03); cap96 above2.5',
            'join_concentration':JOIN,'arena_max_concentration':MAX_ODOR,'join_rate_hz':JOIN_RATE,'log_gain_hz':LOG_GAIN,
            'assumption':'fixed engineered sensory scaffolding; original phase placement/pulses, wiring and all neuron rules unchanged'}}


def save_checkpoint(path: Path,model: ProjectionReadout,*,protocol_sha256: str) -> None:
    if not _valid_digest(protocol_sha256) or any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('finite state and protocol SHA256 required')
    with Path(path).open('xb') as stream:
        torch.save({'contract':range_contract(),'seed':model.seed,'protocol_sha256':protocol_sha256,'state_dict':model.state_dict()},stream)


def load_checkpoint(path: Path,*,protocol_sha256: str,mean,scale) -> ProjectionReadout:
    p=torch.load(Path(path),map_location='cpu',weights_only=True)
    if not isinstance(p,dict) or p.get('contract')!=range_contract() or not _valid_digest(protocol_sha256) or p.get('protocol_sha256')!=protocol_sha256:
        raise ValueError('range checkpoint contract differs')
    try:
        m=ProjectionReadout(mean,scale,seed=p['seed']);m.load_state_dict(p['state_dict'],strict=True)
        if (not torch.equal(m.mean,mean.float()) or not torch.equal(m.scale,scale.float())
                or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in p['state_dict'].values())):
            raise ValueError('range checkpoint normalization/numerics differ')
    except (KeyError,TypeError,RuntimeError) as e:raise ValueError('invalid range checkpoint') from e
    m.eval();return m
