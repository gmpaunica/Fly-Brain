"""Two fixed causal endpoint histories; only the small motor readout learns."""

import math
from pathlib import Path
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .banc_conditioning import contrast_features, CONDITIONING
from .banc_readout import readout_contract, _valid_digest
from .banc_motor import BancMotorDecoder, make_banc_controller
from .banc_network import load_verified_banc_config
from .controller import NeuralController
from .network import NeuralActivity
from flybrain.world.fly import MotorCommand


FAST_TAU = .1
SLOW_TAU = .5


def memory_features(rates: Tensor) -> Tensor:
    """Condition 16 rates: eight fast and eight slow original endpoint histories."""
    if not isinstance(rates, Tensor) or rates.ndim not in (1, 2) or rates.shape[-1] != 16:
        raise ValueError('expected [16] or [N,16] endpoint histories')
    return torch.cat((contrast_features(rates[..., :8]), contrast_features(rates[..., 8:])), dim=-1)


def normalization(rates: Tensor) -> tuple[Tensor, Tensor]:
    """Calculate fixed train-only conditioned mean/population std, floor 1 Hz."""
    if not isinstance(rates, Tensor) or rates.ndim != 2:
        raise ValueError('normalization requires a training matrix')
    values = memory_features(rates.double())
    return values.mean(0).float(), values.std(0, correction=0).clamp_min(1).float()


def refilter_rates(fast_rates: Tensor, case_ids: Sequence[str]) -> Tensor:
    """Recover exact counted spikes from saved fast EMAs and causally refilter.

    Validated integer counts contain the original observed endpoint events,
    not inferred sensory drive. Episode boundaries reset both histories.
    """
    contrast_features(fast_rates)
    if fast_rates.ndim != 2 or len(case_ids) != len(fast_rates):
        raise ValueError('expected episode labels for every saved rate row')
    rates = fast_rates.double()
    decay, alpha = math.exp(-.02 / FAST_TAU), -math.expm1(-.02 / FAST_TAU)
    slow_decay, slow_alpha = math.exp(-.02 / SLOW_TAU), -math.expm1(-.02 / SLOW_TAU)
    previous = torch.zeros(8, dtype=torch.float64)
    history = previous.clone(); current_case = None; result = []
    for row, case in zip(rates, case_ids):
        if case != current_case:
            previous.zero_(); history.zero_(); current_case = case
        count = (row - decay * previous) * .02 / alpha
        integer = count.round()
        if (torch.any(integer < 0) or torch.any(integer > 20)
                or not torch.allclose(count, integer, rtol=0, atol=1e-9)):
            raise ValueError('saved rates do not reconstruct valid integer interval counts')
        history = slow_decay * history + slow_alpha * integer / .02
        result.append(history.clone()); previous = row.clone()
    return torch.stack(result)


class MemoryBancReadout(nn.Module):
    """16->16->2 motor model: 306 learned weights/biases, no learned filter."""

    def __init__(self, mean: Tensor, scale: Tensor, *, seed: int = 1010) -> None:
        super().__init__()
        if (type(seed) is not int or seed < 0
                or any(not isinstance(t, Tensor) or t.device.type != 'cpu'
                       or t.shape != (16,) or not t.is_floating_point()
                       or not torch.isfinite(t).all() for t in (mean, scale))
                or torch.any(mean.abs() > 1000) or torch.any(mean[[0,1,2,3,8,9,10,11]] < 0)
                or torch.any(scale < 1)):
            raise ValueError('expected 16 finite conditioned means and scales >=1 Hz')
        self.seed = seed
        self.register_buffer('mean', mean.detach().float().clone())
        self.register_buffer('scale', scale.detach().float().clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden = nn.Linear(16, 16, device='cpu', dtype=torch.float32)
            self.output = nn.Linear(16, 2, device='cpu', dtype=torch.float32)

    def forward(self, rates: Tensor) -> Tensor:
        """Convert only current causal endpoint histories to bounded motor commands."""
        values = (memory_features(rates).float() - self.mean) / self.scale
        raw = self.output(torch.tanh(self.hidden(values)))
        return torch.stack((torch.sigmoid(raw[..., 0]), torch.tanh(raw[..., 1])), dim=-1)


class MemoryDecoder:
    """Two EMA states observing identical real endpoint spike counts."""

    def __init__(self, topology, model: MemoryBancReadout) -> None:
        self.fast = BancMotorDecoder(topology); self.slow = BancMotorDecoder(topology)
        self.slow.filter_tau_seconds = SLOW_TAU
        self.model = model
        self.feature_names = tuple(f'{tag}:{name}' for tag in ('fast100ms', 'slow500ms')
                                   for name in self.fast.feature_names)

    @property
    def feature_rates(self) -> Tensor:
        """Detached 16-rate snapshot; reading it never advances filter/neural state."""
        return torch.cat((self.fast.feature_rates, self.slow.feature_rates))

    def reset(self) -> None:
        """Reset both causal filters without changing trained weights."""
        self.fast.reset(); self.slow.reset()

    def decode(self, activity: NeuralActivity) -> MotorCommand:
        """Count only actual endpoint spikes, then use the separately trained head."""
        self.fast.decode(activity); self.slow.decode(activity)
        with torch.no_grad():
            output = self.model(self.feature_rates)
        return MotorCommand(float(output[0]), float(output[1]))


def make_memory_controller(model: MemoryBancReadout) -> NeuralController:
    """Create an independent unchanged source circuit plus temporal decoder."""
    core = make_banc_controller(load_verified_banc_config())
    return NeuralController(core.network, MemoryDecoder(core.network.topology, model))


def memory_contract() -> dict:
    """Bind both time constants, endpoint identity, conditioning and source core."""
    original = readout_contract()
    names = original['feature_names']
    return {**original, 'schema': 'banc_memory_readout_v1',
        'architecture': {'inputs':16, 'hidden':16, 'outputs':2, 'hidden_activation':'tanh',
                         'outputs_activation':['sigmoid','tanh']},
        'source_endpoint_names': names,
        'feature_names': [f'{tag}:{name}' for tag in ('fast100ms','slow500ms') for name in names],
        'filter_tau_seconds': [FAST_TAU,SLOW_TAU], 'conditioning': CONDITIONING}


def save_checkpoint(path: Path, model: MemoryBancReadout, *, protocol_sha256: str) -> None:
    """Exclusively save finite state and the complete temporal/core contract."""
    if not _valid_digest(protocol_sha256) or any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('expected finite state and protocol SHA256')
    with Path(path).open('xb') as stream:
        torch.save({'contract':memory_contract(),'seed':model.seed,
                    'protocol_sha256':protocol_sha256,'state_dict':model.state_dict()},stream)


def load_checkpoint(path: Path, *, protocol_sha256: str, mean: Tensor, scale: Tensor) -> MemoryBancReadout:
    """Reject wrong source, features, filter, normalization or protocol."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (not isinstance(payload,dict) or payload.get('contract') != memory_contract()
            or not _valid_digest(protocol_sha256) or payload.get('protocol_sha256') != protocol_sha256):
        raise ValueError('memory checkpoint contract differs')
    try:
        model = MemoryBancReadout(mean,scale,seed=payload['seed'])
        model.load_state_dict(payload['state_dict'],strict=True)
        if (not torch.equal(model.mean,mean.float()) or not torch.equal(model.scale,scale.float())
                or any(t.dtype != torch.float32 or not torch.isfinite(t).all()
                       for t in payload['state_dict'].values())):
            raise ValueError('checkpoint normalization or numerical state differs')
    except (KeyError,TypeError,RuntimeError) as error:
        raise ValueError('invalid memory checkpoint state') from error
    model.eval()
    return model
