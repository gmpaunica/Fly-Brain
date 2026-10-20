"""Trainable endpoint-only motor readout above the frozen sourced BANC circuit."""

from dataclasses import asdict
import hashlib
from importlib.resources import files
from pathlib import Path

import torch
from torch import Tensor, nn

from flybrain.data.banc import BANC_SUBSET_SHA256, BANC_TRANSFORM_VERSION
from flybrain.world.fly import MotorCommand

from .banc_motor import BancMotorDecoder, make_banc_controller
from .banc_network import load_verified_banc_config
from .controller import NeuralController
from .network import NeuralActivity


ARCHITECTURE = {'inputs': 8, 'hidden': 16, 'outputs': 2,
                'hidden_activation': 'tanh', 'outputs_activation': ['sigmoid', 'tanh']}


class BancReadout(nn.Module):
    """178 learned scalars; fixed train-only mean/scale buffers; CPU float32.

    PyTorch linear weights use W[destination, source]. Features are eight
    actual endpoint rates in Hz. No neuron, encoder or graph is trainable here.
    """

    def __init__(self, mean: Tensor, scale: Tensor, *, seed: int = 1010) -> None:
        super().__init__()
        if (type(seed) is not int or seed < 0
                or any(not isinstance(t, Tensor) or t.device.type != 'cpu'
                       or t.shape != (8,) or not t.is_floating_point()
                       or not torch.isfinite(t).all() for t in (mean, scale))
                or torch.any(mean < 0) or torch.any(mean > 1000) or torch.any(scale < 1)):
            raise ValueError('readout requires finite eight-rate mean and scale >= 1 Hz')
        self.seed = seed
        self.register_buffer('mean', mean.detach().to(dtype=torch.float32).clone())
        self.register_buffer('scale', scale.detach().to(dtype=torch.float32).clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden = nn.Linear(8, 16, device='cpu', dtype=torch.float32)
            self.output = nn.Linear(16, 2, device='cpu', dtype=torch.float32)

    def forward(self, rates: Tensor) -> Tensor:
        """Map [8] or [N,8] endpoint rates to bounded forward/turn commands."""
        if (not isinstance(rates, Tensor) or rates.device.type != 'cpu'
                or rates.layout != torch.strided or not rates.is_floating_point()
                or rates.ndim not in (1, 2) or rates.shape[-1] != 8 or rates.numel() == 0
                or not torch.isfinite(rates).all() or torch.any(rates < 0)
                or torch.any(rates > 1000)):
            raise ValueError('expected finite CPU [8] or [N,8] endpoint rates in [0,1000] Hz')
        normalized = (rates.to(dtype=torch.float32) - self.mean) / self.scale
        raw = self.output(torch.tanh(self.hidden(normalized)))
        return torch.stack((torch.sigmoid(raw[..., 0]), torch.tanh(raw[..., 1])), dim=-1)


class BancReadoutDecoder:
    """Reuse the accepted endpoint EMA exactly, then apply the learned readout."""

    def __init__(self, topology, model: BancReadout) -> None:
        self.feature_filter = BancMotorDecoder(topology)
        self.model = model
        self.feature_names = self.feature_filter.feature_names

    @property
    def feature_rates(self) -> Tensor:
        """Detached copy of the eight actual endpoint EMA rates."""
        return self.feature_filter.feature_rates

    def reset(self) -> None:
        """Clear only episode filter state; preserve learned parameters."""
        self.feature_filter.reset()

    def decode(self, activity: NeuralActivity) -> MotorCommand:
        """Read spikes only; the fixed decoder's proxy command is discarded."""
        self.feature_filter.decode(activity)
        with torch.no_grad():
            output = self.model(self.feature_rates)
        return MotorCommand(float(output[0]), float(output[1]))


def make_readout_controller(model: BancReadout) -> NeuralController:
    """Create independent accepted neural/filter state for one live episode."""
    controller = make_banc_controller(load_verified_banc_config())
    return NeuralController(controller.network, BancReadoutDecoder(controller.network.topology, model))


def readout_contract() -> dict:
    """Bind source bytes, accepted dynamics, effective weights and endpoint IDs."""
    config = load_verified_banc_config()
    controller = make_banc_controller(config)
    package = files('flybrain.data')
    weights = controller.network.population.weights.detach().cpu().contiguous()
    return {
        'schema': 'banc_endpoint_readout_v1', 'architecture': ARCHITECTURE,
        'subset_sha256': BANC_SUBSET_SHA256, 'transform_version': BANC_TRANSFORM_VERSION,
        'dynamics_config': asdict(config),
        'accepted_bundle_sha256': {name: hashlib.sha256(package.joinpath(name).read_bytes()).hexdigest()
            for name in ('banc_dynamics_config.json', 'banc_dynamics_results.json',
                         'banc_dynamics_protocol.json')},
        'effective_weights_sha256': hashlib.sha256(weights.numpy().tobytes()).hexdigest(),
        'feature_names': list(controller.decoder.feature_names),
        'filter_tau_seconds': .1, 'feature_units': 'Hz', 'world_dt_seconds': .02,
        'neural_substeps_per_tick': 20, 'device': 'cpu', 'dtype': 'float32',
    }


def _valid_digest(digest: str) -> bool:
    return isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)


def save_checkpoint(path: Path, model: BancReadout, *, protocol_sha256: str) -> None:
    """Exclusively save parameters, fixed normalization and the full contract."""
    if not _valid_digest(protocol_sha256) or any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('checkpoint requires a protocol hash and finite state')
    with Path(path).open('xb') as stream:
        torch.save({'contract': readout_contract(), 'protocol_sha256': protocol_sha256,
                    'seed': model.seed, 'state_dict': model.state_dict()}, stream)


def load_checkpoint(path: Path, *, protocol_sha256: str,
                    mean: Tensor, scale: Tensor) -> BancReadout:
    """Reject incompatible contracts or normalization before replaying on CPU."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or payload.get('contract') != readout_contract()
            or not _valid_digest(protocol_sha256) or payload.get('protocol_sha256') != protocol_sha256):
        raise ValueError('checkpoint source/config/feature/protocol contract differs')
    try:
        model = BancReadout(mean, scale, seed=payload['seed'])
        model.load_state_dict(payload['state_dict'], strict=True)
        if (not torch.equal(model.mean, mean.float()) or not torch.equal(model.scale, scale.float())
                or any(t.dtype != torch.float32 or not torch.isfinite(t).all()
                       for t in payload['state_dict'].values())):
            raise ValueError('checkpoint normalization or numerical state differs')
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError('invalid BANC readout checkpoint state') from exc
    model.eval()
    return model
