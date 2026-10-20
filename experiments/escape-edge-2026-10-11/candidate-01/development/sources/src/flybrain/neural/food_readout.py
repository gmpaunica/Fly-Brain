"""Fixed 85-cell activity into a trainable 22-to-16-to-3 motor readout."""

import hashlib
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from flybrain.world.fly import MotorCommand

from .banc_memory import FAST_TAU, SLOW_TAU
from .banc_motor import BancMotorDecoder
from .banc_projection import ProjectionActivity, projection_contract, projection_features
from .controller import NeuralController
from .food_contact import FoodContactNetwork


def food_features(rates: Tensor) -> Tensor:
    """Condition the old 20 histories, then append two relay spike histories."""
    if (not isinstance(rates, Tensor) or rates.ndim not in (1, 2)
            or rates.shape[-1] != 22 or rates.numel() == 0
            or rates.device.type != 'cpu' or not rates.is_floating_point()
            or not torch.isfinite(rates).all() or torch.any(rates < 0)
            or torch.any(rates > 1000)):
        raise ValueError('expected finite nonnegative [22] or [N,22] neural rates')
    return torch.cat((projection_features(rates[..., :20]), rates[..., 20:]), dim=-1)


def food_normalization(rates: Tensor) -> tuple[Tensor, Tensor]:
    """Fit feature mean and population standard deviation on training rows."""
    if not isinstance(rates, Tensor) or rates.ndim != 2:
        raise ValueError('training matrix required')
    features = food_features(rates.double())
    return (features.mean(0).float(),
            features.std(0, correction=0).clamp_min(1.).float())


class FoodReadout(nn.Module):
    """Only 419 parameters learn; the fixed circuit has no gradient path."""

    def __init__(self, mean: Tensor, scale: Tensor, *, seed: int = 1010) -> None:
        super().__init__()
        if (type(seed) is not int or seed < 0
                or any(not isinstance(t, Tensor) or t.shape != (22,)
                       or t.device.type != 'cpu' or not t.is_floating_point()
                       or not torch.isfinite(t).all() for t in (mean, scale))
                or torch.any(scale < 1)):
            raise ValueError('22 finite CPU normalization values and scale >=1 required')
        self.seed = seed
        self.register_buffer('mean', mean.detach().float().clone())
        self.register_buffer('scale', scale.detach().float().clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden = nn.Linear(22, 16)
            self.output = nn.Linear(16, 3)

    def forward(self, rates: Tensor) -> Tensor:
        """Speed fraction, signed turn and unconstrained stop logit."""
        values = (food_features(rates).float() - self.mean) / self.scale
        raw = self.output(torch.tanh(self.hidden(values)))
        return torch.stack((torch.sigmoid(raw[..., 0]),
                            torch.tanh(raw[..., 1]), raw[..., 2]), dim=-1)


class FoodDecoder:
    """Causal endpoint, PN-event and contact-relay spike-rate histories."""

    def __init__(self, topology, model: FoodReadout) -> None:
        self.fast = BancMotorDecoder(topology)
        self.slow = BancMotorDecoder(topology)
        self.slow.filter_tau_seconds = SLOW_TAU
        self.model = model
        self.feature_names = tuple(food_contract()['feature_names'])
        self.reset()

    @property
    def feature_rates(self) -> Tensor:
        """Detached raw 22-rate snapshot without advancing any state."""
        return torch.cat((self.fast.feature_rates, self.slow.feature_rates,
                          self.pn_fast, self.pn_slow,
                          self.contact_fast, self.contact_slow))

    def reset(self) -> None:
        """Clear all filters at an episode boundary."""
        self.fast.reset()
        self.slow.reset()
        self.pn_fast = torch.zeros(2, dtype=torch.float64, device='cpu')
        self.pn_slow = torch.zeros(2, dtype=torch.float64, device='cpu')
        self.contact_fast = torch.zeros(1, dtype=torch.float64, device='cpu')
        self.contact_slow = torch.zeros(1, dtype=torch.float64, device='cpu')

    def observe(self, activity: ProjectionActivity) -> Tensor:
        """Update filters once from actual interval spikes and PN events."""
        if (not isinstance(activity, ProjectionActivity)
                or activity.spike_counts.shape != (85,)
                or activity.projection_event_rates is None
                or activity.projection_event_rates.shape != (2,)):
            raise ValueError('85-cell measured projection activity required')
        self.fast.decode(activity)
        self.slow.decode(activity)
        pn = activity.projection_event_rates.double()
        relay = activity.spike_counts[FoodContactNetwork.RELAY].double().reshape(1) / activity.elapsed_seconds
        for name, value, tau in (('pn_fast', pn, FAST_TAU),
                                 ('pn_slow', pn, SLOW_TAU),
                                 ('contact_fast', relay, .05),
                                 ('contact_slow', relay, .2)):
            alpha = -math.expm1(-activity.elapsed_seconds / tau)
            setattr(self, name, (1 - alpha) * getattr(self, name) + alpha * value)
        return self.feature_rates

    def decode(self, activity: ProjectionActivity) -> MotorCommand:
        """Apply the frozen learned head and issue exactly zero for positive stop."""
        rates = self.observe(activity)
        with torch.no_grad():
            output = self.model(rates)
        if float(output[2]) > 0:
            return MotorCommand()
        return MotorCommand(float(output[0]), float(output[1]))


def make_food_controller(model: FoodReadout) -> NeuralController:
    """Construct independent neural/filter state around a shared frozen head."""
    network = FoodContactNetwork()
    return NeuralController(network, FoodDecoder(network.topology, model))


def food_contract() -> dict:
    """Pin feature order, engineered pathway and fixed effective graph."""
    network = FoodContactNetwork()
    source = projection_contract()
    return {'schema': 'food_stop_readout_v1',
            'source_subset_sha256': network.source_topology.source.get('subset_sha256'),
            'source_ids': list(network.source_topology.root_ids),
            'engineered_ids': list(network.topology.root_ids[83:]),
            'effective_weights_sha256': hashlib.sha256(
                network.population.weights.numpy().tobytes()).hexdigest(),
            'feature_names': source['feature_names'] + [
                'fast50ms:engineered:contact-relay',
                'slow200ms:engineered:contact-relay'],
            'architecture': {'inputs': 22, 'hidden': 16, 'outputs': 3,
                             'parameters': 419, 'stop_rule': 'logit > 0 -> exact zero'},
            'odor_rate': '96 * concentration / (concentration + 0.1) Hz',
            'contact_rate_hz': 100., 'contact_relay_jump': 2.,
            'feature_signal': 'actual spikes and PN incoming events only; no raw odor or geometry'}


def save_food_checkpoint(path: Path, model: FoodReadout, *, protocol_sha256: str) -> None:
    """Save one finite, contract-bound checkpoint without overwriting."""
    if (len(protocol_sha256) != 64 or any(ch not in '0123456789abcdef' for ch in protocol_sha256)
            or any(not torch.isfinite(value).all() for value in model.state_dict().values())):
        raise ValueError('finite state and protocol SHA256 required')
    with Path(path).open('xb') as stream:
        torch.save({'contract': food_contract(), 'seed': model.seed,
                    'protocol_sha256': protocol_sha256,
                    'state_dict': model.state_dict()}, stream)


def load_food_checkpoint(path: Path, *, protocol_sha256: str,
                         mean: Tensor, scale: Tensor) -> FoodReadout:
    """Reject stale graph, feature schema, normalization and nonfinite weights."""
    record = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (not isinstance(record, dict) or record.get('contract') != food_contract()
            or record.get('protocol_sha256') != protocol_sha256):
        raise ValueError('food checkpoint contract differs')
    try:
        model = FoodReadout(mean, scale, seed=record['seed'])
        model.load_state_dict(record['state_dict'], strict=True)
        if (not torch.equal(model.mean, mean.float())
                or not torch.equal(model.scale, scale.float())
                or any(value.dtype != torch.float32 or not torch.isfinite(value).all()
                       for value in record['state_dict'].values())):
            raise ValueError('food checkpoint normalization or numerics differ')
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError('food checkpoint state invalid') from exc
    model.eval()
    return model
