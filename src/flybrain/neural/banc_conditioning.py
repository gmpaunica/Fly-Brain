"""Fixed bilateral feature conditioning above the unchanged BANC spike circuit."""

from pathlib import Path

import torch
from torch import Tensor, nn

from .banc_readout import readout_contract, _valid_digest


PAIRS = ((0, 1), (2, 5), (3, 6), (4, 7))
CONDITIONING = {'version': 'bilateral-common-contrast-v1',
                'pairs': [list(pair) for pair in PAIRS],
                'order': 'four pair means, then four right-minus-left differences',
                'normalization': 'training population mean/std; floor 1 Hz'}


def contrast_features(rates: Tensor) -> Tensor:
    """Re-express [8] or [N,8] endpoint rates as means and signed contrasts.

    This invertible arithmetic transform uses the same eight spike-rate inputs.
    It supplies no sensory measurements, labels or world geometry.
    """
    if (not isinstance(rates, Tensor) or rates.device.type != 'cpu'
            or rates.layout != torch.strided or not rates.is_floating_point()
            or rates.ndim not in (1, 2) or rates.shape[-1] != 8 or not rates.numel()
            or not torch.isfinite(rates).all() or torch.any(rates < 0)
            or torch.any(rates > 1000)):
        raise ValueError('expected finite CPU [8] or [N,8] rates in [0,1000] Hz')
    means = [(rates[..., left] + rates[..., right]) * .5 for left, right in PAIRS]
    differences = [rates[..., right] - rates[..., left] for left, right in PAIRS]
    return torch.stack((*means, *differences), dim=-1)


def normalization(training_rates: Tensor) -> tuple[Tensor, Tensor]:
    """Freeze eight train-only conditioned population means/stds in float32."""
    if not isinstance(training_rates, Tensor) or training_rates.ndim != 2:
        raise ValueError('normalization requires a training sample matrix')
    values = contrast_features(training_rates.double())
    return values.mean(0).float(), values.std(0, correction=0).clamp_min(1).float()


class ConditionedBancReadout(nn.Module):
    """178 learned scalars; fixed conditioning and train-only normalization."""

    def __init__(self, mean: Tensor, scale: Tensor, *, seed: int = 1010) -> None:
        super().__init__()
        if (type(seed) is not int or seed < 0
                or any(not isinstance(t, Tensor) or t.device.type != 'cpu'
                       or t.shape != (8,) or not t.is_floating_point()
                       or not torch.isfinite(t).all() for t in (mean, scale))
                or torch.any(mean.abs() > 1000) or torch.any(mean[:4] < 0)
                or torch.any(scale < 1)):
            raise ValueError('expected finite conditioned means and scales >=1 Hz')
        self.seed = seed
        self.register_buffer('mean', mean.detach().float().clone())
        self.register_buffer('scale', scale.detach().float().clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden = nn.Linear(8, 16, device='cpu', dtype=torch.float32)
            self.output = nn.Linear(16, 2, device='cpu', dtype=torch.float32)

    def forward(self, rates: Tensor) -> Tensor:
        """Calculate bounded commands from current actual endpoint EMA rates."""
        normalized = (contrast_features(rates).float() - self.mean) / self.scale
        raw = self.output(torch.tanh(self.hidden(normalized)))
        return torch.stack((torch.sigmoid(raw[..., 0]), torch.tanh(raw[..., 1])), dim=-1)


def conditioned_contract() -> dict:
    """Retain the original source/feature contract and identify the fixed transform."""
    return {**readout_contract(), 'schema': 'banc_conditioned_readout_v1',
            'conditioning': CONDITIONING}


def save_checkpoint(path: Path, model: ConditionedBancReadout, *, protocol_sha256: str) -> None:
    """Exclusively save readout state and source/conditioning/protocol contracts."""
    if not _valid_digest(protocol_sha256) or any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('checkpoint requires finite state and a protocol digest')
    with Path(path).open('xb') as stream:
        torch.save({'contract': conditioned_contract(), 'protocol_sha256': protocol_sha256,
                    'seed': model.seed, 'state_dict': model.state_dict()}, stream)


def load_checkpoint(path: Path, *, protocol_sha256: str,
                    mean: Tensor, scale: Tensor) -> ConditionedBancReadout:
    """Reject wrong circuit, transform, feature order, normalization or protocol."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or payload.get('contract') != conditioned_contract()
            or not _valid_digest(protocol_sha256) or payload.get('protocol_sha256') != protocol_sha256):
        raise ValueError('checkpoint circuit/conditioning/protocol contract differs')
    try:
        model = ConditionedBancReadout(mean, scale, seed=payload['seed'])
        model.load_state_dict(payload['state_dict'], strict=True)
        if (not torch.equal(model.mean, mean.float()) or not torch.equal(model.scale, scale.float())
                or any(t.dtype != torch.float32 or not torch.isfinite(t).all()
                       for t in payload['state_dict'].values())):
            raise ValueError('checkpoint normalization or numerical state differs')
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError('invalid conditioned checkpoint state') from error
    model.eval()
    return model
