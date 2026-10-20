"""Motor features from membrane charge, with explicit engineered compensation.

Charge balance reconstructs drive from actual LIF transitions, including leak
and discarded reset overshoot. Sensory neurons have no incoming synapses, so
their measured drive approximates the rational encoder's output. Independently
undoing that compression preserves odor magnitude across spike resets. This
does not establish any benefit from the downstream synthetic graph.

The decoder accepts neural activity only. No observation, concentration,
bilateral contrast, or teacher command is passed into it. Its feature meaning
differs from the old nine-value voltage readout and requires a new checkpoint
contract even though both models have nine inputs.
"""

import torch
from torch import Tensor

from flybrain.world.fly import MotorCommand

from .network import NeuralActivity
from .readout import GROUP_NAMES, SpikeFeatureFilter, TrainableMotorReadout
from .topology import SyntheticTopology


ENCODER_GAIN = 2.5
# Clamp each side's independent fraction Q/gain before taking its logit.
# The endpoints map to finite +/-13.8155; no bilateral operation occurs here.
CHARGE_EPSILON = 1e-6
FEATURE_NAMES = GROUP_NAMES + ('sensory_left_log_charge',
                               'sensory_right_log_charge')


class ChargeFeatureFilter:
    """Seven spike EMAs followed by two independent log charge features.

For the rational encoder, logit(Q/2.5) approximates log(c/scale).
This is fixed feature engineering; the final motor weights remain trainable.
Below the documented floor, very weak odor distinctions are discarded.
"""

    def __init__(self, topology: SyntheticTopology, alpha: float = 0.5) -> None:
        self.spikes = SpikeFeatureFilter(topology, alpha)
        self._size = len(topology.labels)
        self._sensory = tuple(
            torch.tensor(topology.indices(group), dtype=torch.int64, device='cpu')
            for group in ('sensory_left', 'sensory_right')
        )
        if any(topology.weights[indices].any().item() for indices in self._sensory):
            raise ValueError('charge features require sensory groups without incoming edges')

    def reset(self) -> None:
        """Clear the spike EMA; charge features have no additional history."""
        self.spikes.reset()

    def update(self, activity: NeuralActivity) -> Tensor:
        """Read measured neural charge and return a detached nine-value vector."""
        charge = activity.integrated_charge
        if (not isinstance(charge, Tensor) or charge.device.type != 'cpu'
                or charge.layout != torch.strided or charge.shape != (self._size,)
                or not charge.is_floating_point()
                or not torch.isfinite(charge).all().item()):
            raise ValueError('activity charge must be a finite CPU neuron vector')
        sides = torch.stack([charge[indices].double().mean()
                             for indices in self._sensory])
        fractions = (sides / ENCODER_GAIN).clamp(CHARGE_EPSILON,
                                               1.0 - CHARGE_EPSILON)
        features = (torch.log(fractions) - torch.log1p(-fractions)).float()
        return torch.cat((self.spikes.update(activity), features)).detach().clone()


class ChargeMotorDecoder:
    """Decode measured membrane charge with a separately versioned readout."""

    def __init__(self, topology: SyntheticTopology, model: TrainableMotorReadout,
                 alpha: float = 0.5) -> None:
        if model.feature_count != 9:
            raise ValueError('charge decoder requires a nine-input readout')
        self.feature_filter = ChargeFeatureFilter(topology, alpha)
        self.model = model

    def reset(self) -> None:
        """Clear temporal activity features while preserving learned weights."""
        self.feature_filter.reset()

    def decode(self, activity: NeuralActivity) -> MotorCommand:
        """Produce one bounded command using only supplied neural activity."""
        features = self.feature_filter.update(activity)
        with torch.no_grad():
            output = self.model(features)
        return MotorCommand(float(output[0].item()), float(output[1].item()))
