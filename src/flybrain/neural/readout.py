"""Trainable motor readout above a frozen synthetic spiking network."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from flybrain.world.fly import MotorCommand

from .network import NeuralActivity
from .topology import SyntheticTopology


GROUP_NAMES = (
    'sensory_left', 'sensory_right', 'interneuron_left',
    'interneuron_right', 'motor_left', 'motor_forward', 'motor_right',
)


def sensory_voltage_means(activity: NeuralActivity,
                          topology: SyntheticTopology) -> Tensor:
    """Copy post-interval mean voltages for the bilateral sensory groups."""
    voltage = activity.snapshot.voltage
    if (not isinstance(voltage, Tensor) or voltage.device.type != 'cpu'
            or voltage.shape != (len(topology.labels),)
            or not voltage.is_floating_point() or not bool(torch.isfinite(voltage).all())):
        raise ValueError('voltage must be a finite CPU neuron vector')
    return torch.stack(tuple(
        voltage[list(topology.indices(group))].mean()
        for group in ('sensory_left', 'sensory_right')
    )).detach().clone()


class SpikeFeatureFilter:
    """EMA of each group's mean spikes per 20 ms neural decision interval."""

    def __init__(self, topology: SyntheticTopology, alpha: float = 0.5) -> None:
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ValueError('alpha must be a finite number in (0,1]')
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('alpha must be a finite number in (0,1]')
        self.alpha = float(alpha)
        self._size = len(topology.labels)
        self._indices = tuple(
            torch.tensor(topology.indices(name), dtype=torch.int64)
            for name in GROUP_NAMES
        )
        self._state = torch.zeros(len(GROUP_NAMES), dtype=torch.float32)

    def reset(self) -> None:
        """Forget spike history at an episode boundary."""
        self._state.zero_()

    def update(self, activity: NeuralActivity) -> Tensor:
        """Return a detached feature copy without changing neural activity."""
        counts = activity.spike_counts
        if (not isinstance(counts, Tensor) or counts.device.type != 'cpu'
                or counts.dtype != torch.int64 or counts.shape != (self._size,)
                or bool((counts < 0).any().item())):
            raise ValueError('spike_counts must be nonnegative CPU int64 neuron counts')
        means = torch.stack([
            counts[indices].to(torch.float32).mean() for indices in self._indices
        ])
        self._state = self.alpha * means + (1.0 - self.alpha) * self._state
        return self._state.clone()


class HybridFeatureFilter:
    """Seven spike EMA values plus two post-interval sensory voltage means."""

    def __init__(self, topology: SyntheticTopology, alpha: float = 0.5) -> None:
        self.topology = topology
        self.spikes = SpikeFeatureFilter(topology, alpha)

    def reset(self) -> None:
        """Forget spike history at an episode boundary."""
        self.spikes.reset()

    def update(self, activity: NeuralActivity) -> Tensor:
        """Return a detached nine-value feature vector for one decision tick."""
        return torch.cat((self.spikes.update(activity),
                          sensory_voltage_means(activity, self.topology).to(torch.float32)))


class TrainableMotorReadout(nn.Module):
    """A seven- or nine-input linear readout; only its map is trainable."""

    def __init__(self, feature_count: int = len(GROUP_NAMES)) -> None:
        super().__init__()
        if feature_count not in (7, 9) or type(feature_count) is not int:
            raise ValueError('feature_count must be seven or nine')
        self.feature_count = feature_count
        # Preserve the caller's RNG state; the declared starting map is zero.
        with torch.random.fork_rng(devices=[]):
            self.linear = nn.Linear(feature_count, 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Map (...,7/9) features to forward [0,1], clockwise turn [-1,1]."""
        if features.shape[-1:] != (self.feature_count,):
            raise ValueError('readout input must end with seven or nine features '
                             'matching its feature_count')
        raw = self.linear(features)
        return torch.stack((torch.sigmoid(raw[..., 0]), torch.tanh(raw[..., 1])), dim=-1)


class LearnedMotorDecoder:
    """Use the fixed spike graph's activity and a trained readout for commands."""

    def __init__(self, topology: SyntheticTopology, model: TrainableMotorReadout,
                 alpha: float = 0.5) -> None:
        self.feature_filter = (HybridFeatureFilter(topology, alpha)
                               if model.feature_count == 9 else
                               SpikeFeatureFilter(topology, alpha))
        self.model = model

    def reset(self) -> None:
        """Clear temporal features while retaining the learned parameters."""
        self.feature_filter.reset()

    def decode(self, activity: NeuralActivity) -> MotorCommand:
        """Produce one bounded physical command without an autograd tape."""
        features = self.feature_filter.update(activity)
        with torch.no_grad():
            output = self.model(features)
        return MotorCommand(float(output[0].item()), float(output[1].item()))
