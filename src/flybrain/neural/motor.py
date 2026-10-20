"""Hand-written motor decoding from complete neural spike counts."""

import math

import torch

from flybrain.world.fly import MotorCommand

from .network import NeuralActivity
from .topology import SyntheticTopology


class MotorDecoder:
    """Convert motor-population counts to a bounded, smoothed command.

    One spike per neuron during a 20 ms network advance is the arbitrary
    full-scale reference. This decoder is engineering scaffolding: it is not
    learned and its scaling is not a biological measurement.
    """

    def __init__(self, topology: SyntheticTopology, smoothing: float = 0.5) -> None:
        if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)):
            raise ValueError('smoothing must be a finite number in (0,1]')
        smoothing = float(smoothing)
        if not math.isfinite(smoothing) or not 0.0 < smoothing <= 1.0:
            raise ValueError('smoothing must be a finite number in (0,1]')
        self._smoothing = smoothing
        self._size = len(topology.groups)
        self._indices = {
            group: torch.tensor(topology.indices(group), dtype=torch.int64, device='cpu')
            for group in ('motor_left', 'motor_forward', 'motor_right')
        }
        self._forward = 0.0
        self._turn = 0.0

    def reset(self) -> None:
        """Clear actuator smoothing without changing topology or configuration."""
        self._forward = 0.0
        self._turn = 0.0

    def decode(self, activity: NeuralActivity) -> MotorCommand:
        """Decode complete interval counts; reject malformed data atomically."""
        counts = activity.spike_counts
        if not isinstance(counts, torch.Tensor):
            raise ValueError('spike_counts must be a tensor')
        if counts.device.type != 'cpu':
            raise ValueError('spike_counts must be on CPU')
        if counts.dtype != torch.int64:
            raise ValueError('spike_counts must use int64 counts')
        if counts.shape != (self._size,):
            raise ValueError(f'spike_counts must have shape ({self._size},)')
        if bool((counts < 0).any().item()):
            raise ValueError('spike_counts must be nonnegative')

        def normalized(group: str) -> float:
            indices = self._indices[group]
            mean_count = float(counts[indices].sum().item()) / len(indices)
            return min(mean_count, 1.0)

        raw_forward = normalized('motor_forward')
        raw_turn = normalized('motor_right') - normalized('motor_left')
        alpha = self._smoothing
        new_forward = alpha * raw_forward + (1.0 - alpha) * self._forward
        new_turn = alpha * raw_turn + (1.0 - alpha) * self._turn
        command = MotorCommand(
            min(max(new_forward, 0.0), 1.0),
            min(max(new_turn, -1.0), 1.0),
        )
        self._forward = command.forward
        self._turn = command.turn
        return command
