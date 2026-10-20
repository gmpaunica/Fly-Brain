"""Deterministic concentration-to-current encoding; no geometry or steering."""

import math
from typing import Protocol

import torch
from torch import Tensor

from .topology import NeuralTopology


class OdorObservation(Protocol):
    """Structural boundary containing only two raw scalar concentrations."""

    @property
    def odor_left(self) -> float: ...

    @property
    def odor_right(self) -> float: ...


class SensoryEncoder:
    """Independent linear or rational concentration-to-current mapping.

    Linear is 2*min(c/scale,1). Rational is 2.5*c/(c+scale).
    Scale and gain are arbitrary fixed scaffolding, not biological estimates.
    Output has no tonic current; the network owns that separate assumption.
    """

    def __init__(self, topology: NeuralTopology, scale: float = 0.025,
                 mode: str = 'linear') -> None:
        if (not isinstance(scale, (int, float)) or isinstance(scale, bool)
                or not math.isfinite(scale) or scale <= 0):
            raise ValueError('encoder scale must be positive and finite')
        if mode not in ('linear', 'rational'):
            raise ValueError('encoder mode must be linear or rational')
        self.scale = float(scale)
        self.mode = mode
        self._size = len(topology.labels)
        self._dtype = topology.weights.dtype
        self._left = topology.indices('sensory_left')
        self._right = topology.indices('sensory_right')

    def encode(self, observation: OdorObservation) -> Tensor:
        """Validate and encode the two values without accepting a world object."""
        values = (observation.odor_left, observation.odor_right)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError('concentrations must be finite and nonnegative')
        current = torch.zeros(self._size, dtype=self._dtype, device='cpu')
        for value, group in zip(values, (self._left, self._right)):
            if self.mode == 'linear':
                # Saturate before division so huge finite values stay finite.
                encoded = 2.0 * (min(value, self.scale) / self.scale)
            else:
                encoded = 2.5 * (value / (value + self.scale))
            current[list(group)] = encoded
        return current
