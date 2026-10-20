"""Explicit alternative engineered dynamics at BANC projection neurons only."""

import math

import torch
from torch import Tensor

from .lif import LIFConfig, LIFPopulation


class ProjectionChargePopulation(LIFPopulation):
    """PN-only tau=25 ms, threshold=1.1 and subtractive reset.

    These are assumptions, not sourced physiology. Other cells keep the base
    LIF rules. At most one spike occurs per step; excess above threshold remains
    available on later steps. Charge recording observes the actual transition.
    """

    def __init__(self, weights: Tensor, projection_indices: tuple[int, ...], *,
                 record_charge: bool = False) -> None:
        super().__init__(weights, LIFConfig(), record_charge=record_charge)
        if (not projection_indices or len(set(projection_indices)) != len(projection_indices)
                or any(type(i) is not int or not 0 <= i < self.size for i in projection_indices)):
            raise ValueError('projection indices must be unique valid neuron indices')
        self._projection_mask = torch.zeros(self.size, dtype=torch.bool, device='cpu')
        self._projection_mask[list(projection_indices)] = True
        self._thresholds = torch.full((self.size,), self.config.threshold, dtype=self.dtype, device='cpu')
        self._thresholds[self._projection_mask] = 1.1
        self._decay = torch.full((self.size,), self._decay, dtype=self.dtype, device='cpu')
        self._input_fraction = torch.full((self.size,), self._input_fraction, dtype=self.dtype, device='cpu')
        self._decay[self._projection_mask] = math.exp(-self.config.dt / .025)
        self._input_fraction[self._projection_mask] = -math.expm1(-self.config.dt / .025)

    def _detect_spikes(self, candidate: Tensor) -> Tensor:
        return candidate >= self._thresholds

    def _reset_voltage(self, candidate: Tensor, spikes: Tensor) -> Tensor:
        hard_reset = super()._reset_voltage(candidate, spikes)
        return torch.where(self._projection_mask, candidate - spikes * self._thresholds, hard_reset)


class ProjectionFilteredPopulation(LIFPopulation):
    """Spread PN input jumps in time; retain area over the complete filter tail.

    Non-PN rows retain the original instantaneous jumps and hard-reset LIF.
    This temporal assumption is not provided by the BANC connectome.
    """

    def __init__(self, weights: Tensor, projection_indices: tuple[int, ...],
                 config: LIFConfig | None = None) -> None:
        if (not projection_indices or len(set(projection_indices)) != len(projection_indices)
                or any(type(i) is not int or not 0 <= i < weights.shape[0] for i in projection_indices)):
            raise ValueError('projection indices must be unique valid neuron indices')
        self._projection_indices = torch.tensor(projection_indices, dtype=torch.int64, device='cpu')
        super().__init__(weights, config)
        self._filter_decay = math.exp(-self.config.dt / .005)

    @property
    def synaptic_trace(self) -> Tensor:
        """Copy of current filtered jumps at the selected PNs, in index order."""
        return self._filter_state.clone()

    def reset(self) -> None:
        """Clear voltage, delayed spikes, time and every synaptic-filter state."""
        super().reset()
        self._filter_state = torch.zeros(len(self._projection_indices), dtype=self.dtype, device='cpu')
        self._pending_filter: Tensor | None = None

    def _synaptic_input(self) -> Tensor:
        jumps = super()._synaptic_input()
        self._pending_filter = (self._filter_decay * self._filter_state
                                + (1 - self._filter_decay) * jumps[self._projection_indices])
        jumps[self._projection_indices] = self._pending_filter
        return jumps

    def step(self, current: Tensor) -> Tensor:
        """Commit the filter only after the actual LIF transition succeeds."""
        try:
            spikes = super().step(current)
            self._filter_state = self._pending_filter
            return spikes
        finally:
            self._pending_filter = None
