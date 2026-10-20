"""Assumed normalized LIF dynamics, not biologically measured parameters."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class LIFConfig:
    """Seconds for dt/tau; arbitrary normalized units for other quantities.

    Current is a steady-state displacement from rest. Synaptic weights are
    end-of-step voltage jumps. Equality fires, overshoot is discarded, and
    no refractory counter is used (one spike maximum per neuron per step).
    """

    dt: float = 0.001
    tau: float = 0.020
    rest: float = 0.0
    threshold: float = 1.0
    reset: float = 0.0

    def __post_init__(self) -> None:
        for name in ('dt', 'tau', 'rest', 'threshold', 'reset'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if self.dt <= 0 or self.tau <= 0:
            raise ValueError('dt and tau must be positive seconds')
        if self.rest >= self.threshold or self.reset >= self.threshold:
            raise ValueError('rest and reset must be below threshold')


def validate_float_tensor(tensor: Tensor, shape: tuple[int, ...],
                          dtype: torch.dtype) -> None:
    """Reject inconsistent CPU tensors before a state transition."""
    if (tensor.shape != shape or tensor.dtype != dtype
            or tensor.device.type != 'cpu' or tensor.layout != torch.strided):
        raise ValueError('tensor must have matching shape/dtype and be dense CPU')
    if not torch.isfinite(tensor).all().item():
        raise ValueError('tensor must be finite')


@dataclass(frozen=True)
class ChargeLosses:
    """Float64 membrane charge lost to leak and reset in one actual substep."""

    leak: Tensor
    reset: Tensor


class LIFPopulation:
    """Synchronous fixed-weight population; all recurrence uses prior spikes.

    W[post,pre] contributes to post one substep after pre spikes. State and
    weights are privately owned; public properties and results are copies.
    This is inference-only, with no autograd route or trainable parameters.
    """

    def __init__(self, weights: Tensor, config: LIFConfig | None = None, *,
                 record_charge: bool = False) -> None:
        if type(record_charge) is not bool:
            raise ValueError('record_charge must be a bool')
        self.record_charge = record_charge
        if (weights.ndim != 2 or weights.shape[0] == 0
                or weights.shape[0] != weights.shape[1]
                or weights.dtype not in (torch.float32, torch.float64)):
            raise ValueError('weights must be a nonempty square float32/64 matrix')
        validate_float_tensor(weights, tuple(weights.shape), weights.dtype)
        self.config = config or LIFConfig()
        self.size = weights.shape[0]
        self.dtype = weights.dtype
        scalars = torch.tensor([self.config.rest, self.config.reset,
                                self.config.threshold], dtype=self.dtype, device='cpu')
        if (not torch.isfinite(scalars).all().item()
                or scalars[0] >= scalars[2] or scalars[1] >= scalars[2]):
            raise ValueError('voltage parameters must remain valid in tensor dtype')
        self._weights = weights.detach().clone()
        self._decay = math.exp(-self.config.dt / self.config.tau)
        # expm1 retains the input contribution when dt/tau is very small.
        self._input_fraction = -math.expm1(-self.config.dt / self.config.tau)
        self.reset()

    @property
    def weights(self) -> Tensor:
        """Copy of fixed synaptic jumps, indexed [post, pre]."""
        return self._weights.clone()

    @property
    def voltage(self) -> Tensor:
        """Copy of current post-reset voltages."""
        return self._voltage.clone()

    @property
    def spikes(self) -> Tensor:
        """Copy of spikes from the most recent substep."""
        return self._spikes.clone()

    @property
    def charge_losses(self) -> ChargeLosses | None:
        """Copy optional last-step losses, including discarded spike overshoot."""
        if self._charge_losses is None:
            return None
        return ChargeLosses(self._charge_losses.leak.clone(),
                            self._charge_losses.reset.clone())

    def reset(self) -> None:
        """Restore rest, clear pending spikes, and restart the step counter."""
        self._voltage = torch.full((self.size,), self.config.rest,
                                   dtype=self.dtype, device='cpu')
        self._spikes = torch.zeros(self.size, dtype=torch.bool, device='cpu')
        self.step_count = 0
        self._charge_losses: ChargeLosses | None = None

    @torch.no_grad()
    def step(self, current: Tensor) -> Tensor:
        """Advance one configured substep; reject invalid input atomically.

        candidate = rest + (v-rest)*exp(-dt/tau) + I*(1-exp(-dt/tau))
                    + W @ previous_spikes
        """
        validate_float_tensor(current, (self.size,), self.dtype)
        relaxed = (self.config.rest
                   + (self._voltage - self.config.rest) * self._decay)
        candidate = (relaxed + current * self._input_fraction
                     + self._synaptic_input())
        if not torch.isfinite(candidate).all().item():
            raise ValueError('voltage transition overflowed; state is unchanged')
        spikes = self._detect_spikes(candidate)
        voltage = self._reset_voltage(candidate, spikes)
        if self.record_charge:
            # Observe the actual rounded transition, never copy input current.
            # Double accumulation lets pure decay cancel exactly at zero drive.
            self._charge_losses = ChargeLosses(
                self._voltage.double() - relaxed.double(),
                candidate.double() - voltage.double(),
            )
        self._voltage = voltage
        self._spikes = spikes
        self.step_count += 1
        return spikes.clone()

    def _synaptic_input(self) -> Tensor:
        """Compute delayed jumps; subclasses may explicitly model input timing."""
        return self._weights @ self._spikes.to(self.dtype)

    def _detect_spikes(self, candidate: Tensor) -> Tensor:
        return candidate >= self.config.threshold

    def _reset_voltage(self, candidate: Tensor, spikes: Tensor) -> Tensor:
        return torch.where(spikes, self.config.reset, candidate)
