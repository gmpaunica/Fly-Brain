"""Standalone synthetic network; embodied control belongs to M4."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .lif import LIFConfig, LIFPopulation
from .sensory import OdorObservation, SensoryEncoder
from .topology import build_synthetic_topology


@dataclass(frozen=True)
class NeuralSnapshot:
    """Detached post-substep state copies; spikes describe the last step only."""

    step_count: int
    voltage: Tensor
    spikes: Tensor


@dataclass(frozen=True)
class NeuralActivity:
    """Counts over [start_step,end_step) and the resulting state snapshot.

    Step counts identify completed substeps; elapsed time is simulation seconds.
    Counts retain every spike even when the last-step spike vector is silent.
    Optional integrated_charge is a detached float64 neuron vector: interval
    membrane change plus measured leak/reset losses, divided by the number
    of substeps and the LIF input fraction. It has drive-equivalent units.
    It includes synaptic jumps where present; only sensory neurons in this
    synthetic graph have no incoming edges and thus expose external drive.
    """

    start_step: int
    end_step: int
    elapsed_seconds: float
    spike_counts: Tensor
    snapshot: NeuralSnapshot
    integrated_charge: Tensor | None = None


class SyntheticNetwork:
    """Fixed 100-neuron CPU graph with deterministic sensory and tonic drive.

    No world imports, coordinates, motor command, training or biological graph.
    Twenty 1 ms substeps match the future 20 ms world tick; this class never
    advances physics. Tonic forward current 1.3 is hand-written scaffolding.
    """

    def __init__(self, seed: int = 7, dtype: torch.dtype = torch.float32,
                 encoder_scale: float = 0.025,
                 encoder_mode: str = 'linear', *,
                 record_charge: bool = False) -> None:
        self.topology = build_synthetic_topology(seed, dtype)
        self.population = LIFPopulation(self.topology.weights, LIFConfig(),
                                        record_charge=record_charge)
        self._encoder = SensoryEncoder(self.topology, encoder_scale, encoder_mode)
        self._tonic = torch.zeros(self.population.size, dtype=dtype, device='cpu')
        self._tonic[list(self.topology.indices('motor_forward'))] = 1.3

    def encode(self, observation: OdorObservation) -> Tensor:
        """Return sensory-only current, independently encoding both antennae."""
        return self._encoder.encode(observation)

    def reset(self) -> None:
        """Clear all episode state, preserving fixed weights and encoding."""
        self.population.reset()

    def snapshot(self) -> NeuralSnapshot:
        """Copy state without stepping, sampling RNG, or retaining autograd."""
        return NeuralSnapshot(self.population.step_count,
                              self.population.voltage, self.population.spikes)

    def advance(self, observation: OdorObservation, substeps: int = 20) -> NeuralActivity:
        """Hold encoded plus tonic current for a positive substep count.

        Optional charge recording observes membrane transitions without
        changing dynamics. It never copies the encoder current into activity.
        """
        if type(substeps) is not int or substeps <= 0:
            raise ValueError('substeps must be a positive integer')
        current = self.encode(observation) + self._tonic
        start = self.population.step_count
        counts = torch.zeros(self.population.size, dtype=torch.int64, device='cpu')
        if self.population.record_charge:
            initial_voltage = self.population.voltage.double()
            losses = torch.zeros(self.population.size, dtype=torch.float64,
                                 device='cpu')
        for _ in range(substeps):
            counts += self.population.step(current).to(torch.int64)
            if self.population.record_charge:
                step_losses = self.population.charge_losses
                assert step_losses is not None
                losses += step_losses.leak + step_losses.reset
        end = self.population.step_count
        integrated_charge = None
        if self.population.record_charge:
            fraction = -math.expm1(-self.population.config.dt
                                  / self.population.config.tau)
            integrated_charge = (self.population.voltage.double() - initial_voltage
                                 + losses) / (substeps * fraction)
        return NeuralActivity(start, end, (end - start) * self.population.config.dt,
                              counts, self.snapshot(), integrated_charge)
