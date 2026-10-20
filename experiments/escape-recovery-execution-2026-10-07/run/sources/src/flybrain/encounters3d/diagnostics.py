"""Passive substep measurements of the fixed, assumed source-circuit dynamics.

Current units are normalized voltage increments per 1 ms, not physical amperes.
The original flight kernel remains untouched as the independent parity oracle.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from flybrain.flight3d.neural import FlightBrain
from .neural import EscapeBrain


@torch.jit.script
def _advance_measured(voltage: torch.Tensor, previous: torch.Tensor, phase: torch.Tensor,
                      rates: torch.Tensor, pre: torch.Tensor, post: torch.Tensor,
                      weights: torch.Tensor, inputs: torch.Tensor, live: torch.Tensor,
                      steps: int):
    counts = torch.zeros_like(voltage, dtype=torch.int64)
    delivered = torch.zeros_like(weights)
    effective = (~inputs[post]) & live[post]
    current_sum = torch.zeros_like(voltage)
    current_peak = torch.zeros_like(voltage)
    discarded = torch.zeros_like(voltage)
    peak_voltage = torch.full_like(voltage, -float('inf'))
    peak_phase = torch.zeros_like(voltage)
    first = torch.full_like(counts, -1)
    for step in range(steps):
        # Original operations, in their original order and precision.
        voltage = voltage * 0.9900498337491681
        contribution = weights * previous[pre]
        voltage.index_add_(0, post, contribution)
        delivered = delivered + contribution * effective
        phase = phase + rates * .001
        # Observe before threshold/reset. These tensors never feed state updates.
        incoming = torch.zeros_like(voltage)
        incoming.index_add_(0, post, contribution * effective)
        lost = torch.zeros_like(voltage)
        lost.index_add_(0, post, contribution * (~effective))
        current_sum = current_sum + incoming
        current_peak = torch.maximum(current_peak, incoming)
        discarded = discarded + lost
        peak_voltage = torch.maximum(peak_voltage, torch.where(
            inputs | ~live, torch.zeros_like(voltage), voltage))
        peak_phase = torch.maximum(peak_phase, torch.where(
            inputs & live, phase, torch.zeros_like(phase)))
        forced = phase >= 1.
        phase = torch.where(forced, phase - 1., phase)
        spikes = torch.where(inputs, forced, voltage >= 1.) & live
        voltage = torch.where(inputs | ~live, torch.zeros_like(voltage),
                              voltage - spikes.to(voltage.dtype))
        previous = spikes.to(voltage.dtype)
        counts = counts + spikes.to(torch.int64)
        first = torch.where(spikes & (first < 0), torch.full_like(first, step + 1), first)
    return (voltage, previous, phase, counts, delivered, current_sum, current_peak,
            discarded, peak_voltage, peak_phase, first)


@dataclass(frozen=True)
class DiagnosticInterval:
    """One immutable interval in manifest order; first times use the 1 ms grid."""
    start_seconds: float
    end_seconds: float
    current_sum: tuple[float, ...]
    current_peak: tuple[float, ...]
    discarded_current: tuple[float, ...]
    peak_voltage: tuple[float, ...]
    peak_phase: tuple[float, ...]
    spike_counts: tuple[int, ...]
    first_spike_seconds: tuple[float | None, ...]
    episode_first_spike_seconds: tuple[float | None, ...]
    silenced: tuple[str, ...]


class DiagnosticMixin:
    """Identical stepping plus interval publication and a five-second trace ring.

    Every completed interval is measured. Investigation streams every record to
    disk; the interactive viewer retains the last 250 intervals to bound memory.
    No gradients, optimizer, new connections or altered sensory assignments.
    """
    def reset(self) -> None:
        super().reset()
        self.diagnostic: DiagnosticInterval | None = None
        self.diagnostic_intervals: list[DiagnosticInterval] = []
        self._first_spike: list[float | None] = [None] * len(self.ids)

    @torch.no_grad()
    def advance_rates(self, rates: np.ndarray, steps: int = 20) -> np.ndarray:
        """Measure effective/discarded current and peaks without feeding them back."""
        values = np.asarray(rates, dtype=float)
        if values.shape != (len(self.ids),) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 240):
            raise ValueError('one finite 0..240 Hz input rate per neuron required')
        if type(steps) is not int or steps <= 0:
            raise ValueError('positive integer neural steps required')
        live = torch.tensor([i not in self.silenced for i in self.ids])
        result = _advance_measured(self.voltage, self.previous, self.phase,
            torch.from_numpy(values), self.pre, self.post, self.weights, self.inputs, live, steps)
        self.voltage, self.previous, self.phase, counts, delivered = result[:5]
        start = self.steps * .001
        self.steps += steps
        elapsed = steps * .001
        for k, tau in enumerate((.1, .5)):
            alpha = -math.expm1(-elapsed / tau)
            self.history[k] = (1-alpha)*self.history[k] + alpha*counts.double()/elapsed
            self.history[k, ~live] = 0.
        self.last_counts = counts.numpy().copy()
        totals = np.zeros(len(self.edge_order))
        totals[self.dynamic_edges] = delivered.numpy()
        self.edge_transmissions = tuple(float(v) for v in totals)
        first = tuple((int(step) + round(start/.001)) * .001 if step >= 0 else None
                      for step in result[-1].tolist())
        for i, time in enumerate(first):
            if self._first_spike[i] is None: self._first_spike[i] = time
        self.diagnostic = DiagnosticInterval(start, self.steps * .001,
            *(tuple(float(v) for v in tensor.tolist()) for tensor in result[5:10]),
            tuple(int(v) for v in self.last_counts), first, tuple(self._first_spike),
            tuple(sorted(self.silenced)))
        self.diagnostic_intervals.append(self.diagnostic)
        if len(self.diagnostic_intervals) > 250: del self.diagnostic_intervals[0]
        return self.last_counts.copy()


class DiagnosticEscapeBrain(DiagnosticMixin, EscapeBrain):
    """Unchanged predator encoder with passive measurements."""


class DiagnosticFlightBrain(DiagnosticMixin, FlightBrain):
    """Unchanged flower brain with passive measurements."""


def stage_indices(manifest: dict) -> dict[str, tuple[int, ...]]:
    """Display membership is metadata/roles, independent of neural activity."""
    nodes = manifest['nodes']
    return {'Visual neurons': tuple(i for i, n in enumerate(nodes) if 'vision' in n['roles']),
            'DNp03': tuple(i for i, n in enumerate(nodes) if n['metadata'].get('cell_type') == 'DNp03'),
            'DNp26': tuple(i for i, n in enumerate(nodes) if n['metadata'].get('cell_type') == 'DNp26'),
            'Wing-control neurons': tuple(i for i, n in enumerate(nodes)
                if set(n['roles']) & {'wing_power', 'wing_steering'})}


def retained_paths(manifest: dict, starts: set[str], silenced: set[str] | None = None) -> set[str]:
    """Reachability of retained non-self edges, not a guarantee of electrical drive."""
    blocked = silenced or set()
    adjacency = {n['id']: [] for n in manifest['nodes']}
    for e in manifest['edges']:
        if e['pre'] != e['post'] and e['post'] not in blocked:
            adjacency[e['pre']].append(e['post'])
    found = starts - blocked
    pending = list(found)
    while pending:
        for post in adjacency[pending.pop()]:
            if post not in found:
                found.add(post); pending.append(post)
    return found
