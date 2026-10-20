"""Display-only history of completed intervals, separate from neural dynamics."""
from dataclasses import dataclass
import math
import numpy as np
from .contracts import FlightSnapshot

RATE_TAU_SECONDS = .200
PULSE_IDLE_SECONDS = .600


@dataclass(frozen=True)
class DisplayFrame:
    """One immutable publication in the scientific snapshot's neuron order."""
    snapshot: FlightSnapshot
    episode_id: int
    step_index: int
    cumulative_counts: tuple[int, ...]
    last_active_times: tuple[float | None, ...]
    firing_rates_hz: tuple[float, ...]


class SpikeRecorder:
    """Observe completed control intervals exactly once; never advance a session.

    Counts aggregate the neural steps in each interval. Last activity denotes
    its end, not a measured spike location or subinterval spike timestamp.
    Only the simulation owner calls observe/reset; readers use frozen frames.
    """
    def __init__(self, snapshot: FlightSnapshot):
        self._episode_id = -1
        self.reset(snapshot)

    @staticmethod
    def _validate(snapshot: FlightSnapshot) -> None:
        if not math.isfinite(snapshot.time) or snapshot.time < 0:
            raise ValueError('finite nonnegative simulation time required')
        if len(snapshot.neuron_ids) != len(snapshot.spike_counts):
            raise ValueError('spike counts must match neuron order')
        if len(set(snapshot.neuron_ids)) != len(snapshot.neuron_ids):
            raise ValueError('unique neuron IDs required')
        if any(type(count) is not int or count < 0 for count in snapshot.spike_counts):
            raise ValueError('nonnegative integer spike counts required')

    def reset(self, snapshot: FlightSnapshot) -> DisplayFrame:
        """Start a distinct episode even when seed and simulation time repeat."""
        self._validate(snapshot)
        self._episode_id += 1
        size = len(snapshot.neuron_ids)
        self.frame = DisplayFrame(snapshot, self._episode_id, 0, (0,)*size,
                                  (None,)*size, (0.,)*size)
        return self.frame

    def observe(self, snapshot: FlightSnapshot) -> DisplayFrame:
        """Record one strictly later successful interval in spikes/s."""
        self._validate(snapshot)
        old = self.frame
        if snapshot.neuron_ids != old.snapshot.neuron_ids or snapshot.seed != old.snapshot.seed:
            raise ValueError('episode identity/order changed without reset')
        dt = snapshot.time-old.snapshot.time
        if dt <= 0:
            raise ValueError('completed intervals must have increasing time')
        decay = math.exp(-dt/RATE_TAU_SECONDS)
        counts = snapshot.spike_counts
        self.frame = DisplayFrame(
            snapshot, old.episode_id, old.step_index+1,
            tuple(total+count for total, count in zip(old.cumulative_counts, counts)),
            tuple(snapshot.time if count else last for count, last in zip(counts, old.last_active_times)),
            tuple(decay*rate+(1-decay)*count/dt for rate, count in zip(old.firing_rates_hz, counts)))
        return self.frame


def pulse_strengths(frame: DisplayFrame) -> np.ndarray:
    """Ease each last spike's display color continuously to idle over 600 ms.

    Cubic smoothstep has zero slope at both ends. A spike refreshes strength to
    one; silence fades it to zero without a residual-color cutoff. The separate
    200 ms firing-rate average and all recorded counts remain unchanged.
    """
    ages = np.array([math.inf if last is None else max(0., frame.snapshot.time-last)
                     for last in frame.last_active_times])
    fraction=np.clip(ages/PULSE_IDLE_SECONDS,0.,1.)
    # Retain the elapsed-time roundoff tolerance so invisible tails expire on time.
    fraction=np.where(ages >= PULSE_IDLE_SECONDS-1e-12,1.,fraction)
    # Factorized 1-smoothstep avoids cancellation near the idle endpoint.
    return (1.-fraction)**2*(1.+2.*fraction)
