"""RGB-derived apparent size only; engineering thresholds never infer range."""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np
from flybrain.encounters3d.neural import threat_contrast
from flybrain.flight3d.sensors import ELEVATION
from flybrain.flight3d.contracts import CONTROL_DT


def angular_size(eyes: np.ndarray) -> float:
    """Equivalent circular angular diameter from contrast-weighted solid angle.

    Fractional cyan occupancy uses the unchanged eight-ray RGB dilution. The
    pixel solid-angle approximation uses the existing eye sampling geometry;
    it works across both eye seams and returns no bearing or physical distance.
    Occlusion reduces apparent visible area, an explicit engineering assumption.
    """
    fraction = np.clip(threat_contrast(eyes)/.9, 0, 1)
    areas = (math.pi/48)*math.radians(160)/23*np.cos(ELEVATION)[None, :, None]
    solid_angle = float(np.sum(fraction*areas))
    return math.degrees(2*math.acos(float(np.clip(1-solid_angle/(2*math.pi), -1, 1))))


@dataclass(frozen=True)
class ThreatMeasurement:
    size_degrees: float = 0.
    positive_growth_degrees_s: float = 0.
    strong: bool = False
    weak_continuous: bool = False


class ThreatMonitor:
    """100 ms moving average, onset/growth initiation, 300 ms weak hysteresis."""
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.sizes = deque([0.]*5, maxlen=5)
        self.growth = deque([0.]*5, maxlen=5)
        self.last_raw = 0.; self.weak_steps = 0
        self.absent_steps = 5; self.appearance_pending = False
        self.value = ThreatMeasurement()

    def update(self, eyes: np.ndarray) -> ThreatMeasurement:
        raw = angular_size(eyes)
        if raw == 0:
            self.absent_steps += 1
            if self.absent_steps >= 5:
                self.appearance_pending = False
        else:
            if self.absent_steps >= 5:
                # A new large appearance can initiate once after smoothing.
                self.appearance_pending = raw >= 20
            self.absent_steps = 0
        self.sizes.append(raw)
        self.growth.append(max(0., (raw-self.last_raw)/CONTROL_DT))
        self.last_raw = raw
        size = float(np.mean(self.sizes)); growth = float(np.mean(self.growth))
        strong = size >= 20 and (self.appearance_pending or growth >= 10)
        if strong:
            self.appearance_pending = False
        self.weak_steps = self.weak_steps+1 if size <= 10 and growth <= 5 else 0
        self.value = ThreatMeasurement(size, growth, strong, self.weak_steps >= 15)
        return self.value
