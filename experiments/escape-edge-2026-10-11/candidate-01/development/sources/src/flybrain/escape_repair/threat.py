"""Separate sustained visual expansion from rectified pixel-edge jitter."""
from collections import deque
from dataclasses import replace
import numpy as np
from flybrain.flight3d.contracts import CONTROL_DT
from flybrain.retreat3d.threat import ThreatMonitor, ThreatMeasurement


class ClearanceMonitor(ThreatMonitor):
    """Retain onset exactly; clear only after 300 ms small, non-expanding image.

    Apparent size remains RGB-derived. Positive-only frame derivatives are
    useful for onset but biased by sampling noise. Clearance uses net change
    across the same continuously-small window, in degrees per second.
    """

    def reset(self) -> None:
        super().reset()
        self.clearance_sizes = deque(maxlen=15)
        self.clearance_growth = 0.

    def update(self, eyes: np.ndarray) -> ThreatMeasurement:
        value = super().update(eyes)
        if value.size_degrees <= 10:
            self.clearance_sizes.append(value.size_degrees)
        else:
            self.clearance_sizes.clear()
        self.clearance_growth = 0.
        complete = len(self.clearance_sizes) == 15
        if complete:
            self.clearance_growth = (self.clearance_sizes[-1] - self.clearance_sizes[0]) / (14 * CONTROL_DT)
        self.value = replace(value, weak_continuous=complete and self.clearance_growth <= 5)
        return self.value
