"""Carry the IMU home envelope through the escape-to-idle handoff."""
import math
import numpy as np
from flybrain.flight3d.contracts import FlightCommand, FlightObservation
from flybrain.retreat3d.controller import IdleFlight, stabilize
from flybrain.turnflee.control import EscapeBehavior


class BoundedIdleFlight(IdleFlight):
    """Limit outward velocity directly, without waiting for heading to turn.

    Displacement is integrated from the fly's IMU relative to its calibrated
    start, never supplied by the world. Beyond 0.22 m, command inward motion;
    the 2/s servo leaves room for inertia within the centered 0.6 m arena.
    """

    def command(self, observation: FlightObservation, *, update: bool = True) -> FlightCommand:
        command = super().command(observation, update=update)
        e = self.estimate
        radius = float(np.linalg.norm(e.displacement[:2]))
        if self.state != 'cruising' or radius <= .18:
            return command
        radial = e.displacement[:2] / radius
        desired = np.array((.06 * math.sin(self.heading), .06 * math.cos(self.heading),
                            2 * (.1 - e.altitude)))
        outward = float(desired[:2] @ radial)
        limit = float(np.clip(2 * (.22 - radius), -.06, .06))
        if outward > limit:
            desired[:2] -= (outward - limit) * radial
        return stabilize(e, desired, command.yaw, command.legs)


class SafeEscapeBehavior(EscapeBehavior):
    """Retain neural side/turn/flee/braking and replace only idle translation."""

    def reset(self) -> None:
        super().reset()
        self.idle = BoundedIdleFlight(self.seed)
