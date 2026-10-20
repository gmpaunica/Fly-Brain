"""IMU-based sustained escape to an authored boundary, never object geometry."""
import math
from dataclasses import replace
import numpy as np

from flybrain.flight3d.contracts import FlightCommand, FlightObservation
from flybrain.retreat3d.controller import KinematicEstimate, stabilize
from flybrain.retreat3d.threat import ThreatMeasurement
from flybrain.turnflee.control import EscapePolicy, EscapeBehavior, WingResponse

# Authored calibration: the fixed start is centered in a .6 by .6 m arena.
# These relative bounds are modeling assumptions, never runtime world inputs.
EDGE_LIMIT = .285
FLEE_SPEED = .12
ALTITUDE = .10


class EdgeEscapePolicy(EscapePolicy):
    """Latch the original spike-triggered turn until a safe edge hover.

    Neural spikes choose left/right. This hand-written servo uses an IMU
    estimate to stabilize flight and approach the authored map boundary.
    """

    def reset(self) -> None:
        super().reset()
        self.endpoint: np.ndarray | None = None
        self.braking_reason: str | None = None

    def command(self, estimate: KinematicEstimate, observation: FlightObservation,
                response: WingResponse, threat: ThreatMeasurement) -> FlightCommand | None:
        """Consume estimated self-motion, local senses and a neural side impulse."""
        if self.state == 'idle':
            # Retain the original event/impulse deadline and signed turn rule.
            command = super().command(estimate, observation, response,
                                      replace(threat, weak_continuous=False))
            if self.state == 'idle':
                return command
            direction = np.array((math.sin(self.target_heading), math.cos(self.target_heading)))
            origin = estimate.displacement[:2]
            distances = [(math.copysign(EDGE_LIMIT, d)-origin[i])/d
                         for i, d in enumerate(direction) if abs(d) > 1e-12]
            length = max(0., min(distances))
            self.endpoint = np.clip(origin+length*direction, -EDGE_LIMIT, EDGE_LIMIT)
        else:
            self.tick += 1
            self.recovered = False
            self.last_response = response
        self.home_guard = False
        e = estimate
        yaw_error = (self.target_heading-e.orientation[2]+math.pi) % math.tau-math.pi
        if self.state == 'turning' and abs(yaw_error) <= math.radians(3):
            self.state = 'fleeing'
        desired = np.zeros(3)
        desired[2] = float(np.clip(2*(ALTITUDE-e.altitude), -.06, .06))
        if self.state != 'turning':
            delta = self.endpoint-e.displacement[:2]
            distance = float(np.linalg.norm(delta))
            if self.state == 'fleeing' and distance <= FLEE_SPEED/2:
                self.state = 'edge_braking'
                self.braking_reason = 'edge_endpoint'
            if self.state == 'edge_braking' and distance <= .005 and np.linalg.norm(e.velocity[:2]) <= .01:
                self.state = 'edge_hover'
            desired[:2] = 2*delta
            speed = float(np.linalg.norm(desired[:2]))
            if speed > FLEE_SPEED:
                desired[:2] *= FLEE_SPEED/speed
        self.desired_velocity = desired.copy()
        yaw = float(np.clip(yaw_error*2/math.pi, -1, 1))
        return stabilize(e, desired, yaw, 1. if observation.foot_contact else 0.)


class EdgeEscapeBehavior(EscapeBehavior):
    """Preserve idle/flower behavior; active escape never returns to idle."""

    def reset(self) -> None:
        super().reset()
        self.escape = EdgeEscapePolicy(self.seed)
