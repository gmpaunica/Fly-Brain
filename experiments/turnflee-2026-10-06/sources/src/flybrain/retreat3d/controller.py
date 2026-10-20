"""Declared engineered idle motor policy and sensory-derived teacher handoff."""
import math
import random
import numpy as np
from flybrain.flight3d.contracts import CONTROL_DT, FOOT_LENGTH, FlightCommand, FlightObservation
from flybrain.flight3d.sensors import body_basis
from flybrain.flight3d.world import GRAVITY, DRAG, MAX_TILT
from flybrain.flight3d.teacher import FlightTeacher


class KinematicEstimate:
    """Integrate local IMU, assuming initial upright rest on calibrated ground.

    Displacement/altitude are estimates relative to the known initial rest;
    no world/target pose ever initializes this object or a controller handoff.
    """
    def __init__(self):
        self.orientation = np.zeros(3); self.velocity = np.zeros(3)
        self.displacement = np.zeros(3); self.altitude = FOOT_LENGTH
        self.first = True

    def update(self, observation: FlightObservation) -> None:
        if self.first:
            self.first = False
            return
        self.orientation += np.array(observation.angular_velocity)*CONTROL_DT
        acceleration = body_basis(self.orientation)@np.array(observation.specific_force)-np.array((0., 0., GRAVITY))
        old_velocity = self.velocity.copy()
        self.velocity += acceleration*CONTROL_DT
        # Four semi-implicit 5 ms world steps; trapezoid plus half a physics step.
        change = (old_velocity*.0075+self.velocity*.0125)
        self.displacement += change; self.altitude += float(change[2])
        if observation.foot_contact:
            self.velocity[:] = 0.


def stabilize(estimate: KinematicEstimate, desired: np.ndarray, yaw: float = 0., legs: float = 0.) -> FlightCommand:
    """Calibrated velocity servo; no perception, neural decoder or object geometry."""
    force = 5*(desired-estimate.velocity)+DRAG*estimate.velocity+np.array((0., 0., GRAVITY))
    heading = estimate.orientation[2]
    forward = np.array((math.sin(heading), math.cos(heading), 0.))
    right = np.array((math.cos(heading), -math.sin(heading), 0.))
    pitch = math.atan2(float(force@forward), max(force[2], 1.))/MAX_TILT
    roll = math.atan2(float(force@right), max(force[2], 1.))/MAX_TILT
    up = body_basis(estimate.orientation)[2, 2]
    power = float(np.clip(force[2]/(2*GRAVITY*max(up, .5)), 0, 1))
    return FlightCommand(power, float(np.clip(pitch, -1, 1)), float(np.clip(roll, -1, 1)),
                         float(np.clip(yaw, -1, 1)), legs)


class IdleFlight:
    """Seeded 3–6 s cruise, safe landing, 2–5 s rest. Modeling defaults only."""
    def __init__(self, seed: int):
        self.seed = seed; self.reset()

    def reset(self) -> None:
        self.rng = random.Random(self.seed); self.estimate = KinematicEstimate()
        self.state = 'takeoff'; self.elapsed = 0.; self.heading = 0.
        self.turn = 0.; self.turn_target = 0.; self.turn_timer = 0.
        self.cruise_durations = []; self.rest_durations = []
        self.deadline = 0.

    def transition(self, state: str) -> None:
        self.state = state; self.elapsed = 0.
        if state == 'cruising':
            self.deadline = self.rng.uniform(3, 6); self.cruise_durations.append(self.deadline)
        elif state == 'resting':
            self.deadline = self.rng.uniform(2, 5); self.rest_durations.append(self.deadline)

    def command(self, observation: FlightObservation, *, update: bool = True) -> FlightCommand:
        if update:
            self.estimate.update(observation)
        e = self.estimate; self.elapsed += CONTROL_DT
        if self.state == 'takeoff' and e.altitude >= .095:
            self.transition('cruising')
        if self.state == 'cruising' and self.elapsed >= self.deadline:
            self.transition('landing')
        if self.state == 'landing' and observation.foot_contact:
            self.transition('resting')
        if self.state == 'resting':
            if self.elapsed < self.deadline:
                return FlightCommand(legs=1)
            self.transition('takeoff')
        desired = np.zeros(3); yaw = 0.; legs = 0.
        if self.state == 'takeoff':
            desired[2] = min(.06, max(.012, 2*(.1-e.altitude)))
            legs = 1.
        elif self.state == 'cruising':
            self.turn_timer -= CONTROL_DT
            if self.turn_timer <= 0:
                self.turn_target = self.rng.uniform(-.5, .5); self.turn_timer = self.rng.uniform(.5, 1.5)
            # Smooth local heading; inward bias uses only integrated displacement.
            self.turn += -math.expm1(-CONTROL_DT/.5)*(self.turn_target-self.turn)
            self.heading += self.turn*CONTROL_DT
            radius = float(np.linalg.norm(e.displacement[:2]))
            if radius > .10:
                home = math.atan2(-e.displacement[0], -e.displacement[1])
                error = (home-self.heading+math.pi) % math.tau-math.pi
                self.heading += float(np.clip(error, -.9*CONTROL_DT, .9*CONTROL_DT))
            desired[:2] = .06*np.array((math.sin(self.heading), math.cos(self.heading)))
            desired[2] = 2*(.1-e.altitude)
            error = (self.heading-e.orientation[2]+math.pi) % math.tau-math.pi
            yaw = float(np.clip(error, -.4, .4))
        elif self.state == 'landing':
            desired[2] = -min(.035, max(.012, 2*(e.altitude-FOOT_LENGTH)))
            legs = 1.
        return stabilize(e, desired, yaw, legs)


class BehaviorController:
    """Idle and retained flower teacher. Escape is enabled only by a passed gate."""
    def __init__(self, seed: int):
        self.seed = seed; self.reset()

    def reset(self) -> None:
        self.idle = IdleFlight(self.seed); self.kind = 'none'
        self.teacher: FlightTeacher | None = None; self.flower_takeoff = False
        self.last_contact = True

    @property
    def state(self) -> str:
        if self.kind == 'flower':
            return 'flower-takeoff' if self.flower_takeoff else 'flower-resting' if self.last_contact else 'flower'
        return self.idle.state

    def _transfer(self) -> None:
        e = self.idle.estimate
        self.teacher.orientation = e.orientation.copy()
        self.teacher.velocity = e.velocity.copy()
        self.teacher.altitude = e.altitude
        self.teacher.first = False

    def encounter(self, kind: str, observation: FlightObservation | None = None) -> None:
        """Object mode event only, never an environment pose/velocity handoff."""
        if observation is not None:
            self.last_contact = observation.foot_contact
        self.kind = kind; self.teacher = None
        if kind == 'flower':
            self.teacher = FlightTeacher(); self._transfer()
            self.flower_takeoff = self.last_contact
            if self.flower_takeoff:
                self.idle.transition('takeoff')
        elif self.last_contact:
            self.idle.transition('resting')
        elif self.idle.state == 'resting':
            self.idle.transition('takeoff')

    def command(self, observation: FlightObservation) -> FlightCommand:
        if not isinstance(observation, FlightObservation):
            raise ValueError('FlightObservation required')
        self.idle.estimate.update(observation); self.last_contact = observation.foot_contact
        if self.kind == 'flower':
            if self.flower_takeoff:
                if self.idle.estimate.altitude < .095:
                    return self.idle.command(observation, update=False)
                self.flower_takeoff = False
            self._transfer()
            # Shared estimate already consumed this observation; skip the old
            # teacher's duplicate IMU integration, retaining its visual servo.
            self.teacher.first = True
            return self.teacher.command(observation)
        return self.idle.command(observation, update=False)
