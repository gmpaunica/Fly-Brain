"""Honest gate-failed circuit probe and explicitly engineered local hover."""
import math
import numpy as np
from flybrain.flight3d.contracts import CONTROL_DT, FlightCommand, FlightObservation
from flybrain.flight3d.sensors import body_basis
from flybrain.flight3d.world import GRAVITY, DRAG, MAX_TILT
from .neural import EscapeBrain


class LocalHover:
    """Maintain initial flight height using integrated body-local IMU signals.

    Initial rest, gravity and actuator calibration are engineering assumptions.
    This receives no scene/eye/odor information and does not generate escape.
    Relative height/velocity are estimates, not environment coordinates.
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.orientation=np.zeros(3); self.velocity=np.zeros(3)
        self.height_change=0.; self.first=True

    def command(self, observation: FlightObservation) -> FlightCommand:
        if self.first:
            self.first=False
        else:
            self.orientation+=np.array(observation.angular_velocity)*CONTROL_DT
            acceleration=body_basis(self.orientation)@np.array(observation.specific_force)-np.array((0.,0.,GRAVITY))
            self.velocity+=acceleration*CONTROL_DT
            self.height_change+=self.velocity[2]*CONTROL_DT
        desired=np.array((0.,0.,-2*self.height_change))
        force=5*(desired-self.velocity)+DRAG*self.velocity+np.array((0.,0.,GRAVITY))
        yaw=self.orientation[2]
        forward=np.array((math.sin(yaw),math.cos(yaw),0.))
        right=np.array((math.cos(yaw),-math.sin(yaw),0.))
        pitch=math.atan2(float(force@forward),max(force[2],1.))/MAX_TILT
        roll=math.atan2(float(force@right),max(force[2],1.))/MAX_TILT
        up=body_basis(self.orientation)[2,2]
        power=float(np.clip(force[2]/(2*GRAVITY*max(up,.5)),0,1))
        return FlightCommand(power,float(np.clip(pitch,-1,1)),float(np.clip(roll,-1,1)))


class PredatorController:
    """Advance actual visual circuitry; no escape decoder passed characterization.

    A hover stabilizer is explicitly the sole actuator controller in this failed
    gate preview. Neither RGB recognition nor neural threat activity supplies
    steering. Do not describe this as verified neural avoidance.
    """
    avoidance_supported=False

    def __init__(self, manifest: dict) -> None:
        self.brain=EscapeBrain(manifest)
        self.stabilizer=LocalHover()

    def reset(self) -> None:
        self.brain.reset(); self.stabilizer.reset()

    def command(self, observation: FlightObservation) -> FlightCommand:
        if not isinstance(observation,FlightObservation):
            raise ValueError('FlightObservation required')
        self.brain.advance(observation)
        return self.stabilizer.command(observation)
