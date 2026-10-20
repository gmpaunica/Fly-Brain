"""Documented visual-servo teacher; reads only eyes, odor and body-local sensors."""

import math
import numpy as np

from .contracts import CONTROL_DT, FlightCommand, FlightObservation, FLOWER_RADIUS, FOOT_LENGTH
from .sensors import EYE_RAYS, body_basis, flower_contrast
from .world import GRAVITY, DRAG, MAX_TILT


def visual_measurement(eyes: np.ndarray) -> tuple[np.ndarray,float] | None:
    """Infer a flower direction/range from RGB and known apparent-size calibration.

    This is an engineered visual detector INSIDE the teacher/encoder, not an
    environment-supplied target bearing. It has no world/target object access.
    Range assumes the experiment's fixed petal radius; it is an image estimate.
    """
    contrast=flower_contrast(eyes)
    mask=contrast>0
    directions=EYE_RAYS[mask]
    if not len(directions): return None
    weights=contrast[mask]
    direction=np.average(directions,axis=0,weights=weights)
    direction/=np.linalg.norm(direction)
    spread=np.sqrt(np.maximum(0,1-np.clip(directions@direction,-1,1)**2))
    angular_radius=max(float(np.max(spread)),.025)
    distance=float(np.clip(FLOWER_RADIUS/angular_radius,.015,.6))
    return direction*distance, float(mask.sum())


class FlightTeacher:
    """Visual target servo and integrated local IMU; no absolute pose input.

    Gravity, flower size and actuator calibration are declared engineering
    knowledge. Orientation/velocity are internal estimates integrated from
    local feedback, with the experiment's fixed initial rest condition.
    """

    def __init__(self): self.reset()

    def reset(self) -> None:
        self.orientation=np.zeros(3); self.velocity=np.zeros(3)
        self.first=True
        self.last_visual=None
        self.altitude=0.
        self.search_entry=None
        self.search_elapsed=0.

    def command(self,observation: FlightObservation) -> FlightCommand:
        """Infer a bounded command from the current observation and own history."""
        if not isinstance(observation,FlightObservation): raise ValueError('FlightObservation required')
        if self.first: self.first=False
        else:
            self.orientation+=np.array(observation.angular_velocity)*CONTROL_DT
            acceleration=body_basis(self.orientation)@np.array(observation.specific_force)-np.array((0,0,GRAVITY))
            self.velocity+=acceleration*CONTROL_DT
            self.altitude+=self.velocity[2]*CONTROL_DT
        if observation.foot_contact:
            self.velocity[:]=0
            return FlightCommand(legs=1)
        sight=visual_measurement(observation.eyes)
        basis=body_basis(self.orientation)
        if sight is None:
            desired=np.zeros(3)
            if self.search_entry is None:
                self.search_entry=self.altitude
                self.search_elapsed=0.
            self.search_elapsed+=CONTROL_DT
            phase=math.tau*self.search_elapsed/8.
            target=self.search_entry+.015*math.sin(phase)
            desired[2]=.015*math.tau/8.*math.cos(phase)+2*(target-self.altitude)
            yaw=.15
        else:
            self.search_entry=None
            self.search_elapsed=0.
            relative=basis@sight[0]
            horizontal=float(np.linalg.norm(relative[:2]))
            # Approach above the flower, then lower the feet onto its surface.
            clearance=.035*np.clip(horizontal/.06,0,1)
            error=relative+np.array((0,0,.004+FOOT_LENGTH+clearance))
            desired=np.clip(error*2.,-.10,.10)
            # Slower final descent supports controlled surface contact.
            desired[2]=max(desired[2],-.035)
            az=math.atan2(sight[0][0],sight[0][1])
            yaw=float(np.clip(az*.4,-.35,.35)) if horizontal>.025 else 0.
            self.last_visual=tuple(relative)
        acceleration=5.*(desired-self.velocity)+DRAG*self.velocity
        force=acceleration+np.array((0,0,GRAVITY))
        forward=np.array((math.sin(self.orientation[2]),math.cos(self.orientation[2]),0.))
        right=np.array((math.cos(self.orientation[2]),-math.sin(self.orientation[2]),0.))
        pitch=math.atan2(float(force@forward),max(force[2],1.))/MAX_TILT
        roll=math.atan2(float(force@right),max(force[2],1.))/MAX_TILT
        # Compensate current tilt when calculating vertical lift.
        power=float(np.clip(force[2]/(2*GRAVITY*max(basis[2,2],.5)),0,1))
        return FlightCommand(power,float(np.clip(pitch,-1,1)),float(np.clip(roll,-1,1)),yaw,1.)
