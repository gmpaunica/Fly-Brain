"""Fixed-step simplified flight and landing mechanics in metres and seconds."""

import math
import random
import numpy as np

from .contracts import (ARENA, START, PHYSICS_DT, CONTROL_DT, FOOT_LENGTH,
                        FLOWER_RADIUS, NECTAR_RADIUS, FlightCommand, FlightSnapshot, TrialSpec)
from .sensors import body_basis, observe_world

GRAVITY = 9.81
DRAG = 4.
MAX_TILT = .5


def make_trial(seed: int, *, sector: int | None=None, height: int | None=None) -> TrialSpec:
    """Randomize a flower with a local RNG; the body's initial pose never changes."""
    rng=random.Random(seed)
    sector=rng.randrange(8) if sector is None else sector
    height=rng.randrange(3) if height is None else height
    if sector not in range(8) or height not in range(3):
        raise ValueError('sector 0..7 and height 0..2 required')
    angle=(sector+rng.random())*math.tau/8
    radius=rng.uniform(.08,.22)
    z=rng.uniform(*((.035,.075),(.085,.115),(.135,.205))[height])
    return TrialSpec(seed,(.3+radius*math.sin(angle),.3+radius*math.cos(angle),z),
                     ('lower','level','higher')[height],sector)


class FlightWorld:
    """Engineering flight model; commands set power and bounded attitude targets.

    Lift acceleration is 2*g*power along body up. Tilt actuators have an 80 ms
    response, an assumed stabilization mechanism rather than a reconstructed
    wingbeat model. Physics applies support/friction only on actual contact.
    """

    def __init__(self,case: TrialSpec):
        self.case=case
        self.reset()

    def reset(self) -> None:
        """Restore the exact initial body, timing and contact history."""
        self.position=np.array(START,dtype=float)
        self.velocity=np.zeros(3)
        self.orientation=np.zeros(3)
        self.angular_velocity=np.zeros(3)
        self.specific_force=np.zeros(3)
        self.physics_steps=0
        self.command=FlightCommand()
        self.foot_contact=False; self.food_contact=False; self.crashed=False
        self.dwell=0.; self.first_contact=None; self.impact_speed=None
        self.departures=0

    @property
    def time(self) -> float:
        return self.physics_steps*PHYSICS_DT

    def observe(self):
        """Sample current sensors without advancing state or random generators."""
        return observe_world(self)

    def step(self,command: FlightCommand) -> FlightSnapshot:
        """Hold a neural/manual command across four fixed physics substeps."""
        if not isinstance(command,FlightCommand): raise ValueError('FlightCommand required')
        self.command=command
        old_velocity=self.velocity.copy()
        old_orientation=self.orientation.copy()
        for _ in range(4): self._physics(command)
        self.angular_velocity=(self.orientation-old_orientation)/CONTROL_DT
        acceleration=(self.velocity-old_velocity)/CONTROL_DT
        self.specific_force=body_basis(self.orientation).T@(acceleration+np.array((0.,0.,GRAVITY)))
        return self.snapshot()

    def _physics(self,command: FlightCommand) -> None:
        dt=PHYSICS_DT
        self.physics_steps+=1
        if self.crashed:
            self.velocity[:]=0.; self.angular_velocity[:]=0.
            self.food_contact=False; self.dwell=0.
            return
        old=self.position.copy()
        old_contact=self.foot_contact
        if old_contact and command.power<=.5:
            self.velocity[:]=0.; self.angular_velocity[:]=0.
            self.orientation[:2]*=math.exp(-dt/.08)
            self.specific_force[:]=(0.,0.,GRAVITY)
            self.dwell+=dt if self.food_contact else 0.
            return
        previous_angles=self.orientation.copy()
        self.orientation[0]+=(-math.expm1(-dt/.08))*(command.roll*MAX_TILT-self.orientation[0])
        self.orientation[1]+=(-math.expm1(-dt/.08))*(command.pitch*MAX_TILT-self.orientation[1])
        self.orientation[2]+=command.yaw*math.pi*dt
        self.angular_velocity=(self.orientation-previous_angles)/dt
        basis=body_basis(self.orientation)
        force=basis[:,2]*(2*GRAVITY*command.power)-DRAG*self.velocity
        self.specific_force=basis.T@force
        self.velocity+=(force-np.array((0.,0.,GRAVITY)))*dt
        self.position+=self.velocity*dt
        self.foot_contact=False; self.food_contact=False
        fx,fy,fz=self.case.flower
        radial=math.hypot(self.position[0]-fx,self.position[1]-fy)
        crossing=old[2]-FOOT_LENGTH>=fz-1e-12 and self.position[2]-FOOT_LENGTH<=fz
        if crossing and radial<=FLOWER_RADIUS:
            speed=float(np.linalg.norm(self.velocity))
            safe=(command.legs>=.5 and abs(self.velocity[2])<=.08
                  and np.linalg.norm(self.velocity[:2])<=.08
                  and max(abs(self.orientation[:2]))<=math.radians(20))
            self.impact_speed=speed
            self.position[2]=fz+FOOT_LENGTH
            if safe:
                self.foot_contact=True
                self.food_contact=radial<=NECTAR_RADIUS
                self.velocity[:]=0.
                if self.first_contact is None: self.first_contact=self.time
            else: self.crashed=True
        if radial<.0035 and self.position[2]<fz-.003:
            self.crashed=True
        if self.position[2]<FOOT_LENGTH:
            self.position[2]=FOOT_LENGTH; self.crashed=True
        for axis,limit in enumerate(ARENA):
            if self.position[axis]<.0015 or self.position[axis]>limit-.0015:
                self.position[axis]=np.clip(self.position[axis],.0015,limit-.0015)
                self.crashed=True
        if old_contact and not self.foot_contact: self.departures+=1
        if self.crashed: self.food_contact=False; self.velocity[:]=0.
        self.dwell=self.dwell+dt if self.food_contact and not self.crashed else 0.

    def snapshot(self) -> FlightSnapshot:
        """Copy state into immutable values for drawing and measurement."""
        return FlightSnapshot(self.time,tuple(self.position),tuple(self.velocity),
            tuple(self.orientation),self.case.flower,self.command,self.foot_contact,
            self.food_contact,self.dwell,self.crashed,self.case.seed)
