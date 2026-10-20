"""Environment-only exclusive scene, ray vision and fixed-step contact physics."""
from dataclasses import dataclass
import math
import random
import numpy as np
from flybrain.flight3d.contracts import (
    ARENA, FOOT_LENGTH, PHYSICS_DT, FlightCommand, FlightObservation, TrialSpec,
)
from flybrain.flight3d.sensors import SUBPIXEL_RAYS, body_basis, eye_images
from flybrain.flight3d.world import FlightWorld, GRAVITY, DRAG, MAX_TILT, make_trial
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .diagnostics import DiagnosticInterval

PREDATOR_RADIUS = .018  # metres; stationary spherical high-contrast proxy
PREDATOR_RGB = (.05, .95, .95)


@dataclass(frozen=True)
class SceneSpec:
    """Seed, kind and XYZ are restricted to environment/evaluation/presentation."""
    seed: int
    kind: str
    position: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.kind not in ('flower', 'predator'):
            raise ValueError('exactly one flower or predator required')
        TrialSpec(self.seed, self.position)  # shared arena/seed validation
        object.__setattr__(self, 'position', tuple(self.position))


def make_scene(seed: int, kind: str) -> SceneSpec:
    """Reproducible environment setup using a local seed stream."""
    if kind == 'flower':
        return SceneSpec(seed, kind, make_trial(seed).flower)
    rng = random.Random(seed)
    angle = rng.random()*math.tau
    radius = rng.uniform(.08, .18)
    return SceneSpec(seed, kind, (.3+radius*math.sin(angle),
                                 .3+radius*math.cos(angle), rng.uniform(.075, .135)))


def encounter_eyes(position: np.ndarray, orientation: np.ndarray, scene: SceneSpec) -> np.ndarray:
    """Nearest-hit RGB only; independent rays then existing eight-ray averaging.

    Cyan recognition is an engineered sensory assumption. No distances, hit
    masks, identity or object-relative direction cross the observation boundary.
    The flower's existing visual surfaces are preserved exactly.
    """
    if scene.kind == 'flower':
        return eye_images(position, orientation, scene.position)
    rays = SUBPIXEL_RAYS @ body_basis(orientation).T
    shape = rays.shape[:-1]
    rgb = np.empty((*shape, 3), dtype=np.float32)
    rgb[:] = (.035, .055, .09)
    dz = rays[..., 2]
    ground_t = np.divide(-position[2], dz, out=np.full(shape, np.inf), where=np.abs(dz)>1e-9)
    ground = (ground_t>0) & np.isfinite(ground_t)
    safe = np.where(ground, ground_t, 0.)
    gx = position[0]+rays[...,0]*safe
    gy = position[1]+rays[...,1]*safe
    checker = (np.floor(gx/.025)+np.floor(gy/.025)).astype(np.int64)%2
    rgb[ground] = np.stack((.11+.025*checker, .17+.03*checker, .085+.02*checker), axis=-1)[ground]
    nearest = np.where(ground, ground_t, np.inf)
    offset = position-np.array(scene.position)
    b = np.sum(rays*offset, axis=-1)
    disc = b*b-(np.dot(offset, offset)-PREDATOR_RADIUS**2)
    entry = -b-np.sqrt(np.maximum(disc, 0.))
    exit_t = -b+np.sqrt(np.maximum(disc, 0.))
    t = np.where(entry>0, entry, exit_t)
    visible = (disc>=0) & (t>0) & (t<nearest)
    rgb[visible] = PREDATOR_RGB
    return rgb.mean(axis=-2)


@dataclass(frozen=True)
class EncounterSnapshot:
    """Immutable evaluator/display data, separate from sensory observations."""
    time: float
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    orientation: tuple[float, float, float]
    command: FlightCommand
    foot_contact: bool
    food_contact: bool
    dwell: float
    crashed: bool
    seed: int
    object_kind: str
    object_position: tuple[float, float, float]
    neuron_ids: tuple[str, ...] = ()
    spike_counts: tuple[int, ...] = ()
    mode: str = 'teacher'
    edge_order: tuple[tuple[str, str], ...] = ()
    edge_transmissions: tuple[float, ...] = ()
    edge_activity: tuple[float, ...] = ()
    pathway: 'DiagnosticInterval | None' = None


class EncounterWorld(FlightWorld):
    """Reuse flower mechanics unchanged; predator has no flower/odor/contact path."""
    def __init__(self, scene: SceneSpec):
        self.scene = scene
        super().__init__(TrialSpec(scene.seed, scene.position) if scene.kind=='flower' else scene)

    @property
    def objects(self) -> tuple[SceneSpec, ...]:
        """The exclusive stationary object, for environment/evaluator inspection."""
        return (self.scene,)

    def observe(self) -> FlightObservation:
        if self.scene.kind=='flower':
            return super().observe()
        return FlightObservation(0., 0., encounter_eyes(self.position, self.orientation, self.scene),
                                 tuple(self.angular_velocity), tuple(self.specific_force), False, False)

    def _physics(self, command: FlightCommand) -> None:
        if self.scene.kind=='flower':
            super()._physics(command)
            return
        self.physics_steps += 1
        if self.crashed:
            self.velocity[:]=0.; self.angular_velocity[:]=0.
            return
        old=self.position.copy()
        previous=self.orientation.copy()
        alpha=-math.expm1(-PHYSICS_DT/.08)
        self.orientation[0] += alpha*(command.roll*MAX_TILT-self.orientation[0])
        self.orientation[1] += alpha*(command.pitch*MAX_TILT-self.orientation[1])
        self.orientation[2] += command.yaw*math.pi*PHYSICS_DT
        self.angular_velocity=(self.orientation-previous)/PHYSICS_DT
        basis=body_basis(self.orientation)
        force=basis[:,2]*(2*GRAVITY*command.power)-DRAG*self.velocity
        self.specific_force=basis.T@force
        self.velocity += (force-np.array((0.,0.,GRAVITY)))*PHYSICS_DT
        self.position += self.velocity*PHYSICS_DT
        # Swept sphere contact prevents tunneling during a physics interval.
        segment=self.position-old
        length=float(np.dot(segment,segment))
        fraction=float(np.clip(np.dot(np.array(self.scene.position)-old,segment)/length,0,1)) if length else 0.
        distance=np.linalg.norm(old+fraction*segment-np.array(self.scene.position))
        if distance <= PREDATOR_RADIUS+.0015:
            self.crashed=True; self.impact_speed=float(np.linalg.norm(self.velocity))
        if self.position[2]<FOOT_LENGTH:
            self.position[2]=FOOT_LENGTH; self.crashed=True
        for axis,limit in enumerate(ARENA):
            if not .0015<=self.position[axis]<=limit-.0015:
                self.position[axis]=np.clip(self.position[axis],.0015,limit-.0015)
                self.crashed=True
        if self.crashed: self.velocity[:]=0.

    def snapshot(self) -> EncounterSnapshot:
        return EncounterSnapshot(self.time, tuple(self.position), tuple(self.velocity),
            tuple(self.orientation), self.command, self.foot_contact, self.food_contact,
            self.dwell, self.crashed, self.scene.seed, self.scene.kind, self.scene.position)
