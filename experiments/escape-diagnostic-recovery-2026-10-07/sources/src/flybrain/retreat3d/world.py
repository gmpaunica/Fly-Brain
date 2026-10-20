"""Optional encounter physics and environment-only fixed-step predator input."""
from dataclasses import dataclass, replace
import math
import numpy as np
from flybrain.flight3d.contracts import (ARENA, CONTROL_DT, PHYSICS_DT, FOOT_LENGTH,
    FLOWER_RADIUS, NECTAR_RADIUS, FlightCommand, FlightObservation, TrialSpec)
from flybrain.flight3d.world import FlightWorld, GRAVITY, DRAG, MAX_TILT, make_trial
from flybrain.flight3d.sensors import SUBPIXEL_RAYS, body_basis, observe_world
from flybrain.encounters3d.scene import (EncounterSnapshot, PREDATOR_RADIUS,
    encounter_eyes, make_scene)


@dataclass(frozen=True)
class PredatorInput:
    """Normalized environment axes; never part of a fly observation/command."""
    axes: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        values = np.asarray(self.axes, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all() or np.linalg.norm(values) > 1+1e-12:
            raise ValueError('finite normalized predator axes required')
        object.__setattr__(self, 'axes', tuple(float(x) for x in values))

    @property
    def velocity(self) -> tuple[float, float, float]:
        return tuple(.08*x for x in self.axes)

    @classmethod
    def from_keys(cls, keys: set[str]) -> 'PredatorInput':
        axes = np.array((('d' in keys)-('a' in keys), ('w' in keys)-('s' in keys),
                         ('e' in keys)-('q' in keys)), dtype=float)
        norm = float(np.linalg.norm(axes))
        return cls(tuple(axes/max(1., norm)))


@dataclass(frozen=True)
class EncounterObject:
    seed: int
    kind: str
    position: tuple[float, float, float]

    def __post_init__(self):
        if self.kind == 'flower':
            TrialSpec(self.seed, self.position)
        elif self.kind == 'predator':
            if len(self.position) != 3 or any(not math.isfinite(v) or not
                PREDATOR_RADIUS <= v <= limit-PREDATOR_RADIUS for v, limit in zip(self.position, ARENA)):
                raise ValueError('predator sphere must fit inside arena')
        else:
            raise ValueError('flower or predator required')
        object.__setattr__(self, 'position', tuple(float(x) for x in self.position))


@dataclass(frozen=True)
class RetreatSnapshot(EncounterSnapshot):
    """Optional-object pose for rendering/evaluation; controller receives none."""
    object_position: tuple[float, float, float] | None
    behavior_state: str = 'resting'
    decoder_confident: bool = False
    decoder_norm: float = 0.
    decoder_similarity: float = 0.
    decoder_margin: float = 0.
    wing_increment: tuple[int, ...] = ()
    angular_size_degrees: float = 0.
    angular_growth_degrees_s: float = 0.
    neural_origin_seconds: float = 0.
    escape_supported: bool = False


def ground_eyes(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Original sky/checker rays without any encounter surface."""
    rays = SUBPIXEL_RAYS @ body_basis(orientation).T
    shape = rays.shape[:-1]; rgb = np.empty((*shape, 3), dtype=np.float32)
    rgb[:] = (.035, .055, .09)
    dz = rays[..., 2]
    t = np.divide(-position[2], dz, out=np.full(shape, np.inf), where=dz < -1e-9)
    ground = (t > 0) & np.isfinite(t); safe = np.where(ground, t, 0.)
    checker = (np.floor((position[0]+rays[..., 0]*safe)/.025)+
               np.floor((position[1]+rays[..., 1]*safe)/.025)).astype(np.int64) % 2
    rgb[ground] = np.stack((.11+.025*checker, .17+.03*checker, .085+.02*checker), axis=-1)[ground]
    return rgb.mean(axis=-2)


def swept_contact(fly_old: np.ndarray, fly_new: np.ndarray, predator_old: np.ndarray,
                  predator_new: np.ndarray, radius: float) -> bool:
    """Closest relative segment against combined sphere radius, including rests."""
    start = fly_old-predator_old; segment = (fly_new-predator_new)-start
    length = float(segment@segment)
    fraction = float(np.clip(-(start@segment)/length, 0, 1)) if length else 0.
    return bool(np.linalg.norm(start+fraction*segment) <= radius)


class RetreatWorld(FlightWorld):
    """Retained flight/flower mechanics with safe ground support and moving threat."""
    def __init__(self, seed: int):
        self.seed = seed; self.scene: EncounterObject | None = None
        self.spawn_index = 0; self.predator_input = PredatorInput()
        super().__init__(make_trial(seed))

    def reset(self) -> None:
        super().reset()
        self.position[2] = FOOT_LENGTH; self.foot_contact = True
        self.specific_force[:] = (0., 0., GRAVITY)
        self.scene = None; self.spawn_index = 0; self.predator_input = PredatorInput()

    @property
    def objects(self) -> tuple[EncounterObject, ...]:
        return () if self.scene is None else (self.scene,)

    def spawn(self, kind: str, position: tuple | None = None) -> None:
        """Replace one object while preserving every body and timing variable."""
        seed = self.seed+self.spawn_index+1
        generated = make_scene(seed, kind)
        scene = EncounterObject(seed, kind, position or generated.position)
        self.spawn_index += 1; self.scene = scene
        if kind == 'flower':
            self.case = TrialSpec(self.seed, scene.position)
        # Removing a supporting flower removes support, while leaving pose and
        # momentum untouched. Contact is an environment sensor, not a pose reset.
        on_ground = self.position[2] <= FOOT_LENGTH+1e-12
        on_flower = (kind == 'flower' and abs(self.position[2]-FOOT_LENGTH-scene.position[2]) <= 1e-12
                     and np.linalg.norm(self.position[:2]-np.array(scene.position[:2])) <= FLOWER_RADIUS)
        self.foot_contact = self.foot_contact and bool(on_ground or on_flower)
        self.food_contact = False; self.dwell = 0.; self.first_contact = None
        self.predator_input = PredatorInput()

    def observe(self) -> FlightObservation:
        if self.scene is not None and self.scene.kind == 'flower':
            return observe_world(self)
        eyes = (ground_eyes(self.position, self.orientation) if self.scene is None else
                encounter_eyes(self.position, self.orientation, self.scene))
        return FlightObservation(0., 0., eyes, tuple(self.angular_velocity),
                                 tuple(self.specific_force), self.foot_contact, False)

    def step(self, command: FlightCommand, predator: PredatorInput | None = None) -> RetreatSnapshot:
        self.predator_input = predator or PredatorInput()
        return super().step(command)

    def _physics(self, command: FlightCommand) -> None:
        dt = PHYSICS_DT; self.physics_steps += 1
        old = self.position.copy()
        predator_old = None
        if self.scene is not None and self.scene.kind == 'predator':
            predator_old = np.array(self.scene.position)
            moved = np.clip(predator_old+np.array(self.predator_input.velocity)*dt,
                            PREDATOR_RADIUS, np.array(ARENA)-PREDATOR_RADIUS)
            self.scene = replace(self.scene, position=tuple(moved))
        if self.crashed:
            self.velocity[:] = 0.; self.angular_velocity[:] = 0.
            self.food_contact = False; self.dwell = 0.
            return
        old_contact = self.foot_contact
        supported = old_contact and command.power <= .5
        if supported:
            self.velocity[:] = 0.; self.angular_velocity[:] = 0.
            self.orientation[:2] *= math.exp(-dt/.08)
        else:
            previous = self.orientation.copy(); alpha = -math.expm1(-dt/.08)
            self.orientation[0] += alpha*(command.roll*MAX_TILT-self.orientation[0])
            self.orientation[1] += alpha*(command.pitch*MAX_TILT-self.orientation[1])
            self.orientation[2] += command.yaw*math.pi*dt
            self.angular_velocity = (self.orientation-previous)/dt
            basis = body_basis(self.orientation)
            force = basis[:, 2]*(2*GRAVITY*command.power)-DRAG*self.velocity
            self.velocity += (force-np.array((0., 0., GRAVITY)))*dt
            self.position += self.velocity*dt
            self.foot_contact = False; self.food_contact = False
            safe = (command.legs >= .5 and abs(self.velocity[2]) <= .08 and
                    np.linalg.norm(self.velocity[:2]) <= .08 and
                    max(abs(self.orientation[:2])) <= math.radians(20))
            if self.scene is not None and self.scene.kind == 'flower':
                fx, fy, fz = self.scene.position
                radial = math.hypot(self.position[0]-fx, self.position[1]-fy)
                if old[2]-FOOT_LENGTH >= fz-1e-12 and self.position[2]-FOOT_LENGTH <= fz and radial <= FLOWER_RADIUS:
                    self.impact_speed = float(np.linalg.norm(self.velocity))
                    self.position[2] = fz+FOOT_LENGTH
                    if safe:
                        self.foot_contact = True; self.food_contact = radial <= NECTAR_RADIUS
                        self.velocity[:] = 0.
                        if self.first_contact is None: self.first_contact = self.time
                    else:
                        self.crashed = True
                if radial < .0035 and self.position[2] < fz-.003:
                    self.crashed = True
            if self.position[2] <= FOOT_LENGTH and self.velocity[2] <= 0:
                self.impact_speed = float(np.linalg.norm(self.velocity))
                self.position[2] = FOOT_LENGTH
                if safe:
                    self.foot_contact = True; self.velocity[:] = 0.
                else:
                    self.crashed = True
        # Apply even during support: a moving predator can collide with a resting fly.
        if predator_old is not None and swept_contact(old, self.position, predator_old,
                np.array(self.scene.position), PREDATOR_RADIUS+.0015):
            self.crashed = True; self.impact_speed = float(np.linalg.norm(self.velocity))
        for axis, limit in enumerate(ARENA):
            if not .0015 <= self.position[axis] <= limit-.0015:
                self.position[axis] = np.clip(self.position[axis], .0015, limit-.0015)
                self.crashed = True
        if old_contact and not self.foot_contact:
            self.departures += 1
        if self.crashed:
            self.food_contact = False; self.foot_contact = False; self.velocity[:] = 0.
        self.dwell = self.dwell+dt if self.food_contact and not self.crashed else 0.

    def snapshot(self) -> RetreatSnapshot:
        return RetreatSnapshot(self.time, tuple(self.position), tuple(self.velocity),
            tuple(self.orientation), self.command, self.foot_contact, self.food_contact,
            self.dwell, self.crashed, self.seed, 'none' if self.scene is None else self.scene.kind,
            None if self.scene is None else self.scene.position)
