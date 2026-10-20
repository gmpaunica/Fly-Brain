"""Explicit SI-unit boundaries between physics, sensing, neurons and views."""

from dataclasses import dataclass, field
import math
import numpy as np

CONTROL_DT = .020
PHYSICS_DT = .005
NEURAL_DT = .001
START = (.3, .3, .1)
ARENA = (.6, .6, .3)
FOOT_LENGTH = .003
FLOWER_RADIUS = .025
NECTAR_RADIUS = .015


@dataclass(frozen=True)
class FlightCommand:
    """Normalized power [0,1], tilt/yaw [-1,1], leg extension [0,1].

    Positive pitch tilts thrust forward; positive roll tilts right; positive
    yaw turns right. These actuator mappings are engineering assumptions.
    """

    power: float = 0.
    pitch: float = 0.
    roll: float = 0.
    yaw: float = 0.
    legs: float = 0.

    def __post_init__(self) -> None:
        for key in ('power', 'pitch', 'roll', 'yaw', 'legs'):
            value = getattr(self, key)
            low = 0 if key in ('power', 'legs') else -1
            if not math.isfinite(value) or not low <= value <= 1:
                raise ValueError(f'{key} outside [{low},1]')

    def vector(self) -> np.ndarray:
        """Training target order: power, pitch, roll, yaw, legs."""
        return np.array((self.power,self.pitch,self.roll,self.yaw,self.legs), dtype=np.float32)


@dataclass(frozen=True)
class FlightObservation:
    """Local senses only; eye RGB samples are [eye,row,column,channel]."""

    odor_left: float
    odor_right: float
    eyes: np.ndarray = field(repr=False, compare=False)
    angular_velocity: tuple[float, float, float]
    specific_force: tuple[float, float, float]
    foot_contact: bool
    food_contact: bool

    def __post_init__(self) -> None:
        scalars = (self.odor_left,self.odor_right,*self.angular_velocity,*self.specific_force)
        if not all(math.isfinite(x) for x in scalars) or min(self.odor_left,self.odor_right)<0:
            raise ValueError('invalid sensory values')
        pixels = np.asarray(self.eyes,dtype=np.float32)
        if pixels.shape != (2,24,48,3) or not np.isfinite(pixels).all() or pixels.min()<0 or pixels.max()>1:
            raise ValueError('eyes must be finite RGB [2,24,48,3] in [0,1]')
        pixels = np.frombuffer(pixels.tobytes(),dtype=np.float32).reshape(pixels.shape)
        object.__setattr__(self,'eyes',pixels)


@dataclass(frozen=True)
class TrialSpec:
    """Environment/evaluator configuration. Never pass this to a controller."""

    seed: int
    flower: tuple[float,float,float]
    height_group: str = 'custom'
    sector: int = -1

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed<0:
            raise ValueError('nonnegative integer seed required')
        if len(self.flower)!=3 or any(not math.isfinite(v) for v in self.flower):
            raise ValueError('finite flower XYZ required')
        if not (.04<=self.flower[0]<=.56 and .04<=self.flower[1]<=.56 and .025<=self.flower[2]<=.23):
            raise ValueError('flower must fit within the arena')


@dataclass(frozen=True)
class FlightSnapshot:
    """Immutable presentation/evaluation data; has no simulation methods."""

    time: float
    position: tuple[float,float,float]
    velocity: tuple[float,float,float]
    orientation: tuple[float,float,float]
    flower: tuple[float,float,float]
    command: FlightCommand
    foot_contact: bool
    food_contact: bool
    dwell: float
    crashed: bool
    seed: int
    neuron_ids: tuple[str,...] = ()
    spike_counts: tuple[int,...] = ()
    mode: str = 'manual'
    edge_order: tuple[tuple[str,str],...] = ()
    edge_transmissions: tuple[float,...] = ()
    edge_activity: tuple[float,...] = ()
