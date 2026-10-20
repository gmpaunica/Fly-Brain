"""Causal orchestration between world sensing and the neural controller."""

from dataclasses import dataclass
import math

from .neural.controller import NeuralController
from .neural.network import NeuralActivity
from .simulation import FixedStepSimulation
from .world.fly import Fly, MotorCommand, validate_finite
from .world.sensing import SensoryObservation, sample_odor


@dataclass(frozen=True)
class BodyState:
    """Immutable copy of the resulting fly body state after a world tick."""

    x: float
    y: float
    heading: float
    radius: float
    forward_speed: float
    angular_speed: float

    @classmethod
    def from_fly(cls, fly: Fly) -> 'BodyState':
        """Copy every public scalar body field without retaining mutable state."""
        return cls(
            fly.x,
            fly.y,
            fly.heading,
            fly.radius,
            fly.forward_speed,
            fly.angular_speed,
        )


@dataclass(frozen=True)
class ClosedLoopSnapshot:
    """One causal tick: pre-pose input, neural output, then resulting body."""

    tick_id: int
    pre_movement_observation: SensoryObservation
    activity: NeuralActivity
    command: MotorCommand
    collision: bool
    body: BodyState


class ClosedLoopSimulation:
    """Own fixed-step accumulation around an existing world and controller."""

    def __init__(self, world: FixedStepSimulation,
                 controller: NeuralController) -> None:
        if not math.isclose(
            world.dt,
            controller.neural_interval_seconds,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('world tick must equal exactly 20 neural substeps')
        self.world = world
        self.controller = controller
        self._accumulator = 0.0
        self.last_snapshot: ClosedLoopSnapshot | None = None

    def step(self) -> ClosedLoopSnapshot:
        """Sample, infer, decode, and move exactly once in causal order."""
        pre_movement = sample_odor(self.world.fly, self.world.odor_source)
        decision = self.controller.decide(pre_movement)
        collision = self.world.step(decision.command)
        snapshot = ClosedLoopSnapshot(
            tick_id=self.world.tick_count,
            pre_movement_observation=pre_movement,
            activity=decision.activity,
            command=decision.command,
            collision=collision,
            body=BodyState.from_fly(self.world.fly),
        )
        self.last_snapshot = snapshot
        return snapshot

    def advance(self, elapsed: float) -> int:
        """Consume whole closed-loop ticks and retain fractional frame time."""
        return len(self.advance_snapshots(elapsed))

    def advance_snapshots(self, elapsed: float) -> tuple[ClosedLoopSnapshot, ...]:
        """Advance elapsed time and return every completed causal tick."""
        validate_finite(elapsed, 'elapsed')
        if elapsed < 0:
            raise ValueError('elapsed must be nonnegative')
        total = self._accumulator + elapsed
        validate_finite(total, 'accumulated elapsed')
        ticks = math.floor(total / self.world.dt + 1e-12)
        snapshots = tuple(self.step() for _ in range(ticks))
        self._accumulator = max(0.0, total - ticks * self.world.dt)
        return snapshots

    def reset(self, seed: int = 0) -> None:
        """Reset every episode state while retaining source and fixed topology."""
        self.world.reset(seed)
        self.controller.reset()
        self._accumulator = 0.0
        self.last_snapshot = None
