"""Explicit fixed physics ticks and a wall-clock frame accumulator."""

import math

from .world.arena import Arena
from .world.fly import Fly, MotorCommand, validate_finite
from .world.odor import OdorSource
from .world.sensing import sample_odor


class FixedStepSimulation:
    """Same tick sequence for equal duration and per-tick commands."""

    def __init__(self, arena: Arena, fly: Fly, dt: float = 0.02,
                 odor_source: OdorSource | None = None) -> None:
        validate_finite(dt, 'dt')
        if dt <= 0:
            raise ValueError('dt must be positive')
        arena.validate_body(fly)
        self.arena = arena
        self.fly = fly
        self.dt = dt
        self.tick_count = 0
        self._accumulator = 0.0
        self.odor_source = odor_source
        # Current-pose HUD observation; future neural inference must sample
        # separately before movement rather than reuse a stale display value.
        self.observation = sample_odor(self.fly, self.odor_source)

    def step(self, command: MotorCommand) -> bool:
        """Advance exactly one physics tick without reading a clock."""
        collision = self.arena.step(self.fly, command, self.dt)
        self.tick_count += 1
        self.observation = sample_odor(self.fly, self.odor_source)
        return collision

    def advance(self, elapsed: float, command: MotorCommand) -> int:
        """Consume whole ticks, retaining remainder; allow 1e-12 tick roundoff."""
        validate_finite(elapsed, 'elapsed')
        if elapsed < 0:
            raise ValueError('elapsed must be nonnegative')
        total = self._accumulator + elapsed
        validate_finite(total, 'accumulated elapsed')
        ticks = math.floor(total / self.dt + 1e-12)
        for _ in range(ticks):
            self.step(command)
        self._accumulator = max(0.0, total - ticks * self.dt)
        return ticks

    def reset(self, seed: int = 0) -> None:
        """Respawn body and clear all episode timing state."""
        self.fly = self.arena.spawn(seed)
        self.tick_count = 0
        self._accumulator = 0.0
        self.observation = sample_odor(self.fly, self.odor_source)
