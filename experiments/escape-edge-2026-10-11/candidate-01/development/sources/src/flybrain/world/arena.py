"""Bounds and seeded spawning, independent of graphics."""

from dataclasses import dataclass
import math
import random

from .fly import Fly, MotorCommand, validate_finite


@dataclass(frozen=True)
class Arena:
    """Rectangular arena in screen coordinates; no bouncing or wrapping."""

    width: float = 800.0
    height: float = 600.0

    def __post_init__(self) -> None:
        for name in ('width', 'height'):
            value = getattr(self, name)
            validate_finite(value, name)
            if value <= 0:
                raise ValueError(f'{name} must be positive')

    def validate_body(self, fly: Fly) -> None:
        """Check that the body has a valid radius and fits this arena."""
        validate_finite(fly.radius, 'radius')
        if fly.radius <= 0 or 2 * fly.radius > min(self.width, self.height):
            raise ValueError('arena must fit a positive body diameter')

    def spawn(self, seed: int = 0) -> Fly:
        """Spawn at center using a fresh local RNG, never the global RNG."""
        fly = Fly(self.width / 2, self.height / 2,
                  random.Random(seed).uniform(0, math.tau))
        self.validate_body(fly)
        return fly

    def step(self, fly: Fly, command: MotorCommand, dt: float) -> bool:
        """Move and clamp center; report crossing, excluding exact contact."""
        self.validate_body(fly)
        fly.step(command, dt)
        if dt == 0:
            return False
        x = min(max(fly.x, fly.radius), self.width - fly.radius)
        y = min(max(fly.y, fly.radius), self.height - fly.radius)
        collision = x != fly.x or y != fly.y
        fly.x, fly.y = x, y
        return collision
