"""Screen-coordinate kinematics with seconds and clockwise radians."""

from dataclasses import dataclass, field
import math

MAX_FORWARD_SPEED = 100.0
MAX_ANGULAR_SPEED = math.pi
# Engineering scaffolding; not measured biological antenna geometry.
ANTENNA_FORWARD_OFFSET = 10.0
ANTENNA_LATERAL_OFFSET = 5.0


def validate_finite(value: float, name: str) -> None:
    """Reject nonfinite public numeric inputs."""
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')


@dataclass(frozen=True)
class MotorCommand:
    """Normalized forward [0,1] and clockwise turn [-1,1]."""

    forward: float = 0.0
    turn: float = 0.0

    def __post_init__(self) -> None:
        validate_finite(self.forward, 'forward')
        validate_finite(self.turn, 'turn')
        if not 0 <= self.forward <= 1 or not -1 <= self.turn <= 1:
            raise ValueError('forward must be in [0,1], turn in [-1,1]')


@dataclass
class Fly:
    """Mutable body; speeds are commands, not measured wall displacement."""

    x: float
    y: float
    heading: float = 0.0
    radius: float = 6.0
    forward_speed: float = field(default=0.0, init=False)
    angular_speed: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        for name in ('x', 'y', 'heading', 'radius'):
            validate_finite(getattr(self, name), name)
        if self.radius <= 0:
            raise ValueError('radius must be positive')
        self.heading %= math.tau

    def antenna_positions(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return left/right world points ahead of the body, without clamping."""
        cos_h, sin_h = math.cos(self.heading), math.sin(self.heading)
        front_x = self.x + ANTENNA_FORWARD_OFFSET * cos_h
        front_y = self.y + ANTENNA_FORWARD_OFFSET * sin_h
        left_x = ANTENNA_LATERAL_OFFSET * sin_h
        left_y = -ANTENNA_LATERAL_OFFSET * cos_h
        return ((front_x + left_x, front_y + left_y),
                (front_x - left_x, front_y - left_y))

    def step(self, command: MotorCommand, dt: float) -> None:
        """Turn before translating; zero seconds preserves all fields."""
        validate_finite(dt, 'dt')
        if dt < 0:
            raise ValueError('dt must be nonnegative')
        if dt == 0:
            return
        self.forward_speed = command.forward * MAX_FORWARD_SPEED
        self.angular_speed = command.turn * MAX_ANGULAR_SPEED
        self.heading = (self.heading + self.angular_speed * dt) % math.tau
        self.x += math.cos(self.heading) * self.forward_speed * dt
        self.y += math.sin(self.heading) * self.forward_speed * dt
