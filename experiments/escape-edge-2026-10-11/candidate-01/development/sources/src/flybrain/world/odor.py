"""Static scalar odor approximation in arbitrary concentration units."""

from dataclasses import dataclass
import math

from .fly import validate_finite


@dataclass(frozen=True)
class OdorSource:
    """Synthetic softened inverse-square field, defined beyond arena walls."""

    x: float
    y: float
    strength: float = 1000.0
    softening: float = 20.0

    def __post_init__(self) -> None:
        for name in ('x', 'y', 'strength', 'softening'):
            validate_finite(getattr(self, name), name)
        if self.strength < 0 or self.softening <= 0:
            raise ValueError('strength must be nonnegative and softening positive')
        validate_finite(self.strength / self.softening / self.softening,
                        'peak concentration')

    def sample(self, x: float, y: float) -> float:
        """Sample strength/(distance²+softening²), without squaring overflow."""
        validate_finite(x, 'sample x')
        validate_finite(y, 'sample y')
        softened_distance = math.hypot(x - self.x, y - self.y, self.softening)
        return self.strength / softened_distance / softened_distance
