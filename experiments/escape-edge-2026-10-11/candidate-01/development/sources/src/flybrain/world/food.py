"""Local food contact sampling, separate from neural inference and scoring."""

from dataclasses import dataclass
import math

from .fly import Fly, validate_finite
from .odor import OdorSource
from .sensing import SensoryObservation, sample_odor


@dataclass(frozen=True)
class FoodObservation(SensoryObservation):
    """Odor and simulated touch/taste only; no coordinates or motor answer."""

    food_contact: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.food_contact) is not bool:
            raise ValueError('food_contact must be boolean')


@dataclass(frozen=True)
class FoodPatch:
    """World-owned edible disk; radius matches the existing 10-unit drawing.

    Contact occurs when the fly body touches the patch. This is an engineered
    sensory assumption, not a reconstructed biological taste receptor.
    """

    x: float
    y: float
    radius: float = 10.0

    def __post_init__(self) -> None:
        for name in ('x', 'y', 'radius'):
            validate_finite(getattr(self, name), name)
        if self.radius <= 0:
            raise ValueError('food radius must be positive')

    def touches(self, fly: Fly) -> bool:
        """Sample physical overlap in the world, without supplying distance."""
        return math.hypot(fly.x - self.x, fly.y - self.y) <= self.radius + fly.radius


def sample_food(fly: Fly, source: OdorSource | None,
                food: FoodPatch | None) -> FoodObservation:
    """Sample independently available odor and contact modalities."""
    odor = sample_odor(fly, source)
    return FoodObservation(odor.odor_left, odor.odor_right,
                           food is not None and food.touches(fly))
