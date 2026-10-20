"""Sensory-only observation boundary; coordinates stay in world sampling."""

from dataclasses import dataclass

from .fly import Fly, validate_finite
from .odor import OdorSource


@dataclass(frozen=True)
class SensoryObservation:
    """Two raw concentrations; no pose, target, bearing or steering answer."""

    odor_left: float
    odor_right: float

    def __post_init__(self) -> None:
        for name in ('odor_left', 'odor_right'):
            value = getattr(self, name)
            validate_finite(value, name)
            if value < 0:
                raise ValueError(f'{name} must be nonnegative')


def sample_odor(fly: Fly, source: OdorSource | None) -> SensoryObservation:
    """Independently sample both antenna points; no source gives zero input."""
    if source is None:
        return SensoryObservation(0.0, 0.0)
    left, right = fly.antenna_positions()
    return SensoryObservation(source.sample(*left), source.sample(*right))
