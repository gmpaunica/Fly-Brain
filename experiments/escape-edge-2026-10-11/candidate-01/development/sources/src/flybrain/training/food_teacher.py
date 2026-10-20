"""Deterministic sensory-only demonstrations for approach and contact stop."""

import math

from flybrain.world.fly import MotorCommand
from flybrain.world.food import FoodObservation


class FoodTeacher:
    """Hand-written data-collection scaffold; never receives a world pose."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear the previous odor used to detect falling equal-sided signals."""
        self.previous_mean: float | None = None
        self.searching = False

    def command(self, observation: FoodObservation) -> MotorCommand:
        """Return decreasing speed, contrast turn, search turn or exact stop."""
        if not isinstance(observation, FoodObservation):
            raise ValueError('food sensory observation required')
        mean = (observation.odor_left + observation.odor_right) / 2
        falling = self.previous_mean is not None and mean < self.previous_mean - 1e-7
        rising = self.previous_mean is not None and mean > self.previous_mean + 1e-7
        self.previous_mean = mean
        if observation.food_contact:
            return MotorCommand()
        contrast = ((observation.odor_right - observation.odor_left)
                    / max(observation.odor_right + observation.odor_left, 1e-9))
        turn = max(-1., min(1., 8. * contrast))
        if abs(contrast) < .01 and falling:
            self.searching = True
        if self.searching and rising:
            self.searching = False
        if self.searching:
            turn = .8
        forward = .2 + .5 * math.exp(-mean / .3)
        return MotorCommand(forward, turn)
