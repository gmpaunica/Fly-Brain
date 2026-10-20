"""World contact sensing and behavioral observation around unchanged controllers."""

import math

from .behavior import FoodVisit
from .closed_loop import BodyState, ClosedLoopSimulation, ClosedLoopSnapshot
from .neural.controller import NeuralController
from .simulation import FixedStepSimulation
from .world.food import FoodPatch, sample_food


class ForagingSimulation(ClosedLoopSimulation):
    """Supply local food sensing and measure contact without overriding movement.

    The historical controller accepts the richer observation but uses only its
    two odor values. Learning the contact response is a separate pending task.
    """

    def __init__(self, world: FixedStepSimulation, controller: NeuralController) -> None:
        super().__init__(world, controller)
        source = world.odor_source
        self.food = FoodPatch(source.x, source.y) if source is not None else None
        self.visit = FoodVisit()
        self._refresh_contact()

    def _refresh_contact(self) -> None:
        self.world.observation = sample_food(self.world.fly, self.world.odor_source, self.food)
        self.visit.contact = self.world.observation.food_contact

    def step(self) -> ClosedLoopSnapshot:
        """Sample senses, run neurons, apply their command, then measure the tick."""
        fly = self.world.fly
        previous_x, previous_y = fly.x, fly.y
        observation = sample_food(fly, self.world.odor_source, self.food)
        decision = self.controller.decide(observation)
        collision = self.world.step(decision.command)
        self._refresh_contact()
        self.visit.observe(self.world.tick_count, self.world.observation.food_contact,
            decision.command, math.hypot(fly.x - previous_x, fly.y - previous_y) / self.world.dt)
        snapshot = ClosedLoopSnapshot(self.world.tick_count, observation, decision.activity,
                                      decision.command, collision, BodyState.from_fly(fly))
        self.last_snapshot = snapshot
        return snapshot

    def reset(self, seed: int = 0) -> None:
        """Clear all episode state, including metric history, retaining weights."""
        super().reset(seed)
        self.visit = FoodVisit()
        self._refresh_contact()
