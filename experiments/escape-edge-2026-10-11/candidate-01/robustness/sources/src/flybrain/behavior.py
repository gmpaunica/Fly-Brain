"""Replayable episode seeds and read-only measures of stationary food contact."""

from dataclasses import dataclass, field
import math
import random
import secrets

from .world.fly import MAX_ANGULAR_SPEED, MAX_FORWARD_SPEED, MotorCommand


class EpisodeSequence:
    """New headings per trial, exact replay with a recorded master seed.

    The first episode uses the master seed directly. Later seeds come from a
    local stream. Equal seeds and headings are rejected within this session.
    Unspecified master seeds use system entropy once; rendering uses no RNG.
    """

    def __init__(self, seed: int | None = None) -> None:
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError('episode seed must be a nonnegative integer')
        self.master_seed = secrets.randbits(32) if seed is None else seed
        self.current_seed = self.master_seed
        self.episode_number = 1
        self._rng = random.Random(self.master_seed)
        self._seeds = {self.current_seed}
        self._headings = {random.Random(self.current_seed).uniform(0, math.tau)}

    def next_seed(self) -> int:
        """Advance the local sequence once; replay uses current_seed directly."""
        while True:
            seed = self._rng.getrandbits(32)
            heading = random.Random(seed).uniform(0, math.tau)
            if seed not in self._seeds and heading not in self._headings:
                break
        self.current_seed = seed
        self.episode_number += 1
        self._seeds.add(seed)
        self._headings.add(heading)
        return seed


@dataclass
class FoodVisit:
    """Observe a 50 Hz episode without choosing commands or ending simulation.

    A stationary visit requires contact plus commanded and measured speeds
    <=1 unit/s and absolute angular speed <=0.05 rad/s for 100 ticks (2 s).
    A wall stall, pass-through, or continuing rotation cannot satisfy it.
    Thresholds are engineering acceptance criteria, not biological estimates.
    """

    contact: bool = False
    commanded_speed: float = 0.0
    displacement_speed: float = 0.0
    hold_ticks: int = 0
    longest_hold_ticks: int = 0
    first_contact_tick: int | None = None
    first_settled_tick: int | None = None
    _last_tick: int = field(default=0, init=False, repr=False)

    @property
    def hold_seconds(self) -> float:
        """Current uninterrupted stationary contact duration."""
        return self.hold_ticks * .02

    @property
    def settled(self) -> bool:
        """Whether stationary contact is sustained now, not merely in the past."""
        return self.hold_ticks >= 100

    def observe(self, tick: int, contact: bool, command: MotorCommand,
                displacement_speed: float) -> None:
        """Record post-movement contact and the actual command for one tick."""
        if (type(tick) is not int or tick != self._last_tick + 1 or type(contact) is not bool
                or not math.isfinite(displacement_speed) or displacement_speed < 0):
            raise ValueError('expected next consecutive tick, boolean contact and finite speed >=0')
        self._last_tick = tick
        self.contact = contact
        self.commanded_speed = command.forward * MAX_FORWARD_SPEED
        self.displacement_speed = displacement_speed
        if contact and self.first_contact_tick is None:
            self.first_contact_tick = tick
        stationary = (contact and self.commanded_speed <= 1.
                      and displacement_speed <= 1.
                      and abs(command.turn) * MAX_ANGULAR_SPEED <= .05)
        self.hold_ticks = self.hold_ticks + 1 if stationary else 0
        self.longest_hold_ticks = max(self.longest_hold_ticks, self.hold_ticks)
        if self.settled and self.first_settled_tick is None:
            self.first_settled_tick = tick
