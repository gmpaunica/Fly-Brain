"""Numerical checks for independent trial starts and sustained food contact."""

from dataclasses import fields
import importlib
import math
import random

import pytest

from flybrain.world.arena import Arena
from flybrain.world.fly import Fly, MotorCommand
from flybrain.world.odor import OdorSource


def test_episode_sequence_varies_headings_and_replays_without_global_rng():
    module = importlib.import_module('flybrain.behavior')
    before = random.getstate()
    a, b = module.EpisodeSequence(7), module.EpisodeSequence(7)
    seeds, headings = [], []
    for _ in range(64):
        assert a.current_seed == b.current_seed
        seeds.append(a.current_seed)
        headings.append(Arena().spawn(a.current_seed).heading)
        a.next_seed()
        b.next_seed()
    assert seeds[0] == 7
    assert len(set(seeds)) == len(set(headings)) == 64
    assert {int(h / (math.pi / 2)) for h in headings} == {0, 1, 2, 3}
    assert random.getstate() == before


def test_food_contact_is_local_and_observation_contains_only_senses():
    module = importlib.import_module('flybrain.world.food')
    source = OdorSource(600, 300)
    patch = module.FoodPatch(600, 300)
    contact = module.sample_food(Fly(584, 300, math.pi), source, patch)
    assert contact.food_contact
    assert not module.sample_food(Fly(583.999, 300), source, patch).food_contact
    assert not module.sample_food(Fly(600, 300), source, None).food_contact
    assert module.sample_food(Fly(600, 300), None, patch).food_contact
    assert {f.name for f in fields(contact)} == {'odor_left', 'odor_right', 'food_contact'}
    with pytest.raises(ValueError):
        module.FoodObservation(0, 0, 1)
    with pytest.raises(ValueError):
        module.FoodPatch(0, 0, -1)


def test_speed_control_supports_slowing_and_exact_zero():
    fly = Fly(100, 100)
    fly.step(MotorCommand(.7, 0), .02)
    assert fly.forward_speed == 70
    fly.step(MotorCommand(.2, 0), .02)
    assert fly.forward_speed == 20
    x = fly.x
    fly.step(MotorCommand(0, 0), .02)
    assert fly.x == x and fly.forward_speed == 0


def test_dwell_requires_two_seconds_and_does_not_freeze_after_contact():
    module = importlib.import_module('flybrain.behavior')
    visit = module.FoodVisit()
    for tick in range(99):
        visit.observe(tick + 1, True, MotorCommand(), 0.)
    assert not visit.settled and visit.hold_seconds == pytest.approx(1.98)
    visit.observe(100, True, MotorCommand(), 0.)
    assert visit.settled and visit.hold_seconds == 2
    visit.observe(101, False, MotorCommand(), 0.)
    assert not visit.settled and visit.hold_seconds == 0
    assert visit.first_contact_tick == 1 and visit.first_settled_tick == 100


@pytest.mark.parametrize('command,displacement', [
    (MotorCommand(.7, 0), 70.),  # Pass through food.
    (MotorCommand(.7, 0), 0.),   # A wall cannot masquerade as stopping.
    (MotorCommand(0, .2), 0.),  # Turning in place is not stationary.
    (MotorCommand(), 2.),      # Actual movement must also be small.
])
def test_contact_alone_or_wall_stall_cannot_pass_stop(command, displacement):
    visit = importlib.import_module('flybrain.behavior').FoodVisit()
    for tick in range(120):
        visit.observe(tick + 1, True, command, displacement)
    assert visit.first_contact_tick == 1
    assert not visit.settled and visit.longest_hold_ticks == 0


def test_duplicate_or_missing_ticks_cannot_manufacture_dwell_time():
    visit = importlib.import_module('flybrain.behavior').FoodVisit()
    visit.observe(1, True, MotorCommand(), 0.)
    for invalid in (1, 3):
        with pytest.raises(ValueError):
            visit.observe(invalid, True, MotorCommand(), 0.)
    assert visit.hold_ticks == 1
    visit.observe(2, True, MotorCommand(), 0.)
    assert visit.hold_ticks == 2


def test_rich_senses_reach_controller_with_real_neural_steps_and_no_motor_override():
    from flybrain.neural.verified_banc import load_frozen_controller
    from flybrain.neural.controller import NeuralController
    from flybrain.simulation import FixedStepSimulation

    module = importlib.import_module('flybrain.foraging')
    base = load_frozen_controller()

    class ObservingController(NeuralController):
        def decide(self, observation):
            self.observed = observation
            self.decision = super().decide(observation)
            return self.decision

    controller = ObservingController(base.network, base.decoder)
    world = FixedStepSimulation(Arena(), Fly(600, 300), odor_source=OdorSource(600, 300))
    loop = module.ForagingSimulation(world, controller)
    snapshot = loop.step()
    assert controller.observed.food_contact
    assert snapshot.command == controller.decision.command
    assert snapshot.command.forward > 0 and snapshot.activity.end_step == 20
    assert loop.visit.contact and not loop.visit.settled
    loop.advance_snapshots(.009)
    loop.reset(7)
    assert loop.visit.first_contact_tick is None and loop.visit.hold_ticks == 0
    assert loop.controller.network.snapshot().step_count == loop.world.tick_count == 0
    assert loop.advance_snapshots(.011) == ()
