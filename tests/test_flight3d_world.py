"""3D mechanics, sensory boundaries and reproducible flower trials."""

from dataclasses import fields, replace
import math

import numpy as np
import pytest

from flybrain.flight3d.contracts import FlightCommand, FlightObservation, TrialSpec
from flybrain.flight3d.world import FlightWorld, make_trial


def test_random_flower_preserves_start_and_replay():
    a, b = make_trial(2), make_trial(7)
    assert a == make_trial(2) and a.flower != b.flower
    assert FlightWorld(a).snapshot().position == (0.3, 0.3, 0.1)
    assert FlightWorld(b).snapshot().position == (0.3, 0.3, 0.1)
    a1, a2 = FlightWorld(a), FlightWorld(a)
    for _ in range(12):
        a1.step(FlightCommand(power=.55, pitch=.1, yaw=.2))
        a2.step(FlightCommand(power=.55, pitch=.1, yaw=.2))
    assert a1.snapshot() == a2.snapshot()
    a1.reset()
    assert a1.snapshot() == FlightWorld(a).snapshot()


def test_gravity_hover_climb_and_tilt_are_physics():
    fall, hover, climb, tilt = [FlightWorld(make_trial(10)) for _ in range(4)]
    for _ in range(5):
        fall.step(FlightCommand())
        hover.step(FlightCommand(power=.5))
        climb.step(FlightCommand(power=.6))
        tilt.step(FlightCommand(power=.5, pitch=.3))
    assert fall.position[2] < .1 < climb.position[2]
    assert hover.position[2] == pytest.approx(.1)
    assert tilt.position[1] > .3
    assert hover.physics_steps == 20 and hover.time == pytest.approx(.1)


def test_observation_has_only_sensory_data_and_immutable_images():
    world = FlightWorld(make_trial(4))
    observation = world.observe()
    assert {f.name for f in fields(FlightObservation)} == {
        'odor_left', 'odor_right', 'eyes', 'angular_velocity',
        'specific_force', 'foot_contact', 'food_contact'}
    assert observation.eyes.shape == (2, 24, 48, 3)
    assert not observation.eyes.flags.writeable
    assert np.isfinite(observation.eyes).all()
    with pytest.raises(ValueError):
        observation.eyes[0, 0, 0, 0] = 1
    before = world.snapshot()
    world.observe(); world.observe()
    assert world.snapshot() == before


def test_images_distinguish_above_below_and_left_right():
    images = [FlightWorld(TrialSpec(i, flower=p)).observe().eyes
              for i, p in enumerate(((.4,.42,.16),(.4,.42,.04),(.2,.42,.16)))]
    assert not np.array_equal(images[0], images[1])
    assert not np.array_equal(images[0], images[2])


def test_safe_landing_requires_surface_and_legs_not_nearness():
    case = TrialSpec(10, flower=(.3,.3,.06))
    w = FlightWorld(case)
    w.position[:] = (.3,.3,.063)
    w.velocity[:] = (0,0,-.02)
    w.step(FlightCommand(power=.5, legs=1))
    assert w.foot_contact and w.food_contact and not w.crashed
    for _ in range(101):
        w.step(FlightCommand(legs=1))
    assert w.dwell >= 2 and w.snapshot().velocity == (0.,0.,0.)
    w2 = FlightWorld(case)
    w2.position[:] = (.3,.3,.063)
    w2.velocity[:] = (0,0,-.4)
    w2.step(FlightCommand(legs=1))
    assert w2.crashed and w2.dwell == 0


def test_ground_contact_never_counts_as_food_and_commands_validate():
    w = FlightWorld(TrialSpec(4, flower=(.5,.5,.12)))
    for _ in range(15):
        w.step(FlightCommand())
    assert w.crashed and not w.food_contact and w.dwell == 0
    for kwargs in ({'power':1.1},{'pitch':math.nan},{'yaw':-2},{'legs':-1}):
        with pytest.raises(ValueError): FlightCommand(**kwargs)
    with pytest.raises(ValueError): TrialSpec(1, flower=(.3,.3,.4))
