"""Optional encounters, ground landing safety and environment-only movement."""
import importlib
import math
import numpy as np
import pytest
from flybrain.flight3d.contracts import FlightCommand, FOOT_LENGTH, PHYSICS_DT


def api():
    assert importlib.util.find_spec('flybrain.retreat3d.world') is not None
    return importlib.import_module('flybrain.retreat3d.world')


def test_initial_rest_and_no_object_continuous_spawn():
    m = api(); world = m.RetreatWorld(10)
    assert world.objects == () and world.foot_contact
    assert world.position[2] == FOOT_LENGTH
    world.velocity[:] = (.01, .02, .03)
    world.orientation[:] = (.1, .2, .3)
    before = world.snapshot()
    world.spawn('predator')
    assert len(world.objects) == 1
    assert world.snapshot().position == before.position
    assert world.snapshot().velocity == before.velocity
    assert world.snapshot().orientation == before.orientation
    world.spawn('flower')
    assert world.scene.kind == 'flower' and len(world.objects) == 1
    assert world.observe().food_contact is False


@pytest.mark.parametrize('legs,vx,vz,tilt,safe', [
    (1, .02, -.03, 0., True), (0, .02, -.03, 0., False),
    (1, .09, -.03, 0., False), (1, .02, -.09, 0., False),
    (1, .02, -.03, math.radians(22), False)])
def test_ground_contact_safety(legs, vx, vz, tilt, safe):
    m = api(); w = m.RetreatWorld(11)
    w.foot_contact = False; w.position[2] = FOOT_LENGTH+.00005
    w.velocity[:] = (vx, 0., vz); w.orientation[0] = tilt
    w._physics(FlightCommand(.5, roll=tilt/.5, legs=legs))
    assert w.crashed is not safe and w.foot_contact is safe
    if safe:
        w.step(FlightCommand(legs=1))
        assert not w.crashed and not np.any(w.velocity)
        w.step(FlightCommand(.7, legs=1))
        assert not w.foot_contact and w.position[2] > FOOT_LENGTH


def test_diagonal_controls_arena_limits_and_relative_sweep():
    m = api(); w = m.RetreatWorld(12)
    w.spawn('predator', (.3, .5, .1))
    command = m.PredatorInput.from_keys({'w', 'd', 'e'})
    assert np.linalg.norm(command.velocity) == pytest.approx(.08)
    w.step(FlightCommand(legs=1), command)
    np.testing.assert_allclose(np.array(w.scene.position)-(.3, .5, .1), np.array(command.velocity)*.02)
    w.spawn('predator', (.582, .582, .282))
    for _ in range(10): w.step(FlightCommand(legs=1), command)
    assert max(w.scene.position[:2]) <= .582 and w.scene.position[2] <= .282
    # Predator sweeps into a stationary resting fly. Endpoint-only testing misses
    # this explicit crossing with a larger diagnostic physics interval input.
    w = m.RetreatWorld(13); w.position[:] = (.3, .3, .03)
    w.foot_contact = False; w.spawn('predator', (.3196, .3, .03))
    w.step(FlightCommand(.5), m.PredatorInput.from_keys({'a'}))
    assert w.crashed
    assert 'predator' not in vars(w.observe())


def test_relative_sweep_detects_crossing_without_endpoint_overlap():
    m = api()
    assert m.swept_contact(np.array((-.1, 0, 0)), np.array((.1, 0, 0)),
                           np.array((.1, 0, 0)), np.array((-.1, 0, 0)), .01)
    assert not m.swept_contact(np.array((-.1, .02, 0)), np.array((.1, .02, 0)),
                               np.array((.1, 0, 0)), np.array((-.1, 0, 0)), .01)


def test_replacing_a_supporting_flower_releases_contact_without_pose_reset():
    m = api(); w = m.RetreatWorld(14)
    w.spawn('flower', (.3, .3, .1))
    w.position[2] = .1+FOOT_LENGTH; w.foot_contact = True; w.food_contact = True
    before = w.snapshot()
    w.spawn('predator', (.3, .48, .1))
    assert w.snapshot().position == before.position
    assert not w.observe().foot_contact
    w.step(FlightCommand(legs=1))
    assert w.position[2] < before.position[2]


def test_flower_flight_physics_and_sensors_match_original_above_ground():
    m = api()
    from flybrain.flight3d.world import FlightWorld
    from flybrain.flight3d.contracts import TrialSpec
    position = (.4, .45, .08)
    old = FlightWorld(TrialSpec(15, position)); new = m.RetreatWorld(15)
    new.spawn('flower', position)
    new.position[:] = old.position; new.foot_contact = False; new.specific_force[:] = old.specific_force
    for _ in range(50):
        command = FlightCommand(.5, pitch=.01, roll=-.01, yaw=.02, legs=1)
        old.step(command); new.step(command)
        for field in ('position', 'orientation', 'velocity', 'angular_velocity', 'specific_force'):
            np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
        np.testing.assert_array_equal(old.observe().eyes, new.observe().eyes)
        assert old.observe().odor_left == new.observe().odor_left
