"""Repeated seeded idle cycles and sensory-estimate transfer to the teacher."""
import importlib
import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('flybrain.retreat3d.controller') is not None
    return importlib.import_module('flybrain.retreat3d.controller')


def test_seeded_idle_takeoff_cruise_safe_land_rest_and_envelope():
    m = api()
    from flybrain.retreat3d.world import RetreatWorld
    histories = []
    for _ in range(2):
        w = RetreatWorld(100); c = m.IdleFlight(100)
        states = set(); history = []; max_horizontal = 0.
        for i in range(1750):
            command = c.command(w.observe())
            s = w.step(command)
            states.add(c.state); history.append((c.state, s.position))
            max_horizontal = max(max_horizontal, np.linalg.norm(np.array(s.position[:2])-(.3, .3)))
            assert not s.crashed, (i, c.state, s)
        assert {'takeoff', 'cruising', 'landing', 'resting'} <= states
        assert max_horizontal < .22
        assert any(state == 'cruising' and .08 < p[2] < .12 for state, p in history)
        assert all(3 <= duration <= 6 for duration in c.cruise_durations)
        assert all(2 <= duration <= 5 for duration in c.rest_durations)
        assert len(c.cruise_durations) >= 2
        histories.append(history)
    assert histories[0] == histories[1]


def test_shared_estimate_and_flower_handoff_no_world_initialization():
    m = api()
    from flybrain.retreat3d.world import RetreatWorld
    w = RetreatWorld(101); c = m.BehaviorController(101)
    for _ in range(160): w.step(c.command(w.observe()))
    estimate = c.idle.estimate
    orientation = estimate.orientation.copy(); velocity = estimate.velocity.copy()
    altitude = estimate.altitude
    w.spawn('flower'); c.encounter('flower')
    assert c.teacher is not None
    np.testing.assert_array_equal(c.teacher.orientation, orientation)
    np.testing.assert_array_equal(c.teacher.velocity, velocity)
    assert c.teacher.altitude == altitude
    assert not c.teacher.first
    assert c.idle.estimate is estimate


def test_grounded_flower_takes_off_before_teacher_commands():
    m = api()
    from flybrain.retreat3d.world import RetreatWorld
    w = RetreatWorld(102); c = m.BehaviorController(102)
    w.spawn('flower'); c.encounter('flower')
    command = c.command(w.observe())
    assert command.power > .5 and c.state == 'flower-takeoff'
    for _ in range(220): w.step(c.command(w.observe()))
    assert c.teacher is not None and c.state in ('flower', 'flower-resting')
