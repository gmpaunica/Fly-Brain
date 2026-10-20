import math
from dataclasses import asdict

import pytest

from flybrain.world.arena import Arena
from flybrain.world.fly import Fly, MotorCommand
from flybrain.simulation import FixedStepSimulation


def make_sim():
    return FixedStepSimulation(Arena(), Fly(100,100))


def test_frame_partitions_produce_same_ticks_and_pose():
    a, b = make_sim(), make_sim()
    command = MotorCommand(1,0.25)
    assert a.advance(0.1,command) == 5
    assert sum(b.advance(t,command) for t in [0.01,0.03,0.06]) == 5
    assert a.tick_count == b.tick_count == 5
    assert asdict(a.fly) == asdict(b.fly)


def test_partial_time_and_single_tick():
    sim = make_sim()
    assert sim.advance(0.01,MotorCommand(1)) == 0
    assert sim.advance(0,MotorCommand(1)) == 0
    assert sim.advance(0.01,MotorCommand(1)) == 1
    assert sim.fly.x == 102
    sim.step(MotorCommand(1))
    assert sim.tick_count == 2
    assert sim.fly.x == 104


def test_reset_clears_time_body_and_tick_count():
    sim = make_sim()
    sim.advance(0.03,MotorCommand(1,1))
    sim.reset(7)
    assert sim.tick_count == 0
    assert sim.fly == Arena().spawn(7)
    assert sim.advance(0.01,MotorCommand()) == 0


def test_seeded_per_tick_replay():
    def replay():
        sim = make_sim()
        sim.reset(9)
        trajectory = []
        for command in [MotorCommand(1), MotorCommand(1,-0.5),MotorCommand(),MotorCommand(0,1)] * 30:
            sim.step(command)
            trajectory.append(asdict(sim.fly))
        return trajectory
    assert replay() == replay()


@pytest.mark.parametrize('dt', [0,-1,math.nan,math.inf])
def test_invalid_fixed_time(dt):
    with pytest.raises(ValueError):
        FixedStepSimulation(Arena(),Fly(100,100),dt)


@pytest.mark.parametrize('elapsed', [-1,math.nan,math.inf])
def test_invalid_elapsed(elapsed):
    sim = make_sim()
    with pytest.raises(ValueError):
        sim.advance(elapsed,MotorCommand())
    assert sim.tick_count == 0


def test_sensory_readings_follow_current_pose_after_motion():
    from flybrain.world.odor import OdorSource
    source = OdorSource(110,95,100,2)
    sim = FixedStepSimulation(Arena(),Fly(100,100),odor_source=source)
    assert (sim.observation.odor_left,sim.observation.odor_right) == pytest.approx((25,100/104))
    sim.step(MotorCommand(1))
    assert sim.fly.x == 102
    assert (sim.observation.odor_left,sim.observation.odor_right) == pytest.approx((12.5,100/108))


def test_odor_reset_replay_and_frame_partitions():
    from flybrain.world.odor import OdorSource
    source = OdorSource(600,300)
    a = FixedStepSimulation(Arena(),Fly(100,100),odor_source=source)
    b = FixedStepSimulation(Arena(),Fly(100,100),odor_source=source)
    a.reset(7)
    b.reset(7)
    initial = a.observation
    a.advance(0.1,MotorCommand(1,0.25))
    for t in [0.01,0.03,0.06]:
        b.advance(t,MotorCommand(1,0.25))
    assert a.observation == b.observation
    assert a.observation != initial
    a.reset(7)
    assert a.observation == initial
    assert a.odor_source is source
    assert a.tick_count == 0
    assert a.advance(0.01,MotorCommand()) == 0
    def replay(sim):
        sim.reset(7)
        readings = []
        for cmd in [MotorCommand(1,-1),MotorCommand(1,1),MotorCommand()] * 20:
            sim.step(cmd)
            readings.append(sim.observation)
        return readings
    assert replay(a) == replay(b)


def test_no_source_produces_zero_sensory_input():
    sim = make_sim()
    assert (sim.observation.odor_left,sim.observation.odor_right) == (0,0)
    sim.step(MotorCommand(1))
    assert (sim.observation.odor_left,sim.observation.odor_right) == (0,0)
