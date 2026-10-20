import math
from dataclasses import asdict, FrozenInstanceError

import pytest

from flybrain.world.fly import Fly, MotorCommand


def test_forward_motion_uses_seconds():
    fly = Fly(100, 100)
    fly.step(MotorCommand(1), 0.5)
    assert (fly.x, fly.y, fly.forward_speed) == pytest.approx((150, 100, 100))


def test_turn_before_translation_and_downward_heading():
    fly = Fly(100, 100)
    fly.step(MotorCommand(1, 1), 0.5)
    assert (fly.x, fly.y, fly.heading) == pytest.approx((100, 150, math.pi / 2))
    fly.step(MotorCommand(1), 0.1)
    assert (fly.x, fly.y) == pytest.approx((100, 160))


def test_left_wrap_stop_and_zero_time():
    fly = Fly(100, 100)
    fly.step(MotorCommand(1, -1), 0.5)
    assert fly.heading == pytest.approx(3 * math.pi / 2)
    before = asdict(fly)
    fly.step(MotorCommand(), 0)
    assert asdict(fly) == before
    fly.step(MotorCommand(), 0.1)
    assert (fly.forward_speed, fly.angular_speed) == (0, 0)


@pytest.mark.parametrize('dt', [-1, math.nan, math.inf, -math.inf])
def test_bad_time_rejected_without_mutation(dt):
    fly = Fly(100, 100)
    before = asdict(fly)
    with pytest.raises(ValueError):
        fly.step(MotorCommand(1), dt)
    assert asdict(fly) == before


@pytest.mark.parametrize('kwargs', [dict(forward=-0.1), dict(forward=1.1), dict(turn=-1.1), dict(turn=1.1), dict(forward=math.nan), dict(turn=math.inf)])
def test_bad_commands(kwargs):
    with pytest.raises(ValueError):
        MotorCommand(**kwargs)


def test_command_is_immutable():
    with pytest.raises(FrozenInstanceError):
        MotorCommand().forward = 1


@pytest.mark.parametrize('kwargs', [dict(x=math.nan), dict(y=math.inf), dict(heading=math.nan), dict(radius=0), dict(radius=-1), dict(radius=math.inf)])
def test_bad_body(kwargs):
    values = dict(x=10, y=10)
    values.update(kwargs)
    with pytest.raises(ValueError):
        Fly(**values)


def test_constructor_normalizes_heading():
    assert Fly(10, 10, -math.pi / 2).heading == pytest.approx(3 * math.pi / 2)
