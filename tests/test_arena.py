import math
import random

import pytest

from flybrain.world.arena import Arena
from flybrain.world.fly import Fly, MotorCommand


@pytest.mark.parametrize('x,y,h,want', [(90,50,0,(94,50)), (10,50,math.pi,(6,50)), (50,90,math.pi/2,(50,94)), (50,10,3*math.pi/2,(50,6)), (92,92,math.pi/4,(94,94))])
def test_walls_and_corner(x,y,h,want):
    fly = Fly(x,y,h)
    assert Arena(100,100).step(fly, MotorCommand(1), 0.1)
    assert (fly.x,fly.y) == pytest.approx(want)
    assert fly.heading == pytest.approx(h)
    assert fly.forward_speed == 100


def test_interior_and_exact_contact_are_not_crossings():
    arena = Arena(100,100)
    fly = Fly(84,50)
    assert not arena.step(fly, MotorCommand(1), 0.1)
    assert fly.x == 94
    assert not arena.step(fly, MotorCommand(), 0.1)


def test_spawn_centered_repeatable_and_local_rng():
    before = random.getstate()
    a, b = Arena().spawn(7), Arena().spawn(7)
    assert a == b
    assert (a.x,a.y,a.radius) == (400,300,6)
    assert a.heading == pytest.approx(2.034701269983068)
    assert random.getstate() == before


@pytest.mark.parametrize('kwargs', [dict(width=0),dict(height=-1),dict(width=math.nan),dict(height=math.inf)])
def test_bad_dimensions(kwargs):
    with pytest.raises(ValueError):
        Arena(**kwargs)


def test_body_must_fit_before_motion_or_spawn():
    with pytest.raises(ValueError):
        Arena(11,100).spawn()
    fly = Fly(5,50)
    with pytest.raises(ValueError):
        Arena(11,100).step(fly,MotorCommand(1),0.1)
    assert fly.x == 5
    assert Arena(12,12).spawn().x == 6
