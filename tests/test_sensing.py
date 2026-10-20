import math
from dataclasses import asdict, FrozenInstanceError

import pytest

from flybrain.world.fly import Fly
from flybrain.world.odor import OdorSource
from flybrain.world.sensing import SensoryObservation, sample_odor


@pytest.mark.parametrize('heading,left,right', [
    (0,(110,95),(110,105)),
    (math.pi/2,(105,110),(95,110)),
    (math.pi,(90,105),(90,95)),
    (3*math.pi/2,(95,90),(105,90)),
])
def test_antennae_follow_screen_rotation(heading,left,right):
    actual_left, actual_right = Fly(100,100,heading).antenna_positions()
    assert actual_left == pytest.approx(left)
    assert actual_right == pytest.approx(right)


def test_each_antenna_samples_its_own_point():
    fly = Fly(100,100)
    left_source = OdorSource(110,95,100,2)
    right_source = OdorSource(110,105,100,2)
    left_obs = sample_odor(fly,left_source)
    right_obs = sample_odor(fly,right_source)
    assert (left_obs.odor_left,left_obs.odor_right) == pytest.approx((25,100/104))
    assert (right_obs.odor_left,right_obs.odor_right) == pytest.approx((100/104,25))


def test_symmetric_source_and_sampling_does_not_mutate_body():
    fly = Fly(100,100)
    before = asdict(fly)
    obs = sample_odor(fly,OdorSource(110,100,100,2))
    assert (obs.odor_left,obs.odor_right) == pytest.approx((100/29,100/29))
    assert asdict(fly) == before


def test_antennae_are_not_clamped_at_wall():
    left,right = Fly(6,6,math.pi).antenna_positions()
    assert left == pytest.approx((-4,11))
    assert right == pytest.approx((-4,1))
    assert sample_odor(Fly(6,6,math.pi),OdorSource(50,50)).odor_left > 0


@pytest.mark.parametrize('left,right', [(-1,0),(0,-1),(math.nan,0),(0,math.inf)])
def test_invalid_observations(left,right):
    with pytest.raises(ValueError):
        SensoryObservation(left,right)


def test_observation_has_only_immutable_sensory_values():
    obs = SensoryObservation(1,2)
    assert asdict(obs) == dict(odor_left=1,odor_right=2)
    with pytest.raises(FrozenInstanceError):
        obs.odor_left = 4
