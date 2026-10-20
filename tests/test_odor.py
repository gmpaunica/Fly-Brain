import math
from dataclasses import FrozenInstanceError

import pytest

from flybrain.world.odor import OdorSource


def test_softened_field_expected_values_and_monotonicity():
    source = OdorSource(0, 0, strength=100, softening=2)
    assert source.sample(0, 0) == pytest.approx(25)
    assert source.sample(2, 0) == pytest.approx(12.5)
    assert source.sample(4, 0) == pytest.approx(5)
    assert source.sample(0, 0) > source.sample(2, 0) > source.sample(4, 0)


@pytest.mark.parametrize('x,y', [(3,4),(-3,4),(3,-4),(-3,-4),(0,5)])
def test_radial_symmetry_and_translation(x, y):
    assert OdorSource(10,20,100,2).sample(10+x,20+y) == pytest.approx(100/29)


@pytest.mark.parametrize('kwargs', [dict(x=math.nan),dict(y=math.inf),dict(strength=-1),dict(strength=math.inf),dict(softening=0),dict(softening=-1),dict(softening=math.nan),dict(softening=1e-300)])
def test_invalid_source(kwargs):
    values = dict(x=0,y=0)
    values.update(kwargs)
    with pytest.raises(ValueError):
        OdorSource(**values)


@pytest.mark.parametrize('x,y', [(math.inf,0),(0,math.nan)])
def test_invalid_sampling_coordinates(x,y):
    with pytest.raises(ValueError):
        OdorSource(0,0).sample(x,y)


def test_large_finite_distance_and_zero_strength():
    source = OdorSource(-1e308,0,100,2)
    assert source.sample(1e308,0) == 0
    assert math.isfinite(source.sample(-1e308,0))
    assert OdorSource(0,0,0,1e-300).sample(0,0) == 0


def test_source_is_static():
    with pytest.raises(FrozenInstanceError):
        OdorSource(0,0).x = 1
