"""RGB-only angular measurements, seams, occlusion and weak-threat hysteresis."""
import importlib
import math
import numpy as np
import pytest
from flybrain.encounters3d.scene import encounter_eyes, SceneSpec


def api():
    assert importlib.util.find_spec('flybrain.retreat3d.threat') is not None
    return importlib.import_module('flybrain.retreat3d.threat')


def image(azimuth, radius, elevation=0.):
    angle = math.radians(azimuth); el = math.radians(elevation)
    p = (.3+radius*math.sin(angle)*math.cos(el),
         .3+radius*math.cos(angle)*math.cos(el), .1+radius*math.sin(el))
    return encounter_eyes(np.array((.3, .3, .1)), np.zeros(3), SceneSpec(1, 'predator', p))


@pytest.mark.parametrize('azimuth', [0, 179, -179, 90, -90])
@pytest.mark.parametrize('elevation', [-20, 0, 20])
def test_angular_size_rear_seams_vertical_and_no_distance_output(azimuth, elevation):
    m = api()
    small = m.angular_size(image(azimuth, .17, elevation))
    large = m.angular_size(image(azimuth, .06, elevation))
    assert 5 < small < 20 and 20 < large < 50
    assert large > small


def test_monitor_occlusion_shrinking_reappearance_and_weak_continuity():
    m = api(); monitor = m.ThreatMonitor()
    values = [monitor.update(image(0, .06)) for _ in range(10)]
    value = values[-1]
    assert value.size_degrees >= 20
    assert any(v.strong for v in values)
    assert not value.strong
    blank = np.zeros((2, 24, 48, 3), dtype=np.float32)
    for _ in range(10): value = monitor.update(blank)
    assert not value.weak_continuous
    for _ in range(20): value = monitor.update(blank)
    assert value.weak_continuous
    values = [monitor.update(image(0, .06)) for _ in range(10)]
    assert any(v.strong for v in values)
    for _ in range(30): value = monitor.update(image(0, .25))
    assert value.positive_growth_degrees_s <= 5
    assert value.weak_continuous
    assert not value.strong
    monitor.reset()
    assert monitor.update(blank).size_degrees == 0
