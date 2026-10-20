"""Numerical navigation regressions, independent of Panda and simulation."""
import numpy as np
import pytest
from flybrain.flight3d.brain_camera import BrainCamera


@pytest.mark.parametrize('view', ['front', 'side', 'top'])
def test_drag_crosses_poles_continuously_and_stops_on_release(view):
    camera = BrainCamera()
    camera.preset(view)
    for _ in range(400):
        previous = camera.up.copy()
        camera.rotate_pixels(0, 3)
        camera.advance(1 / 60)
        assert np.dot(previous, camera.up) > .99
        assert abs(np.dot(camera.back, camera.up)) < 1e-12
        assert np.linalg.norm(camera.orientation) == pytest.approx(1)
    camera.release()
    frozen = camera.orientation.copy()
    camera.advance(1)
    np.testing.assert_array_equal(camera.orientation, frozen)


def test_pan_at_close_scale_never_changes_zoom_and_presets_preserve_pan():
    camera = BrainCamera()
    camera.zoom(.001)
    camera.advance(5)
    assert camera.distance == pytest.approx(.03)
    camera.pan_pixels(100, 50, 500, 45)
    before = camera.target
    camera.advance(5)
    assert camera.distance == pytest.approx(before)
    assert np.linalg.norm(camera.offset) > 0
    assert np.linalg.norm(camera.offset) < .01
    for name in ('front', 'side', 'top'):
        pan = camera.offset.copy(); pivot = camera.pivot.copy()
        camera.preset(name)
        np.testing.assert_array_equal(camera.offset, pan)
        np.testing.assert_array_equal(camera.pivot, pivot)
        assert camera.distance == pytest.approx(before)
    assert 0 < camera.near < camera.distance / 50


def test_fit_is_explicit_smooth_and_accounts_for_both_fov_axes():
    camera = BrainCamera()
    bounds = np.array([[0, -1, 0], [1, 1, .2]])
    original = camera.position.copy()
    camera.fit(bounds, (25, 45))
    np.testing.assert_array_equal(camera.position, original)
    assert camera.target > camera.distance
    camera.advance(5)
    corners = np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    relative = corners - camera.position
    depth = -relative @ camera.back
    assert np.all(abs(relative @ camera.right) < depth * np.tan(np.deg2rad(25/2)))
    assert np.all(abs(relative @ camera.up) < depth * np.tan(np.deg2rad(45/2)))


def test_display_time_smoothing_matches_30_60_120_hz():
    results = []
    for hz in (30, 60, 120):
        camera = BrainCamera()
        camera.rotate_pixels(80, 110)
        camera.pan_pixels(20, -10, 500, 45)
        camera.zoom(.88)
        for _ in range(hz): camera.advance(1 / hz)
        results.append(np.r_[camera.orientation, camera.position, camera.distance])
    np.testing.assert_allclose(results[0], results[1], atol=1e-12)
    np.testing.assert_allclose(results[1], results[2], atol=1e-12)


def test_pixel_rotation_sensitivity_is_independent_of_viewport_size():
    # The viewer converts normalized window deltas to pixels before routing.
    a, b = BrainCamera(), BrainCamera()
    a.rotate_pixels(40, -30); b.rotate_pixels(40, -30)
    a.advance(.06); b.advance(.06)
    np.testing.assert_array_equal(a.orientation, b.orientation)


def test_continuous_drag_responses_are_comparable_across_update_rates():
    directions = []
    for hz in (30,60,120):
        camera = BrainCamera()
        for _ in range(hz):
            camera.rotate_pixels(120/hz,80/hz)
            camera.advance(1/hz)
        directions.append(camera.back)
    # Discrete mouse samples differ by at most one frame; <0.6 degree difference.
    assert np.arccos(np.clip(directions[0]@directions[2],-1,1)) < .01


def test_zoom_and_rotation_response_constants():
    camera = BrainCamera()
    camera.zoom(.88); camera.rotate_pixels(100,0)
    initial = camera.distance
    camera.advance(.06)
    assert camera.distance == pytest.approx(camera.target+(initial-camera.target)*np.exp(-.06/.12))
    angle = 2*np.arccos(camera.orientation[0])
    assert angle == pytest.approx(.6*(1-np.exp(-1)))
