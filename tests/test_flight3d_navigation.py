"""Full-rectangle wheel routing and bounded display-only camera interpolation."""
import numpy as np
from flybrain.flight3d import navigation


def test_full_rectangles_route_details_and_blank_gutters_without_camera_zoom():
    layout=navigation.PanelLayout()
    assert layout.route(-.5,.3)=='world'
    assert layout.route(.6,.3)=='brain'
    assert layout.route(.6,-.6)=='details'
    assert layout.route(-.5,-.6)=='help'
    assert layout.route(.6,.95) is None
    assert layout.route(.0,.3) is None


def test_smooth_zoom_clamps_and_pan_stays_in_registered_bounds():
    camera=navigation.CameraMotion(.96,.65,3.)
    camera.zoom(.001)
    assert camera.target==.65 and camera.distance==.96
    camera.advance(.016)
    assert .65<camera.distance<.96
    for _ in range(200): camera.advance(.016)
    assert abs(camera.distance-.65)<1e-8
    camera.pan(np.array([100.,-100.,0.]))
    assert np.linalg.norm(camera.offset)<=.25+1e-12
    assert camera.distance>=camera.minimum+np.linalg.norm(camera.offset)-1e-12
    camera.reset(.96)
    assert camera.distance==camera.target==.96 and not camera.offset.any()
