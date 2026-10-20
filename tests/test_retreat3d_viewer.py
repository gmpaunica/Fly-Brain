"""Real renderer/input checks; saved frames do not establish human live review."""
import importlib
import numpy as np
import torch
import pytest


def test_interactive_controls_continuous_spawn_pause_focus_replay_and_draw(tmp_path):
    assert importlib.util.find_spec('flybrain.retreat3d.viewer') is not None
    from flybrain.retreat3d.viewer import RetreatViewer
    from flybrain.retreat3d.world import PredatorInput
    from panda3d.core import PNMImage, Point2, Point3
    torch.set_num_threads(1)
    app = RetreatViewer(1000001, offscreen=True, window_size=(1000, 700))
    try:
        app.paused = True; app.simulation.pause(); app.load_geometry(100000)
        assert app.session.world.objects == () and app.flower.getNumChildren() == 0
        app.toggle_settings(); app.all_activity(); app.toggle_vnc()
        camera = np.array(app.brain_camera.getMat()); classes = app.enabled_classes
        app.simulation.advance_fixed(20)
        before = app.simulation.display_frame.snapshot
        app.spawn_predator_button['command']()
        assert app.session.snapshot.position == before.position
        assert app.session.snapshot.velocity == before.velocity
        assert not any(app.simulation.display_frame.cumulative_counts)
        assert app.paused and app.settings_open and app.vnc
        np.testing.assert_array_equal(np.array(app.brain_camera.getMat()), camera)
        app.messenger.send('w'); app.messenger.send('d'); app.messenger.send('e')
        assert np.linalg.norm(app.predator_input().velocity) == pytest.approx(.08)
        app.simulation.advance_fixed(8, app.predator_input())
        app.draw_frame(app.simulation.display_frame)
        assert 'escape unavailable' in app.hud.getText().lower()
        assert 'baseline subtraction' in app.activity_label.getText().lower()
        predator = app.flower.find('**/predator-proxy')
        np.testing.assert_allclose(tuple(predator.getPos()), app.session.snapshot.object_position, atol=1e-7)
        frame = app.simulation.display_frame; voltage = app.session.brain.voltage.clone()
        for _ in range(3): app.draw_frame(frame); app.graphicsEngine.renderFrame()
        assert frame is app.simulation.display_frame and torch.equal(voltage, app.session.brain.voltage)
        projected = Point2()
        point = app.cam.getRelativePoint(app.world_scene, Point3(*frame.snapshot.object_position))
        assert app.camLens.project(point, projected)
        region = app.cam.node().getDisplayRegion(0)
        image = PNMImage(); app.win.getScreenshot(image)
        x = int((region.getLeft()+(projected.x+1)/2*(region.getRight()-region.getLeft()))*image.getXSize())
        y = int((1-region.getBottom()-(projected.y+1)/2*(region.getTop()-region.getBottom()))*image.getYSize())
        r, g, b = image.getXel(x, y)
        assert g > r+.2 and b > r+.2, 'moving sphere must render at published environment position'
        app.focus_changed(False)
        assert not app.keys and app.predator_input() == PredatorInput()
        app.keys.update(('w', 'e')); app.toggle_pause()
        assert not app.keys
        app.toggle_pause()
        assert app.paused
        expected = app.session.snapshot
        app.messenger.send('backspace')
        assert app.paused and app.session.snapshot == expected
        app.messenger.send('r')
        assert app.session.world.objects == () and app.session.snapshot.time == 0
        app.spawn_flower_button['command']()
        assert not app.flower.find('**/petal').isEmpty()
        assert app.enabled_classes == classes and app.settings_open
        app.draw_frame(app.simulation.display_frame)
        app.graphicsEngine.renderFrame()
        picture = PNMImage(); app.win.getScreenshot(picture)
        picture.write(str(tmp_path/'review.png'))
        assert picture.getXSize() == 1000
    finally:
        app.destroy()
