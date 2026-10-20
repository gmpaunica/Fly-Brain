"""Actual dashboard widgets and rendering filters preserve the frozen session."""
import importlib

import numpy as np
import pytest
import torch


def app_api():
    return importlib.import_module('flybrain.escape_ui.viewer').DemoViewer


def test_compact_layout_has_ten_rows_and_nonoverlapping_viewport_routing():
    assert importlib.util.find_spec('flybrain.escape_ui.layout'), 'pixel dashboard layout required'
    from flybrain.escape_ui.layout import DashboardLayout
    for width, height in ((1000, 700), (1440, 900)):
        layout = DashboardLayout(width, height)
        assert len(layout.group_rows) == 10
        assert all(layout.sidebar.contains(x, y) for x, y in layout.group_rows)
        assert all(y+15 < layout.sidebar.bottom for _, y in layout.group_rows)
        for name, (left, right, bottom, top) in layout.rectangles.items():
            if name not in ('world', 'brain'):
                continue
            assert -1 < left < right < 1 and -1 < bottom < top < 1
        assert layout.route(*layout.normalized(layout.sidebar.left+30, layout.sidebar.top+90)) == 'details'


def test_owner_visibility_filters_alpha_without_changing_spike_history():
    assert importlib.util.find_spec('flybrain.escape_ui.groups'), 'group presentation required'
    from flybrain.escape_ui.groups import GroupActivityTable
    from flybrain.flight3d.activity import SpikeRecorder
    from test_flight3d_activity import snapshot
    recorder = SpikeRecorder(snapshot(counts=(0, 0)))
    frame = recorder.observe(snapshot(.02, (1, 1)))
    table = GroupActivityTable(frame.snapshot.neuron_ids)
    table.update(frame, False, (True, False))
    assert table.colors[0, 3] > 0 and table.colors[1, 3] == 0
    assert frame.cumulative_counts == (1, 1)
    uploads = table.uploads
    table.update(frame, False, (True, False))
    assert table.uploads == uploads
    table.update(frame, False, (False, True))
    assert table.colors[0, 3] == 0 and table.colors[1, 3] > 0


@pytest.mark.parametrize('size', [(1000, 700), (1440, 900)])
def test_dashboard_real_checkboxes_tabs_and_passive_capture(size, tmp_path):
    from direct.gui.DirectGui import DirectCheckButton
    from flybrain.escape_edge.delivery import verified_candidate
    torch.set_num_threads(1)
    app = app_api()(1110000, verified_candidate(), verified=True,
                    offscreen=True, window_size=size)
    try:
        app.paused = True
        app.simulation.pause()
        app.draw_frame(app.simulation.display_frame)
        assert hasattr(app, 'group_checks'), 'persistent checkbox sidebar required'
        assert len(app.group_checks) == 10 and app.sidebar_tab == 'groups'
        assert app.whole and not app.vnc
        assert app.pause_button['text'] == 'Resume'
        assert all(isinstance(c, DirectCheckButton) and c['indicatorValue'] for c in app.group_checks.values())
        for button in app.action_buttons.values():
            bounds = button['frameSize']
            assert bounds[3]-bounds[2] >= 44
        for checkbox in app.group_checks.values():
            assert min(checkbox['text_scale']) >= 16
            p = checkbox.getPos(app.pixel2d)
            assert app.layout.sidebar.contains(p.x, -p.z)
        assert -float(app.group_description.getTightBounds()[0].z) <= app.layout.sidebar.bottom-4
        before = app.session.snapshot
        voltage = app.session.brain.voltage.clone()
        reference = app.session.pair.reference.voltage.clone()
        frame = app.simulation.display_frame
        camera = np.array(app.brain_camera.getMat())
        app.group_checks['vision'].commandFunc(None)
        assert 'vision' not in app.enabled_classes
        assert not app.group_checks['vision']['indicatorValue']
        app.set_group_visible('vision', False)
        assert 'vision' not in app.enabled_classes
        app.show_tab('inspect')
        app.show_tab('help')
        assert 'Spawn a flower' in app.help.getText()
        app._ui()
        app._resize()
        app.draw_frame(frame)
        assert app.sidebar_tab == 'help' and not app.group_checks['vision']['indicatorValue']
        app.capture_paused(tmp_path / 'dashboard.png')
        assert (tmp_path / 'dashboard.png').stat().st_size > 10000
        assert app.session.snapshot == before and app.simulation.display_frame is frame
        assert torch.equal(voltage, app.session.brain.voltage)
        assert torch.equal(reference, app.session.pair.reference.voltage)
        np.testing.assert_array_equal(camera, np.array(app.brain_camera.getMat()))
        assert app.session.brain.steps == app.session.pair.reference.steps == 0
        app.cord_check.commandFunc(None)
        assert app.vnc and app.cord_check['indicatorValue']
        app.outline_check.commandFunc(None)
        assert not app.whole and not app.outline_check['indicatorValue']
        app.show_tab('groups')
        p = app.group_checks['odor'].getPos(app.pixel2d)
        nx, ny = app.layout.normalized(p.x, -p.z)
        app.begin_gesture(nx, ny)
        assert not app.drag
        app.zoom_at(.8, nx, ny)
        np.testing.assert_array_equal(camera, np.array(app.brain_camera.getMat()))
        app.messenger.send('space')
        assert not app.paused and app.pause_button['text'] == 'Pause'
        app.pause_button['command']()
        assert app.paused
        assert verified_candidate() is not None
    finally:
        app.destroy()


def test_hidden_owner_leaves_no_pixels_or_depth_occlusion():
    """Use a real GPU buffer; hidden near branches must not mask visible ones."""
    from panda3d.core import Camera, NodePath, OrthographicLens, PNMImage
    from flybrain.escape_ui.groups import GroupActivityTable
    from flybrain.escape_edge.delivery import verified_candidate
    from flybrain.flight3d.activity import SpikeRecorder
    from test_flight3d_activity import snapshot
    torch.set_num_threads(1)
    app = app_api()(1110000, verified_candidate(), verified=True,
                    offscreen=True, window_size=(1000, 700))
    try:
        app.paused = True
        app.simulation.pause()
        recorder = SpikeRecorder(snapshot(counts=(0, 0)))
        frame = recorder.observe(snapshot(.02, (1, 1)))
        table = GroupActivityTable(frame.snapshot.neuron_ids, np.array(((0, 1, 0), (1, 0, 0))))
        table.update(frame, False, (False, True))
        # Green hidden line is nearer the camera and appears first in the geometry.
        lines = np.array([[[0, -.1, -.7], [0, -.1, .7]], [[0, .1, -.7], [0, .1, .7]]])
        root = NodePath('filtered-gpu-test')
        node = table.make_batch(lines, np.arange(2), 'overlapping-owners')
        node.setRenderModeThickness(8)
        node.reparentTo(root)
        lens = OrthographicLens()
        lens.setFilmSize(2, 2)
        camera = root.attachNewNode(Camera('filtered-gpu-camera', lens))
        camera.setPos(0, -2, 0)
        buffer = app.win.makeTextureBuffer('filtered-gpu', 80, 80)
        buffer.setClearColor((0, 0, 0, 0))
        buffer.setClearColorActive(True)
        region = buffer.makeDisplayRegion()
        region.setCamera(camera)
        for _ in range(3):
            app.graphicsEngine.renderFrame()
            app.graphicsEngine.syncFrame()
        image = PNMImage()
        buffer.getScreenshot(image)
        red, green, blue = image.getXel(40, 40)
        assert red > .35 and green < .02 and blue < .02
        table.update(frame, False, (False, False))
        for _ in range(3):
            app.graphicsEngine.renderFrame()
            app.graphicsEngine.syncFrame()
        buffer.getScreenshot(image)
        assert max(image.getXel(40, 40)) < .02
        assert frame.cumulative_counts == (1, 1)
        app.graphicsEngine.removeWindow(buffer)
        root.removeNode()
    finally:
        app.destroy()


def test_filtered_selection_bounds_empty_state_and_replay(tmp_path):
    from flybrain.escape_edge.delivery import verified_candidate
    from flybrain.retreat3d.world import PredatorInput
    torch.set_num_threads(1)
    app = app_api()(1110000, verified_candidate(), verified=True,
                    offscreen=True, window_size=(1000, 700))
    try:
        app.paused = True
        app.simulation.pause()
        assert hasattr(app, 'show_only_group'), 'functional group isolation required'
        app.load_geometry(100000)
        app.show_only_group('vision')
        assert app.enabled_classes == frozenset({'vision'})
        assert all(bool(c['indicatorValue']) == (k == 'vision') for k, c in app.group_checks.items())
        odor_id = next(n['id'] for n in app.manifest['nodes'] if 'odor' in n['roles'])
        app.select(odor_id)
        assert 'hidden by filter' in app.info.getText().lower()
        assert all(node.isHidden() for node in app.selection_nodes)
        app.toggle_technical()
        assert odor_id in app.info.getText()
        points = next(p for key, p in app._pick_parts if key == odor_id)
        local = np.c_[points, np.ones(len(points))] @ np.array(app.brain_scene.getMat(app.brain_camera))
        valid = np.flatnonzero((local[:, 1] > 0) & (points[:, 1] >= -.43))
        assert len(valid)
        index = valid[0]
        fov = np.radians(app.brain_camera.node().getLens().getFov())
        ndc = local[index, [0, 2]] / local[index, 1] / np.tan(fov/2)
        left, right, bottom, top = app.layout.rectangles['brain']
        x = left + (ndc[0]+1)/2*(right-left)
        y = bottom + (ndc[1]+1)/2*(top-bottom)
        saved = app._pick_parts
        app._pick_parts = [(odor_id, points)]
        app._pick_points = None
        app.clear_selection()
        app.pick(x, y)
        assert app.selected is None
        app._pick_parts = saved
        app._pick_points = None
        app.set_outline(False)
        bounds = app.visible_bounds()
        assert bounds is not None and bounds.shape == (2, 3)
        expected = np.vstack([value for (owner, cord), value in app.group_bounds.items()
                              if app.activity_mask[owner] and not cord])
        np.testing.assert_allclose(bounds, [expected.min(axis=0), expected.max(axis=0)])
        vision_id = app.activity_index.ids[min(app.activity_index.members['vision'])]
        app.select(vision_id)
        # The full selected morphology may extend beyond its coarse branch batch.
        extended = np.array([[[bounds[1, 0]+.1, 0, bounds[1, 2]+.1],
                              [bounds[1, 0]+.2, 0, bounds[1, 2]+.2]]])
        app._prepare_selection(extended)
        np.testing.assert_allclose(app.visible_bounds()[1, [0, 2]],
                                   [bounds[1, 0]+.2, bounds[1, 2]+.2])
        app.hide_all_groups()
        assert app.visible_bounds() is None and not app.empty_label.isHidden()
        camera = np.array(app.brain_camera.getMat())
        app.fit_view()
        np.testing.assert_array_equal(camera, np.array(app.brain_camera.getMat()))
        app.show_all_groups()
        assert all(c['indicatorValue'] for c in app.group_checks.values())
        app.show_only_group('vision')
        app.keys.add('w')
        app.spawn_flower_button['command']()
        assert app.session.snapshot.object_kind == 'flower' and not app.keys
        app.spawn_predator_button['command']()
        assert app.session.snapshot.object_kind == 'predator'
        app.simulation.advance_fixed(3, PredatorInput.from_keys({'a'}))
        endpoint = app.session.snapshot
        app.replay_button['command']()
        assert app.session.snapshot == endpoint and app.paused
        assert app.enabled_classes == frozenset({'vision'}) and not app.whole
        app.new_trial_button['command']()
        assert app.session.snapshot.object_kind == 'none' and app.session.snapshot.time == 0
        assert app.enabled_classes == frozenset({'vision'})
    finally:
        app.destroy()
