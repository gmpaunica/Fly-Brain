"""Actual renderer/event contracts; separate from a human live review."""
import numpy as np
import torch


def camera_state(app):
    camera = app.brain_motion
    return np.r_[camera.orientation, camera.orientation_target, camera.pivot,
                 camera.pivot_target, camera.offset, camera.offset_target,
                 camera.distance, camera.target, np.array(app.brain_camera.getMat()).ravel()]


def test_camera_preservation_inspector_gestures_and_routing():
    from flybrain.flight3d.anatomical_viewer import FlightViewer
    from panda3d.core import ClockObject
    torch.set_num_threads(1)
    app = FlightViewer(970000, 'teacher', offscreen=True, window_size=(1000,700))
    clock = ClockObject.getGlobalClock(); clock.setMode(ClockObject.MNonRealTime); clock.setDt(.02)
    try:
        app.paused = True; app.load_geometry(100000)
        assert not app.isAccepting('n') and not app.isAccepting('c')
        assert app.whole and not app.show_edges and not app.vnc
        assert app.views_button is not None
        snapshot = app.session.snapshot; voltage = app.session.brain.voltage.clone()
        app.brain_motion.rotate_pixels(100,300)
        app.brain_motion.pan_pixels(40,20,357,45)
        app.brain_motion.zoom(.88); app.brain_motion.advance(.08); app._brain_camera()
        frozen = camera_state(app)
        for _ in range(4):
            app.toggle_vnc()
            np.testing.assert_array_equal(camera_state(app), frozen)
            assert app.brain_motion.minimum == .03
        app.brain_motion.zoom(.00001); app.brain_motion.advance(5)
        app._brain_camera()
        assert abs(app.distance-.03) < 1e-12
        assert app.brain_camera.node().getLens().getNear() < .001
        # Gestures use window pixels; a pan release cannot select or jump zoom.
        app.begin_gesture(.5,.3,pan=True)
        zoom = app.brain_motion.target
        app.move_gesture(.6,.4); app.brain_motion.advance(.06); app.stop_drag()
        assert app.drag is None and app.selected is None
        assert app.brain_motion.target == zoom
        orientation = app.brain_motion.orientation.copy()
        app.brain_motion.advance(.5)
        np.testing.assert_array_equal(app.brain_motion.orientation, orientation)
        app.begin_gesture(.5,.3)
        app.messenger.send('shift-mouse1-up'); assert app.drag is None
        app.keys.add('shift'); app.begin_gesture(.5,.3)
        app.focus_changed(False); assert not app.keys and app.drag is None
        # UI rebuild leaves camera and selection intact, including full cache state.
        id = next(n['id'] for n in app.manifest['nodes']
                  if n['id'] in app.loaded and n['metadata']['region']=='optic_lobe')
        app.select(id)
        frozen = camera_state(app)
        app.toggle_edges(); assert app.show_edges
        labels = [button['text'] for button in app.member_buttons]
        assert 'Focus selected' in labels and 'Clear selection' in labels
        assert 'retained connections' in app.info_text
        app._ui(); app._resize()
        np.testing.assert_array_equal(camera_state(app), frozen)
        assert app.selected == id
        app.toggle_settings(); assert 'Background anatomy: on' in [b['text'] for b in app.member_buttons]
        app.toggle_mode();assert 'Background anatomy: off' in [b['text'] for b in app.member_buttons]
        app.toggle_mode()
        app.toggle_settings(); assert app.selected == id
        # Explicit focus changes targets, ordinary selection does not.
        app.focus_selected(); assert not np.array_equal(app.brain_motion.pivot_target,app.brain_motion.pivot)
        app.clear_selection(); assert app.selected is None and not app.show_edges
        assert app.edge_node.getNumChildren() == 0
        app.select(sorted(app.source_missing)[0])
        assert 'unavailable' in app.info_text
        app.clear_selection()
        brain, world = app.brain_motion.target, app.world_motion.target
        for xy in ((.6,-.6),(-.6,-.6),(0,.3),(.6,.95)):
            app.zoom_at(.88,*xy)
        assert (app.brain_motion.target,app.world_motion.target) == (brain,world)
        app.zoom_at(.88,-.5,.3)
        assert app.world_motion.target < world and app.brain_motion.target == brain
        app.zoom_at(1.14,.5,.3); assert app.brain_motion.target > brain
        app.fit_view(); assert app.brain_motion.target > .03
        app.toggle_vnc(); app.reset_view(); assert app.vnc
        assert app.session.snapshot == snapshot
        assert torch.equal(app.session.brain.voltage,voltage)
        # A gesture begun at another window size must be discarded on resize.
        app.begin_gesture(.5,.3)
        app._gesture_size=(1440,900)
        app.move_gesture(.6,.4); assert app.drag is None
        app.messenger.send('space'); assert not app.paused
        app.messenger.send('space'); assert app.paused
        app.replay(); assert app.session.snapshot.time == 0
        app.new_trial(); assert app.session.world.case.seed == 970001
    finally:
        app.destroy(); clock.setMode(ClockObject.MNormal)


def test_actual_viewport_sizes_keep_pixel_sensitivity_and_responsive_toolbar():
    from flybrain.flight3d.anatomical_viewer import FlightViewer
    from flybrain.flight3d.brain_camera import BrainCamera
    torch.set_num_threads(1)
    targets=[]
    for width,height in ((1000,700),(1440,900)):
        app=FlightViewer(970000,'teacher',offscreen=True,window_size=(width,height))
        try:
            app.paused=True;app.begin_gesture(.5,.3)
            app.move_gesture(.5+80/width,.3-60/height)
            targets.append(app.brain_motion.orientation_target.copy())
            assert (app.views_button is not None)==(width<1200)
            if app.views_button is not None:
                app.toggle_views();assert all(not b.isHidden() for b in app.views_menu)
                app.stop_drag();target=app.brain_motion.target
                app.zoom_at(.88,.28,.60);assert app.brain_motion.target==target
                app.begin_gesture(.28,.60);assert app.drag is None
                app.preset('side');assert all(b.isHidden() for b in app.views_menu)
            expected=BrainCamera();expected.rotate_pixels(40,-30)
            np.testing.assert_allclose(targets[-1],expected.orientation_target,atol=1e-15)
        finally:app.destroy()
    np.testing.assert_allclose(targets[0],targets[1],atol=1e-15)
