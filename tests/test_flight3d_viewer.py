"""Offscreen event/render integration. This is not a live desktop input review."""
import pytest
pytest.importorskip('panda3d')
from flybrain.flight3d.anatomy import CACHE


@pytest.mark.skipif(not (CACHE/'circuit.json').exists(),reason='prepared BANC cache required')
def test_passive_render_cameras_modes_and_bound_replay_events():
    from flybrain.flight3d.viewer import FlightViewer
    from panda3d.core import ClockObject, PNMImage
    import torch
    torch.set_num_threads(1)
    app=FlightViewer(10001,'teacher',offscreen=True)
    try:
        clock=ClockObject.getGlobalClock(); clock.setMode(ClockObject.MNonRealTime); clock.setDt(.02)
        app.taskMgr.step()
        assert app.session.world.time>0
        assert any(abs(wing.getR())>1e-3 for wing in app.wings)
        app.paused=True
        app.load_geometry(100000)
        snapshot=app.session.snapshot; steps=app.session.brain.steps
        positions=[node.getPos(app.brain_scene) for node in app.activity_nodes if node is not None]
        for preset in ('front','side','top'):
            app.preset(preset); app.zoom(.9); app.draw_snapshot(snapshot); app.graphicsEngine.renderFrame()
            image=PNMImage(); app.win.getScreenshot(image)
            # Heavy anatomy must not make the separate title/UI disappear.
            assert sum(image.getRed(x,y) for x in range(25,350,3) for y in range(20,45))>100
        app.toggle_mode(); app.toggle_vnc()
        assert snapshot==app.session.snapshot and app.session.brain.steps==steps
        assert positions==[node.getPos(app.brain_scene) for node in app.activity_nodes if node is not None]
        app.select(app.session.brain.ids[0]); assert app.session.brain.ids[0] in app.info.getText()
        # Switching after inspecting a context cell must hide its detail too.
        app.selected=next(id for id in app.records if id not in app.nodes)
        app.toggle_mode(); app.toggle_mode()
        assert not app.whole and all(layer.isHidden() for layer in app.selection_nodes)
        app.messenger.send('backspace'); assert app.session.world.time==0
        app.messenger.send('r'); assert app.session.world.case.seed==10002
        app.focus_brain(); app.toggle_follow(); app.draw_snapshot(app.session.snapshot)
    finally:
        app.destroy(); clock.setMode(ClockObject.MNormal)
