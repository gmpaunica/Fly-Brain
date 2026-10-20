"""Anatomical overview, passive activity, cached inspection and responsive UI."""
import torch
import numpy as np
import pytest


def test_real_anatomy_batched_overview_selection_and_passive_render():
    from flybrain.flight3d.anatomical_viewer import FlightViewer
    from panda3d.core import ClockObject,PNMImage
    torch.set_num_threads(1)
    app=FlightViewer(970000,'teacher',offscreen=True,window_size=(1000,700))
    clock=ClockObject.getGlobalClock();clock.setMode(ClockObject.MNonRealTime);clock.setDt(.02)
    try:
        assert not app.vnc and not hasattr(app,'bundle_nodes')
        app.paused=True;app.load_geometry(100000)
        assert len(app.loaded)==585
        assert len(app.source_batches)<150
        assert app.surface_nodes[0].getParent()==app.context[0]
        app.simulation.advance_fixed(20)
        snapshot=app.session.snapshot;steps=app.session.brain.steps
        app.draw_snapshot(snapshot)
        assert len(app.visible_activity_ids)>12
        updates=app.activity_updates
        for name in ('front','side','top'):
            app.preset(name)
            for _ in range(4):app.draw_snapshot(snapshot);app.graphicsEngine.renderFrame()
            picture=PNMImage();app.win.getScreenshot(picture)
            assert sum(picture.getRed(x,y) for x in range(20,400,3) for y in range(18,42))>50
        assert app.activity_updates==updates
        assert app.session.snapshot==snapshot and app.session.brain.steps==steps
        from flybrain.flight3d.session import FlightSession
        from flybrain.flight3d.world import make_trial
        serial=FlightSession(make_trial(970000),'teacher')
        for _ in range(20):serial.advance()
        assert serial.snapshot==snapshot
        for step in range(10):
            app.toggle_view();app.toggle_mode();app.preset(('front','side','top')[step%3])
            serial.advance();app.simulation.advance_fixed(1)
            for _ in range(step%4):app.draw_snapshot(app.session.snapshot)
        assert app.session.snapshot==serial.snapshot
        assert torch.equal(app.session.brain.voltage,serial.brain.voltage)
        # Click a projected real branch; this exercises camera transform and routing.
        app.focus_brain();app.preset('front')
        points=np.concatenate([p for _,p in app._pick_parts])
        local=np.c_[points,np.ones(len(points))]@np.array(app.brain_scene.getMat(app.brain_camera))
        fov=app.brain_camera.node().getLens().getFov()
        projected=local[:,[0,2]]/np.maximum(local[:,1:2],1e-9)/np.tan(np.radians(fov)/2)
        visible=(local[:,1]>0)&(points[:,1]>=-.43)&(np.abs(projected).max(axis=1)<.8)
        xy=projected[np.flatnonzero(visible)[0]]
        left,right,bottom,top=app.layout.rectangles['brain']
        app.pick(left+(xy[0]+1)*(right-left)/2,bottom+(xy[1]+1)*(top-bottom)/2)
        assert app.selected in app.loaded
        id=next(id for id in app.loaded if app.nodes[id]['metadata']['region']!='ventral_nerve_cord')
        app.select(id);assert id in app.info.getText()
        full=app.morphologies.full(id)
        app.select(id);assert app.morphologies.full(id) is full
        app.toggle_edges();assert app.show_edges
        app.toggle_edges();assert not app.edge_node.getNumChildren()
        app.list_groups();app.zoom_at(1.14,.8,-.7)
        assert app.panels['details'].verticalScroll['value']>0
        for key in ('n','tab','v','b','f'):app.messenger.send(key)
        app.replay();app.draw_snapshot(app.session.snapshot)
        assert not app.visible_activity_ids
        app.new_trial();assert app.session.world.case.seed==970001
        app._ui();app._resize();assert app.ui_size==(1000,700)
        assert app.isAccepting('shift-mouse1')
        with pytest.raises(SystemExit):app.messenger.send('escape')
        assert not app.simulation._thread.is_alive()
    finally:app.destroy();clock.setMode(ClockObject.MNormal)
