"""Offscreen grouped UI events. These are not live mouse/keyboard observations."""
import numpy as np
import torch


def test_group_selection_navigation_pause_and_render_independence():
    from flybrain.flight3d.network_viewer import FlightViewer
    from panda3d.core import ClockObject
    torch.set_num_threads(1)
    app=FlightViewer(10001,'teacher',offscreen=True,window_size=(1000,700))
    clock=ClockObject.getGlobalClock();clock.setMode(ClockObject.MNonRealTime);clock.setDt(.02)
    try:
        assert app.network_view and not app.vnc
        assert app.isAccepting('shift-mouse1'), 'Panda emits a modified mouse event for Shift-drag'
        for _ in range(5):app.taskMgr.step()
        app.messenger.send('space');assert app.paused
        snapshot=app.session.snapshot
        assert any(snapshot.edge_transmissions)
        for _ in range(5):app.draw_snapshot(snapshot);app.taskMgr.step()
        assert app.session.snapshot==snapshot
        brain_target=app.brain_motion.target;world_target=app.world_motion.target
        app.list_groups()
        app.zoom_at(1.14,.9,-.7)
        assert app.panels['details'].verticalScroll['value']>0
        assert app.brain_motion.target==brain_target and app.world_motion.target==world_target
        app.zoom_at(.88,-.5,.3);assert app.world_motion.target<world_target
        app.zoom_at(1.14,.5,.3);assert app.brain_motion.target>brain_target
        app.select_group(0)
        assert app.grouped.groups[0].members[0] in [b['text'] for b in app.member_buttons]
        member=app.grouped.groups[0].members[0];app.select(member)
        assert member in app.info.getText()
        for key in ('n','tab','v','1','2','3','b','f'):app.messenger.send(key)
        assert not app.network_view and not app.vnc
        app.messenger.send('backspace');assert app.session.world.time==0
        assert not any(app.session.snapshot.edge_activity)
        app.messenger.send('r');assert app.session.world.case.seed==10002
        app.messenger.send('space');app.taskMgr.step();assert app.session.world.time>0
        # Rebuild responsive UI and verify every viewport retains its rectangle.
        app.select_group(0)
        app._ui();app._resize();app.taskMgr.step()
        assert app.ui_size==(1000,700)
        assert app.member_buttons and 'Group 00' in app.info.getText()
    finally:
        app.destroy();clock.setMode(ClockObject.MNormal)
