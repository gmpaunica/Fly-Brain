"""Real viewer switches alter display colors only; no live-review claim."""
import numpy as np
import torch
from test_flight3d_controls import camera_state


def test_switches_focus_pause_reset_and_geometry_keep_scientific_state():
    from flybrain.flight3d.anatomical_viewer import FlightViewer
    torch.set_num_threads(1)
    app = FlightViewer(970000, 'teacher', offscreen=True, window_size=(1000,700))
    try:
        app.paused = True; app.simulation.advance_fixed(100); app.load_geometry(100000)
        frame = app.simulation.display_frame
        app.draw_frame(frame)
        assert app.enabled_classes == frozenset({'odor'}) and app.activity_scope == 'whole'
        assert 0 < len(app.visible_activity_ids) <= 69
        assert sum(c>0 for c in frame.cumulative_counts) == 530
        voltage = app.session.brain.voltage.clone(); camera = camera_state(app)
        batches = tuple(app.source_batches)
        app.toggle_activity_class('motor'); assert app.enabled_classes == frozenset({'odor','motor'})
        app.toggle_activity_class('odor'); assert app.enabled_classes == frozenset({'motor'})
        app.toggle_sensory(); assert app.activity_index.sensory_state(app.enabled_classes) == 'on'
        app.toggle_sensory(); assert app.enabled_classes == frozenset({'motor'})
        app.all_activity(); assert app.activity_scope == 'whole' and len(app.visible_activity_ids)>200
        all_visible = app.visible_activity_ids
        no_outgoing = next(id for id in app.session.brain.ids if not any(e['pre']==id for e in app.manifest['edges']))
        app.select(no_outgoing); app.set_activity_scope('selected')
        assert app.activity_mask[app._owner_index[no_outgoing]]
        partners = {no_outgoing}
        for e in app.manifest['edges']:
            if e['pre']==no_outgoing: partners.add(e['post'])
            if e['post']==no_outgoing: partners.add(e['pre'])
        assert {id for id,on in zip(app.session.brain.ids,app.activity_mask) if on} == partners
        app.activity_off(); assert not app.visible_activity_ids
        assert app.activity_scope == 'selected'
        np.testing.assert_allclose(app.activity_table.colors, np.tile((.24,.30,.55,.22),(589,1)))
        app.toggle_activity_class('odor')
        retained = app.enabled_classes
        assert {id for id,on in zip(app.session.brain.ids,app.activity_mask) if on} == {
            n['id'] for n in app.manifest['nodes'] if n['id'] in partners and 'odor' in n['roles']}
        assert app.simulation.display_frame is frame
        assert app.session.snapshot is frame.snapshot
        assert torch.equal(app.session.brain.voltage, voltage)
        app.replay(); assert app.enabled_classes == retained and app.activity_scope == 'selected'
        assert not any(app.simulation.display_frame.cumulative_counts)
        app.new_trial(); assert app.enabled_classes == retained and app.activity_scope == 'selected'
        app.clear_selection(); assert app.activity_scope == 'whole' and app.enabled_classes == retained
        cord_only=next(app.session.brain.ids[i] for i in app._known_branch_owners[1]-app._known_branch_owners[0])
        app.select(cord_only)
        assert 'Only nerve-cord branches' in app.info_text and not app.vnc
        assert any(b['text'].startswith('Show nerve cord') for b in app.member_buttons)
        app.clear_selection()
        app.set_activity_scope('selected'); assert app.activity_scope == 'whole'
        app.select(no_outgoing); app.set_activity_scope('selected')
        background = next(id for id in app.records if id not in app.nodes)
        app.select(background); assert app.activity_scope == 'whole' and app.enabled_classes == retained
        np.testing.assert_array_equal(camera_state(app),camera)
        assert tuple(app.source_batches)==batches
        app.all_activity(); app.toggle_vnc(); app.load_geometry(100000)
        app.draw_frame(frame)
        assert len(app.visible_activity_ids)>400
        app.select(sorted(app.source_missing)[0]); assert 'unavailable' in app.info_text
        assert app.simulation.display_frame.snapshot.time==0
    finally: app.destroy()


def test_settings_switch_commands_disabled_scope_and_two_column_legend():
    from flybrain.flight3d.anatomical_viewer import FlightViewer
    torch.set_num_threads(1)
    for size in ((1440,900),(1000,700)):
        app = FlightViewer(970000, 'teacher', offscreen=True, window_size=size)
        try:
            app.paused=True; app.toggle_settings()
            selected_button = next(b for b in app.member_buttons if b['text'].startswith('Selected circuit'))
            assert selected_button['state']=='disabled'
            odor_button = next(b for b in app.member_buttons if b['text'].startswith('Odor inputs'))
            assert '69' in odor_button['text'] and 'on' in odor_button['text']
            # Class switches must be discoverable without scrolling past shortcuts.
            from panda3d.core import Point3
            for _ in range(2):app.taskMgr.step();app.graphicsEngine.syncFrame()
            for c in app.activity_index.classes:
                button=next(b for b in app.member_buttons if b['text'].startswith(c.label+' ('))
                centre=app.render2d.getRelativePoint(button,Point3(0,0,0))
                assert -.82 <= centre.z <= -.48, (c.label,centre.z)
            odor_button['command']()
            assert not app.enabled_classes
            assert 'off' in app.legend_entries['odor'][1].getText()
            assert len(app.legend_entries)==10
            assert len({entry[1].getPos()[0] for entry in app.legend_entries.values()})==2
            assert 'cyan' in app.help.getText().lower() and 'idle' in app.help.getText().lower()
            expand = next(b for b in app.member_buttons if b['text'].startswith('Sensory children'))
            expand['command']()
            assert not any(b['text'].startswith('Odor inputs') for b in app.member_buttons)
            app.all_activity()
            assert all('on' in pair[1].getText() for pair in app.legend_entries.values())
            # A frameColor option alone is insufficient if the UI frame is not drawn.
            from panda3d.core import PNMImage, Point3
            for _ in range(3):app.graphicsEngine.renderFrame();app.graphicsEngine.syncFrame()
            picture=PNMImage();assert app.win.getScreenshot(picture)
            swatch=app.legend_entries['odor'][0]
            centre=app.render2d.getRelativePoint(swatch,Point3(.008,0,0))
            x=round((centre.x+1)*size[0]/2);y=round((1-centre.z)*size[1]/2)
            np.testing.assert_allclose(tuple(picture.getXel(x,y)),np.array((230,180,80))/255,atol=.01)
        finally: app.destroy()
