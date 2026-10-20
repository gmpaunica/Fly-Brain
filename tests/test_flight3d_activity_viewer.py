"""Activity survives skipped refreshes, geometry loading and inspector actions."""
import numpy as np
import torch


def test_mixed_branches_deferred_ownership_and_inspector_are_passive():
    from flybrain.flight3d.anatomical_viewer import FlightViewer
    from flybrain.flight3d.activity import pulse_strengths
    from panda3d.core import GeomVertexReader
    torch.set_num_threads(1)
    app=FlightViewer(970000,'teacher',offscreen=True,window_size=(1440,900))
    try:
        app.paused=True;app.simulation.advance_fixed(100)
        app.all_activity()  # This regression explicitly checks the all-class shortcut.
        frame=app.simulation.display_frame
        app.draw_frame(frame)
        assert not app.visible_activity_ids  # no uploaded geometry yet
        app.load_geometry(100000);app.draw_frame(frame)
        assert len(app.visible_activity_ids)>200
        assert sum(count>0 for count in frame.cumulative_counts)==530
        assert sum(count==0 for count in frame.cumulative_counts)==59
        eligible=[i for i,id in enumerate(frame.snapshot.neuron_ids)
                  if i in app._resident_owners[0] and app.nodes[id]['metadata']['region']=='ventral_nerve_cord'
                  and frame.cumulative_counts[i]>0]
        assert len(eligible)==56
        assert all(frame.snapshot.neuron_ids[i] in app.visible_activity_ids
                   for i in eligible if pulse_strengths(frame)[i]>0)
        # All uploaded/deferred segment endpoints keep ownership after split.
        def verify_batches():
            for node in app.source_batches:
                data=node.node().getGeom(0).getVertexData()
                vertices=np.frombuffer(data.getArray(0).getHandle().getData(),dtype=np.float32).reshape(-1,4)
                indices=vertices[:,3]
                assert np.array_equal(indices[::2],indices[1::2])
                assert indices.min()>=0 and indices.max()<589
        verify_batches()
        deferred=[(lines.copy(),owners.copy()) for lines,owners in app._deferred_cord]
        assert deferred
        original=frame;colors=app.activity_table.colors.copy();uploads=app.activity_table.uploads
        app.toggle_vnc();app.load_geometry(100000);app.draw_frame(frame)
        assert not app._deferred_cord
        assert len(app.visible_activity_ids)>400
        verify_batches()
        cord_nodes=[node for node in app.source_batches if node.getParent()==app.project[1]]
        assert len(cord_nodes)==len(deferred)
        for node,(lines,owners) in zip(cord_nodes,deferred):
            data=node.node().getGeom(0).getVertexData()
            vertices=np.frombuffer(data.getArray(0).getHandle().getData(),dtype=np.float32).reshape(-1,4)
            np.testing.assert_array_equal(vertices[:,:3],lines.astype(np.float32).reshape(-1,3))
            np.testing.assert_array_equal(vertices[:,3],np.repeat(owners,2))
        assert app.activity_table.uploads==uploads
        np.testing.assert_array_equal(app.activity_table.colors,colors)
        no_outgoing=next(id for id,count in zip(frame.snapshot.neuron_ids,frame.cumulative_counts)
                         if count and id in app.visible_activity_ids and not any(e['pre']==id for e in app.manifest['edges']))
        app.select(no_outgoing)
        assert 'Latest interval spikes:' in app.info_text
        assert 'Spikes since reset:' in app.info_text
        assert 'Recent firing rate:' in app.info_text
        assert 'Time since last activity:' in app.info_text
        # Unchanged publications must not re-scan topology or reformat the
        # inspector each display refresh (selection performance regression).
        selection_info=app._selection_info;formatted=[]
        def counted_info(frame):
            formatted.append(frame.step_index)
            return selection_info(frame)
        app._selection_info=counted_info
        for _ in range(3):app.draw_frame(frame)
        assert formatted==[]
        buttons=tuple(app.member_buttons)
        app.panels['details'].verticalScroll['value']=.5
        app.simulation.advance_fixed(1);app.draw_frame(app.simulation.display_frame)
        assert tuple(app.member_buttons)==buttons
        assert app.panels['details'].verticalScroll['value']==.5
        # Follow camera, fly body and activity all use the same acquired frame,
        # even when the worker has already published a newer one.
        app.follow=True
        app.draw_frame(frame)
        target=np.array(frame.snapshot.position)
        offset=app.world_distance*np.array((np.sin(app.world_azimuth)*np.cos(app.world_elevation),
                -np.cos(app.world_azimuth)*np.cos(app.world_elevation),np.sin(app.world_elevation)))
        np.testing.assert_allclose(tuple(app.camera.getPos()),target+offset+app.world_motion.offset,atol=1e-7)
        silent=next(id for id,count in zip(frame.snapshot.neuron_ids,frame.cumulative_counts)
                    if count==0 and not any(e['post']==id and e['pre']!=id for e in app.manifest['edges']))
        app.select(silent)
        assert 'No non-self incoming connections' in app.info_text
        assert 'not since reset' in app.info_text
        app.select(sorted(app.source_missing)[0])
        assert 'unavailable' in app.info_text and 'Spikes since reset:' in app.info_text
        app.clear_selection()
        frozen=app.simulation.display_frame
        for _ in range(3):
            app.toggle_view();app.toggle_mode();app.toggle_vnc();app.load_geometry(100000)
            app.draw_frame(frozen);app.draw_frame(frozen)
        assert app.simulation.display_frame is frozen
        assert original.cumulative_counts==frame.cumulative_counts
        app.replay()
        assert app.simulation.display_frame.episode_id!=frozen.episode_id
        assert not any(app.simulation.display_frame.cumulative_counts)
        assert not app.visible_activity_ids
        app.simulation.advance_fixed(1);app.new_trial()
        assert app.simulation.display_frame.snapshot.seed==970001
        assert not any(app.simulation.display_frame.cumulative_counts)
    finally:app.destroy()


def test_activity_at_matched_simulation_times_is_display_rate_independent():
    from flybrain.flight3d.activity import SpikeRecorder,pulse_strengths
    from flybrain.flight3d.branch_activity import BranchActivityTable
    from flybrain.flight3d.session import FlightSession
    from flybrain.flight3d.world import make_trial
    from flybrain.flight3d.activity_classes import DisplayActivityIndex
    import json
    from pathlib import Path
    torch.set_num_threads(1)
    session=FlightSession(make_trial(970000),'teacher')
    index=DisplayActivityIndex(json.loads(Path('data/flight3d/circuit.json').read_text()))
    odor=next(n['id'] for n in session.manifest['nodes'] if 'odor' in n['roles'])
    mask=index.mask(frozenset({'odor','motor'}),'selected',odor)
    snapshots=[session.advance() for _ in range(100)]
    results=[]
    for hz in (30,60,120):
        session.replay();recorder=SpikeRecorder(session.snapshot)
        table=BranchActivityTable(session.brain.ids,np.asarray(index.activation_colors))
        interval=0;matched=[]
        for draw in range(1,2*hz+1):
            time=draw/hz
            while interval<len(snapshots) and snapshots[interval].time<=time+1e-10:
                recorder.observe(snapshots[interval]);interval+=1
            frame=recorder.frame;table.update(frame,False,mask)
            # Drawing repeated at the same frame cannot change the publication.
            table.update(frame,False,mask)
            if draw in (hz,2*hz):
                assert any(mask) and np.any(np.abs(table.colors[:,:3]-(.24,.30,.55))>1e-8)
                matched.append((frame,pulse_strengths(frame),table.colors.copy()))
        results.append(matched)
    for other in results[1:]:
        for (a,pa,ca),(b,pb,cb) in zip(results[0],other):
            assert a==b
            np.testing.assert_array_equal(pa,pb);np.testing.assert_array_equal(ca,cb)
