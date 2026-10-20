"""Each neuron's last spike fades smoothly without changing recorded history."""
from dataclasses import replace
import numpy as np
import pytest
from test_flight3d_activity import snapshot


def test_quiet_neuron_has_a_visible_eased_fade_and_no_expiry_jump():
    from flybrain.flight3d.activity import SpikeRecorder,pulse_strengths
    recorder=SpikeRecorder(snapshot())
    frame=recorder.observe(snapshot(.02,(1,0)))
    samples=[(0,1),(.02,.9967407407407407),(.2,20/27),(.3,.5),
             (.4,7/27),(.58,.003259259259259),(.6,0),(.8,0)]
    for age,want in samples:
        at=replace(frame,snapshot=replace(frame.snapshot,time=.02+age))
        np.testing.assert_allclose(pulse_strengths(at),(want,0),atol=1e-12)
    # Both ends ease into/out of the fade, including the 600 ms boundary.
    for age in (.000001,.599999):
        strength=pulse_strengths(replace(frame,snapshot=replace(frame.snapshot,time=.02+age)))[0]
        assert min(strength,1-strength)<1e-10
    at_boundary=replace(frame,snapshot=replace(frame.snapshot,time=.02+.6-1e-13))
    assert pulse_strengths(at_boundary)[0]==0  # Ignore floating-point tick roundoff.
    assert recorder.frame is frame and frame.cumulative_counts==(1,0)


def test_independent_last_spikes_retrigger_pause_mask_and_reset():
    from flybrain.flight3d.activity import SpikeRecorder,pulse_strengths
    from flybrain.flight3d.branch_activity import BranchActivityTable
    recorder=SpikeRecorder(snapshot())
    first=recorder.observe(snapshot(.02,(1,0)))
    recorder.observe(snapshot(.12,(0,1)))
    quiet=recorder.observe(snapshot(.32,(0,0)))
    np.testing.assert_allclose(pulse_strengths(quiet),(.5,20/27))
    palette=np.array(((230,180,80),(241,138,72)))/255
    table=BranchActivityTable(quiet.snapshot.neuron_ids,palette)
    table.update(quiet,False,(True,True))
    idle=np.array((.24,.30,.55,.22))
    np.testing.assert_allclose(table.colors[0],idle+(np.r_[palette[0],.43]-idle)*.5)
    frozen=table.colors.copy();uploads=table.uploads
    for _ in range(10):table.update(quiet,False,(True,True))
    np.testing.assert_array_equal(table.colors,frozen);assert table.uploads==uploads
    table.update(quiet,False,(False,True))
    np.testing.assert_allclose(table.colors[0],idle)
    table.update(quiet,False,(True,True))
    np.testing.assert_array_equal(table.colors,frozen)
    refreshed=recorder.observe(snapshot(.34,(1,0)))
    assert pulse_strengths(refreshed)[0]==1
    assert refreshed.cumulative_counts==(2,1) and first.cumulative_counts==(1,0)
    reset=recorder.reset(snapshot());table.update(reset,False,(True,True))
    np.testing.assert_allclose(table.colors,np.tile(idle,(2,1)))
    assert reset.cumulative_counts==(0,0)


def test_quiet_fade_at_shared_times_is_equal_at_30_60_120_display_hz():
    from flybrain.flight3d.activity import SpikeRecorder,pulse_strengths
    from flybrain.flight3d.branch_activity import BranchActivityTable
    stream=[snapshot(i/50,(1 if i==1 else 0,1 if i==6 else 0)) for i in range(1,51)]
    results=[]
    for hz in (30,60,120):
        recorder=SpikeRecorder(snapshot());interval=0;shared=[]
        table=BranchActivityTable(('0','1'),np.array(((230,180,80),(241,138,72)))/255)
        for draw in range(1,hz+1):
            while interval<len(stream) and stream[interval].time<=draw/hz+1e-10:
                recorder.observe(stream[interval]);interval+=1
            table.update(recorder.frame,False,(True,True))
            if draw%(hz//5)==0:
                shared.append((recorder.frame,pulse_strengths(recorder.frame),table.colors.copy()))
        results.append(shared)
    assert results[0][0][1][0]>0 and results[0][-1][1][0]==0
    for other in results[1:]:
        for (a,pa,ca),(b,pb,cb) in zip(results[0],other):
            assert a==b
            np.testing.assert_array_equal(pa,pb);np.testing.assert_array_equal(ca,cb)


def test_actual_gpu_afterglow_interpolates_class_color_and_alpha_to_idle():
    import torch
    from flybrain.flight3d.activity import SpikeRecorder
    from flybrain.flight3d.branch_activity import BranchActivityTable
    from flybrain.flight3d.anatomical_viewer import FlightViewer
    from panda3d.core import OrthographicLens,Camera,NodePath,PNMImage
    torch.set_num_threads(1)
    app=FlightViewer(970000,'teacher',offscreen=True,window_size=(320,240))
    buffer=None;root=None
    try:
        app.paused=True;app.simulation.pause()
        recorder=SpikeRecorder(snapshot(counts=(0,)))
        frame=recorder.observe(snapshot(.02,(1,)))
        palette=np.array(((230,180,80),))/255
        table=BranchActivityTable(('0',),palette)
        root=NodePath('fade-pixel-test')
        node=table.make_batch(np.array([[[0,0,-.6],[0,0,.6]]]),np.array([0]),'fading-line')
        node.setRenderModeThickness(10);node.reparentTo(root)
        buffer=app.win.makeTextureBuffer('afterglow',100,100)
        lens=OrthographicLens();lens.setFilmSize(2,2)
        camera=root.attachNewNode(Camera('afterglow-camera',lens));camera.setPos(0,-2,0)
        region=buffer.makeDisplayRegion();region.setCamera(camera)
        buffer.setClearColor((0,0,0,0));buffer.setClearColorActive(True)
        idle=np.array((.24,.30,.55,.22));active=np.r_[palette[0],.43]
        pixels=[]
        for time,strength in ((.02,1),(.32,.5),(.60,.003259259259259),(.62,0)):
            if time>.02:frame=recorder.observe(snapshot(time,(0,)))
            table.update(frame,False,(True,))
            for _ in range(3):app.graphicsEngine.renderFrame();app.graphicsEngine.syncFrame()
            image=PNMImage();assert buffer.getScreenshot(image)
            actual=np.array(tuple(image.getXel(50,50)))
            rgba=idle+(active-idle)*strength
            np.testing.assert_allclose(actual,rgba[:3]*rgba[3],atol=.01)
            pixels.append(actual)
        assert np.linalg.norm(pixels[-2]-pixels[-1])<.008
        assert recorder.frame.cumulative_counts==(1,)
        assert app.session.snapshot.time==0  # Rendering did not step the real session.
    finally:
        if buffer is not None:app.graphicsEngine.removeWindow(buffer)
        if root is not None:root.removeNode()
        app.destroy()
