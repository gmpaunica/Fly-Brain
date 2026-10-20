"""Real GPU colors and segment ownership, including mixed/deferred anatomy."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_flight3d_activity import snapshot


def test_many_actual_spikes_color_without_edges_and_silent_cells_stay_idle():
    from flybrain.flight3d.activity import SpikeRecorder
    from flybrain.flight3d.branch_activity import BranchActivityTable
    ids = tuple(str(i) for i in range(21))
    table = BranchActivityTable(ids)
    recorder = SpikeRecorder(snapshot(counts=(0,)*21))
    frame = recorder.observe(snapshot(.02, (1,)*20+(0,)))
    table.update(frame, False)
    np.testing.assert_allclose(table.colors[:20], np.tile((1,.84,.07,.43), (20,1)), atol=1/255)
    np.testing.assert_allclose(table.colors[20], (.24,.30,.55,.22), atol=1/255)
    assert table.uploads == 1
    table.update(frame, False); assert table.uploads == 1
    table.update(frame, True); assert table.uploads == 2
    reset = recorder.reset(snapshot(counts=(0,)*21))
    table.update(reset, False)
    np.testing.assert_allclose(table.colors[0], (.24,.30,.55,.22), atol=1/255)


def test_rgba_shader_samples_each_owner_without_channel_swaps_or_interpolation():
    from flybrain.flight3d.anatomical_viewer import FlightViewer
    from flybrain.flight3d.branch_activity import BranchActivityTable
    from panda3d.core import PNMImage
    torch.set_num_threads(1)
    app = FlightViewer(970000, 'teacher', offscreen=True, window_size=(320,240))
    try:
        app.paused=True; app.simulation.pause()
        table = BranchActivityTable(('red', 'green', 'blue'))
        colors = np.array([[1,0,0,.25], [0,1,0,.5], [0,0,1,.75]], dtype=float)
        table.upload_colors(colors)
        np.testing.assert_array_equal(np.frombuffer(table.texture.getRamImageAs('RGBA'),dtype=np.uint8).reshape(3,4),
                                      np.rint(colors*255).astype(np.uint8))
        lines = np.array([[[-.7,0,-.6],[-.7,0,.6]], [[0,0,-.6],[0,0,.6]], [[.7,0,-.6],[.7,0,.6]]])
        node = table.make_batch(lines,np.arange(3),'owner-color-test')
        node.setRenderModeThickness(8); node.reparentTo(app.render2d)
        # Render into an isolated orthographic camera target: avoid app panels.
        from panda3d.core import OrthographicLens, Camera, NodePath
        buffer=app.win.makeTextureBuffer('rgba-test',120,100)
        root=NodePath('test-root'); node.reparentTo(root)
        lens=OrthographicLens(); lens.setFilmSize(2,2)
        camera=root.attachNewNode(Camera('rgba-camera',lens));camera.setPos(0,-2,0)
        region=buffer.makeDisplayRegion();region.setCamera(camera)
        buffer.setClearColor((0,0,0,0));buffer.setClearColorActive(True)
        for _ in range(3):app.graphicsEngine.renderFrame();app.graphicsEngine.syncFrame()
        picture=PNMImage();buffer.getScreenshot(picture)
        for x, channel, alpha in ((18,0,.25),(60,1,.5),(102,2,.75)):
            rgb=np.array(tuple(picture.getXel(x,50)))
            assert rgb[channel] == pytest.approx(alpha, abs=.02)
            assert np.delete(rgb,channel).max() < .02
        app.graphicsEngine.removeWindow(buffer);root.removeNode()
    finally: app.destroy()


def test_static_batch_endpoints_retain_owner_indices():
    from flybrain.flight3d.branch_activity import BranchActivityTable
    from panda3d.core import GeomVertexReader
    table=BranchActivityTable(('a','b','c'))
    lines=np.arange(18,dtype=np.float32).reshape(3,2,3)
    node=table.make_batch(lines,np.array([2,0,1]),'ownership')
    data=node.node().getGeom(0).getVertexData()
    owner=GeomVertexReader(data,'activity_index');position=GeomVertexReader(data,'vertex')
    assert [owner.getData1f() for _ in range(6)] == [2,2,0,0,1,1]
    np.testing.assert_array_equal([tuple(position.getData3f()) for _ in range(6)],lines.reshape(-1,3))
    with pytest.raises(ValueError):table.make_batch(lines,np.array([0,1]),'bad')
    with pytest.raises(ValueError):table.make_batch(lines,np.array([0,1,9]),'bad')
