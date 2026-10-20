"""Display membership must never suppress recording or invent graph partners."""
import json
from pathlib import Path
import numpy as np
import pytest


def index_for_fixture():
    from flybrain.flight3d.activity_classes import DisplayActivityIndex
    return DisplayActivityIndex({
        'nodes': [{'id': i, 'roles': [r]} for i, r in
                  [('a', 'odor'), ('b', 'vision'), ('c', 'wing_power'),
                   ('d', 'wing_steering'), ('e', 'touch')]],
        'edges': [{'pre': a, 'post': b} for a, b in
                  [('a', 'a'), ('a', 'b'), ('b', 'c'), ('d', 'b'), ('a', 'b')]]})


def test_manifest_classes_are_complete_disjoint_and_preserve_roles():
    from flybrain.flight3d.activity_classes import DisplayActivityIndex
    manifest = json.loads(Path('data/flight3d/circuit.json').read_text())
    original = json.dumps(manifest, sort_keys=True)
    index = DisplayActivityIndex(manifest)
    assert len(index.ids) == 589
    assert [len(index.members[c.key]) for c in index.classes] == [69,96,20,16,8,8,2,88,41,241]
    owners = [i for group in index.members.values() for i in group]
    assert sorted(owners) == list(range(589))
    assert json.dumps(manifest, sort_keys=True) == original
    for roles in ([], ['unmapped'], ['odor', 'vision']):
        with pytest.raises(ValueError):
            DisplayActivityIndex({'nodes': [{'id': 'a', 'roles': roles}], 'edges': []})


def test_independent_classes_and_sensory_parent_mixed_on_off():
    index = index_for_fixture()
    enabled = frozenset({'odor', 'motor'})
    assert index.sensory_state(enabled) == 'mixed'
    enabled = index.toggle_sensory(enabled)
    assert index.sensory_state(enabled) == 'on' and 'motor' in enabled
    assert index.mask(enabled) == (True, True, True, True, True)
    enabled = index.toggle_sensory(enabled)
    assert index.sensory_state(enabled) == 'off' and enabled == frozenset({'motor'})
    assert index.mask(enabled) == (False, False, True, True, False)
    assert index.sensory_state(index.toggle_sensory(frozenset())) == 'on'
    assert index.mask(frozenset(index.members)) == (True,)*5
    assert index.mask(frozenset()) == (False,)*5


def test_selected_scope_uses_direct_nonself_partners_and_class_intersection():
    index = index_for_fixture()
    all_classes = frozenset(index.members)
    assert index.mask(all_classes, 'selected', 'b') == (True, True, True, True, False)
    assert index.mask(frozenset({'motor'}), 'selected', 'b') == (False, False, True, True, False)
    # c has no outgoing edges; e has none at all. Neither is excluded.
    assert index.mask(all_classes, 'selected', 'c') == (False, True, True, False, False)
    assert index.mask(all_classes, 'selected', 'e') == (False, False, False, False, True)
    with pytest.raises(ValueError): index.mask(all_classes, 'selected', None)
    with pytest.raises(ValueError): index.mask(all_classes, 'selected', 'background')


def test_class_colors_mask_decay_and_paused_cache_invalidation():
    from flybrain.flight3d.activity import SpikeRecorder
    from flybrain.flight3d.branch_activity import BranchActivityTable
    from test_flight3d_activity import snapshot
    recorder = SpikeRecorder(snapshot(counts=(0,0,0)))
    frame = recorder.observe(snapshot(.02, (1,1,0)))
    palette = np.array([[230,180,80], [134,217,122], [241,138,72]]) / 255
    table = BranchActivityTable(frame.snapshot.neuron_ids, activation_colors=palette)
    table.update(frame, False, (True,False,True))
    np.testing.assert_allclose(table.colors[0], (*palette[0],.43))
    np.testing.assert_allclose(table.colors[1:], np.tile((.24,.30,.55,.22),(2,1)))
    before = table.uploads
    table.update(frame, False, (True,True,True))
    assert table.uploads == before+1
    np.testing.assert_allclose(table.colors[1], (*palette[1],.43))
    table.update(frame, False, (True,True,True)); assert table.uploads == before+1
    decayed = recorder.observe(snapshot(.22, (0,0,0)))
    table.update(decayed, False, (True,True,True))
    expected = np.array((.24,.30,.55,.22)) + (np.r_[palette[0],.43]-np.array((.24,.30,.55,.22)))*(20/27)
    np.testing.assert_allclose(table.colors[0], expected)
    expired = recorder.observe(snapshot(.62, (0,0,0)))
    table.update(expired, False, (True,True,True))
    np.testing.assert_allclose(table.colors, np.tile((.24,.30,.55,.22),(3,1)))
    assert frame.cumulative_counts == (1,1,0)
    with pytest.raises(ValueError): table.update(frame, False, (True,))


def test_actual_gpu_class_endpoints_and_paused_mask_recoloring():
    from flybrain.flight3d.activity_classes import CLASSES
    from flybrain.flight3d.activity import SpikeRecorder
    from flybrain.flight3d.branch_activity import BranchActivityTable
    from flybrain.flight3d.anatomical_viewer import FlightViewer
    from test_flight3d_activity import snapshot
    from panda3d.core import OrthographicLens, Camera, NodePath, PNMImage
    import torch
    torch.set_num_threads(1)
    app = FlightViewer(970000,'teacher',offscreen=True,window_size=(320,240))
    try:
        app.paused=True; app.simulation.pause()
        recorder=SpikeRecorder(snapshot(counts=(0,)*10))
        frame=recorder.observe(snapshot(.02,(1,)*10))
        table=BranchActivityTable(frame.snapshot.neuron_ids,np.array([c.rgb for c in CLASSES]))
        table.update(frame,False,(True,)*10)
        xs=np.linspace(-.9,.9,10)
        lines=np.array([[[x,0,-.6],[x,0,.6]] for x in xs])
        root=NodePath('class-colors-test')
        node=table.make_batch(lines,np.arange(10),'ten-class-endpoints')
        node.setRenderModeThickness(10);node.reparentTo(root)
        buffer=app.win.makeTextureBuffer('class-palette',500,100)
        lens=OrthographicLens();lens.setFilmSize(2,2)
        camera=root.attachNewNode(Camera('class-palette-camera',lens));camera.setPos(0,-2,0)
        region=buffer.makeDisplayRegion();region.setCamera(camera)
        buffer.setClearColor((0,0,0,0));buffer.setClearColorActive(True)
        expected_hex=['E6B450','86D97A','3A9D6D','E889B5','E75C5C','C7DC66','C59BE8','F18A48','B378D9','D7D7DB']
        def render():
            for _ in range(3):app.graphicsEngine.renderFrame();app.graphicsEngine.syncFrame()
            picture=PNMImage();assert buffer.getScreenshot(picture)
            return picture
        picture=render()
        for i,color in enumerate(expected_hex):
            expected=np.array([int(color[j:j+2],16)/255 for j in (0,2,4)])*.43
            np.testing.assert_allclose(tuple(picture.getXel(25+50*i,50)),expected,atol=.012)
        # Same frame, no simulation tick: a disabled switch returns its GPU texel to idle.
        table.update(frame,False,(False,)+(True,)*9)
        picture=render()
        np.testing.assert_allclose(tuple(picture.getXel(25,50)),np.array((.24,.30,.55))*.22,atol=.012)
        assert frame.cumulative_counts==(1,)*10
        app.graphicsEngine.removeWindow(buffer);root.removeNode()
    finally: app.destroy()
