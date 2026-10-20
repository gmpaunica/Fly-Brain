"""Separate passive UI/renderer evidence; this is not human live acceptance."""
import numpy as np
import pytest
import torch
from flybrain.encounters3d.diagnostic_panel import pathway_text
from flybrain.encounters3d.scene import SceneSpec
from flybrain.encounters3d.session import EncounterSession
from flybrain.flight3d.activity import SpikeRecorder


def test_panel_text_reads_completed_record_only():
    s = EncounterSession(SceneSpec(3, 'predator', (.34, .42, .1)))
    recorder = SpikeRecorder(s.snapshot)
    assert 'waiting' in pathway_text(recorder.frame, s.manifest)
    frame = recorder.observe(s.advance())
    before = s.brain.voltage.clone()
    intervals = tuple(s.brain.diagnostic_intervals)
    for _ in range(8): text = pathway_text(frame, s.manifest)
    assert 'DNp03' in text and 'DNp26' in text and 'phase peak' in text
    assert 'Wing-control neurons' in text and 'V peak' in text
    assert 'I discarded sum' in text
    assert intervals == tuple(s.brain.diagnostic_intervals)
    assert torch.equal(before, s.brain.voltage)


@pytest.mark.parametrize('size', [(1000, 700), (1440, 900)])
def test_real_panel_keeps_preferences_pause_exact_replay_and_passive_render(size):
    from flybrain.encounters3d.viewer import EncounterViewer
    app = EncounterViewer(990000, 'predator', offscreen=True, window_size=size)
    try:
        app.paused = True; app.simulation.pause()
        app.all_activity(); app.toggle_settings(); app.toggle_vnc()
        app.brain_motion.rotate_pixels(20, 30); app.brain_motion.advance(.1); app._brain_camera()
        camera = np.array(app.brain_camera.getMat())
        preferences = app.enabled_classes, app.activity_scope, app.settings_open, app.vnc
        snapshots = [app.simulation.advance_fixed(1) for _ in range(5)]
        frame = app.simulation.display_frame
        before = app.session.brain.voltage.clone()
        for _ in range(4):
            app.draw_frame(frame); app.graphicsEngine.renderFrame()
        assert 'phase peak' in app.pathway_label.getText()
        assert torch.equal(before, app.session.brain.voltage)
        assert frame is app.simulation.display_frame
        app.replay()
        assert 'waiting' in app.pathway_label.getText()
        assert app.session.brain.diagnostic_intervals == []
        assert snapshots == [app.simulation.advance_fixed(1) for _ in range(5)]
        app.spawn('flower')
        assert 'waiting' in app.pathway_label.getText() and app.paused
        assert preferences == (app.enabled_classes, app.activity_scope, app.settings_open, app.vnc)
        np.testing.assert_array_equal(camera, np.array(app.brain_camera.getMat()))
        assert 'teacher' in app.help.getText()
        # Layout: three stages and both DN subgroups fit within panel content width.
        bounds = app.pathway_label.getTightBounds()
        assert float(bounds[1].x) < .9 * app.getAspectRatio()
        assert float(bounds[0].z) > -.47  # retained guide begins below diagnostic block
    finally: app.destroy()
