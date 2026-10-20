"""Diagnostic rendering is passive and gate-failed launch remains historical."""
import importlib
from pathlib import Path
import numpy as np
from flybrain.flight3d.anatomy import load_manifest
from flybrain.turnflee.neural import ResponseConfig
from flybrain.turnflee.session import TurnFleeSession


def test_diagnostic_text_names_actual_spikes_and_engineered_ties():
    assert importlib.util.find_spec('flybrain.turnflee.viewer'), 'new response diagnostics required'
    m=importlib.import_module('flybrain.turnflee.viewer')
    s=TurnFleeSession(1150011,ResponseConfig.create(load_manifest(),(2,1)))
    before=s.brain.voltage.clone(); steps=s.brain.steps
    text=m.response_text(s.snapshot)
    assert 'Actual spikes' in text and '(2, 1)' in text and 'idle' in text
    assert 'engineering symmetry breaking' in text
    assert s.brain.steps==steps and (s.brain.voltage==before).all()


def test_offscreen_diagnostics_and_endpoint_replay(tmp_path):
    assert importlib.util.find_spec('flybrain.turnflee.viewer'), 'new response diagnostics required'
    m=importlib.import_module('flybrain.turnflee.viewer')
    app=m.TurnFleeViewer(1150012,ResponseConfig.create(load_manifest(),(2,1)),offscreen=True)
    try:
        app.paused=True; app.simulation.advance_fixed(3)
        app.spawn('predator'); app.simulation.advance_fixed(4)
        app.load_geometry(100000); before=app.simulation.display_frame
        app.draw_frame(before); app.graphicsEngine.renderFrame()
        assert app.simulation.display_frame is before
        assert app.session.brain.steps==80
        assert 'Response' in app.activity_label.getText()
        assert len(app.activity_label.getText().splitlines())<=2, 'top summary must fit above retained anatomy heading'
        app.replay(); assert app.paused
        assert app.simulation.display_frame.snapshot==before.snapshot
    finally: app.destroy()
