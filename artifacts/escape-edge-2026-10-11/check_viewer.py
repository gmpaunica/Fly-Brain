"""Zero-step UI render; this is distinct from a live flight review."""
import json
from pathlib import Path
from flybrain.escape_edge.viewer import EdgeViewer
from flybrain.escape_recovery.config import CandidateConfig
from panda3d.core import PNMImage

art = Path('artifacts/escape-edge-2026-10-11')
app = EdgeViewer(1110000,CandidateConfig.create(),offscreen=True)
try:
    before = app.session.snapshot
    app.capture_paused(art/'viewer-preflight.png')
    assert app.session.snapshot==before
    assert app.session.world.time==app.session.brain.steps==app.session.pair.reference.steps==0
    (art/'offscreen.json').write_text(json.dumps(dict(status='passed',neural_steps=0,
        simulation_seconds=0,visual_review='pending',width=app.win.getXSize(),height=app.win.getYSize())))
    print('Zero-step edge viewer rendered; simulation/neurons unchanged.')
finally:
    app.destroy()
