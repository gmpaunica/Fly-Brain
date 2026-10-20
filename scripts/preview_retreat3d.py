"""Open additive lab paused during engineered idle; no scientific assay."""
import json
from pathlib import Path
import torch
torch.set_num_threads(1)
from flybrain.retreat3d.viewer import RetreatViewer

app = RetreatViewer(1000000)
app.paused = True; app.simulation.advance_fixed(140)
app.draw_frame(app.simulation.display_frame)
Path('artifacts/retreat3d-2026-10-06/preview-ready.json').write_text(json.dumps(dict(
    seed=1000000, paused=True, seconds=app.session.snapshot.time,
    state=app.session.snapshot.behavior_state, escape_supported=False)), encoding='utf-8')
app.run()
