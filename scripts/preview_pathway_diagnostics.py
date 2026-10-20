"""Open the diagnostic UI paused after one development second, no investigation."""
import torch
import json
from pathlib import Path
torch.set_num_threads(1)
from flybrain.encounters3d.viewer import EncounterViewer

app = EncounterViewer(990000, 'predator')
app.paused = True
app.simulation.advance_fixed(50)
app.draw_frame(app.simulation.display_frame)
(Path('artifacts/pathway-diagnostics-2026-10-05')/'preview-ready.json').write_text(
    json.dumps({'seed':990000, 'paused':True, 'simulation_seconds':app.session.snapshot.time})+'\n')
app.run()
