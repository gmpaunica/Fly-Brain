"""Open the approved dashboard paused for human visual/input review."""
import torch
from flybrain.escape_edge.delivery import verified_candidate
from flybrain.escape_ui.viewer import DemoViewer

torch.set_num_threads(1)
candidate = verified_candidate()
if candidate is None:
    raise SystemExit("Verified candidate unavailable")
app = DemoViewer(1110000, candidate, verified=True)
try:
    app.paused = True
    app.simulation.pause()
    app.draw_frame(app.simulation.display_frame)
    app.run()
finally:
    app.destroy()
