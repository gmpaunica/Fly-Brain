"""Open the native review at a paused firing frame with class controls visible."""
import torch
torch.set_num_threads(1)
from flybrain.flight3d.anatomical_viewer import FlightViewer

app=FlightViewer(970000,'teacher')
app.paused=True
app.simulation.advance_fixed(100)
app.load_geometry(100000)
app.draw_frame(app.simulation.display_frame)
app.toggle_settings()
app.run()
