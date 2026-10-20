"""Native teacher preview for the finished display-only fade change."""
import torch
torch.set_num_threads(1)
from flybrain.flight3d.anatomical_viewer import FlightViewer

app=FlightViewer(970000,'teacher')
app.all_activity()
app.toggle_settings()
app.run()
