"""Narrow reproduction of selection frame work after the failed timing gate."""
from pathlib import Path
import cProfile
import pstats
import torch
torch.set_num_threads(1)
from flybrain.flight3d.anatomical_viewer import FlightViewer
from flybrain.flight3d.anatomy import CACHE
from panda3d.core import ClockObject
out=Path(__file__).resolve().parent
app=FlightViewer(970000,'teacher',offscreen=True,window_size=(1440,900))
clock=ClockObject.getGlobalClock();clock.setMode(ClockObject.MNonRealTime);clock.setDt(1/60)
try:
    app.paused=True;app.simulation.pause();app.load_geometry(100000)
    app.simulation.advance_fixed(100);app.draw_frame(app.simulation.display_frame)
    ids=[n['id'] for n in app.manifest['nodes'] if (CACHE/'skeletons'/f"{n['id']}.npz").exists()
         and n['metadata'].get('region')!='ventral_nerve_cord'][::30][:6]
    profile=cProfile.Profile();profile.enable()
    for id in ids:
        app.select(id)
        for _ in range(20):app.taskMgr.step();app.graphicsEngine.syncFrame()
    profile.disable()
    with (out/'selection-profile.txt').open('w') as stream:
        pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(35)
finally:app.destroy();clock.setMode(ClockObject.MNormal)
