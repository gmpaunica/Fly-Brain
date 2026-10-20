"""Minimal zero-step render reproduction with the normal application tasks."""
from pathlib import Path
from flybrain.escape_edge.viewer import EdgeViewer
from flybrain.escape_recovery.config import CandidateConfig
from panda3d.core import PNMImage

art = Path('artifacts/escape-edge-2026-10-11')
app = EdgeViewer(1110000,CandidateConfig.create(),offscreen=True)
try:
    app.paused = True
    app.simulation.pause()
    for i in range(20):
        app.taskMgr.step()
        app.graphicsEngine.syncFrame()
        if i in (2,19):
            image = PNMImage()
            app.win.getScreenshot(image)
            image.write(str(art/f'task-frame-{i}.png'))
            print(i,app.title.getText(),app.hud.getText(),app.session.world.time,app.session.brain.steps,flush=True)
    assert app.session.world.time==app.session.brain.steps==app.session.pair.reference.steps==0
finally:
    app.destroy()
