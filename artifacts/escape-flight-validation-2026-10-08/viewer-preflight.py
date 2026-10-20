from pathlib import Path
from flybrain.escape_recovery.config import CandidateConfig
from flybrain.escape_flight_validation.viewer import PrototypeViewer
from panda3d.core import PNMImage
app=PrototypeViewer(1210100,CandidateConfig.create(),offscreen=True)
try:
    app.paused=True;app.simulation.pause()
    app.title.setText('PREFLIGHT RENDER / physical validation not yet run')
    app.draw_frame(app.simulation.display_frame)
    app.title.setText('PREFLIGHT RENDER / physical validation not yet run')
    for _ in range(3):app.graphicsEngine.renderFrame();app.graphicsEngine.syncFrame()
    image=PNMImage();app.win.getScreenshot(image)
    assert image.write('artifacts/escape-flight-validation-2026-10-08/viewer-preflight.png')
    assert app.session.world.time==0 and app.session.brain.steps==app.session.pair.reference.steps==0
    print('Viewer construction/render passed; world time and paired brain steps remain zero.')
finally:
    app.destroy()
