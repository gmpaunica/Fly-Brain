from flybrain.escape_repair.viewer import RepairViewer
from flybrain.escape_recovery.config import CandidateConfig
from panda3d.core import PNMImage

app = RepairViewer(1310100, CandidateConfig.create(), offscreen=True)
try:
    app.paused = True
    app.simulation.pause()
    app.draw_frame(app.simulation.display_frame)
    for _ in range(3):
        app.graphicsEngine.renderFrame()
        app.graphicsEngine.syncFrame()
    image = PNMImage()
    app.win.getScreenshot(image)
    assert image.write('artifacts/escape-repair-2026-10-08/viewer-preflight.png')
    assert app.session.world.time == app.session.brain.steps == app.session.pair.reference.steps == 0
    print('Repaired viewer renders at zero simulation/neural steps; validation-pending label.')
finally:
    app.destroy()
