from pathlib import Path
import torch
torch.set_num_threads(1)
from flybrain.flight3d.anatomical_viewer import FlightViewer
from panda3d.core import ClockObject,PNMImage
root=Path('.superpowers/navigation-2026-10-02/frames')
root.mkdir(parents=True,exist_ok=True)
for size in ((1440,900),(1000,700)):
    app=FlightViewer(970000,'teacher',offscreen=True,window_size=size)
    clock=ClockObject.getGlobalClock();clock.setMode(ClockObject.MNonRealTime);clock.setDt(1/60)
    try:
        app.paused=True;app.load_geometry(100000)
        def save(name):
            app.draw_snapshot(app.session.snapshot)
            for _ in range(3):app.taskMgr.step();app.graphicsEngine.syncFrame()
            picture=PNMImage();app.win.getScreenshot(picture);picture.removeAlpha();picture.write(str(root/f'{size[0]}-{name}.png'))
        save('initial')
        app.toggle_settings();save('settings');app.toggle_settings()
        id=next(n['id'] for n in app.manifest['nodes'] if n['id'] in app.loaded and n['metadata']['region']=='optic_lobe')
        app.select(id);app._selection_pending[1].result();app.load_geometry(100000)
        app.toggle_edges();save('inspector')
        app.focus_selected();app.brain_motion.settle();app._brain_camera();save('focus-selected')
        app.brain_motion.zoom(.00001);app.brain_motion.settle();app._brain_camera();save('deep-zoom')
        if app.views_button is not None:app.toggle_views();save('views-menu')
    finally:app.destroy();clock.setMode(ClockObject.MNormal)
