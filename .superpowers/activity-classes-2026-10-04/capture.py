"""Saved GPU views for layout review, separate from live interactions."""
import json
from pathlib import Path
import torch
torch.set_num_threads(1)
from flybrain.flight3d.anatomical_viewer import FlightViewer
from panda3d.core import PNMImage, Filename, ClockObject

out=Path(__file__).resolve().parent
frames=out/'frames';frames.mkdir(exist_ok=True)
clock=ClockObject.getGlobalClock();clock.setMode(ClockObject.MNonRealTime);clock.setDt(1/60)
results=[]
for size in ((1440,900),(1000,700)):
    app=FlightViewer(970000,'teacher',offscreen=True,window_size=size)
    try:
        app.paused=True;app.simulation.advance_fixed(100);app.load_geometry(100000)
        def save(name):
            for _ in range(3):app.taskMgr.step();app.graphicsEngine.syncFrame()
            picture=PNMImage();assert app.win.getScreenshot(picture);picture.removeAlpha()
            path=frames/f'{size[0]}-{name}.png'
            assert picture.write(Filename.fromOsSpecific(str(path))) and path.is_file()
        app.draw_frame(app.simulation.display_frame);save('odor-default')
        results.append({'size':size,'cord':False,'classes':'odor','visible_pulses':len(app.visible_activity_ids),'full_firing':sum(c>0 for c in app._display_frame.snapshot.spike_counts),'enabled_firing':app.enabled_firing})
        app.toggle_settings();save('settings-expanded')
        app.toggle_sensory_expansion();save('settings-collapsed')
        app.panels['details'].verticalScroll['value']=.48;save('settings-switches')
        app.all_activity();app.toggle_settings();save('all-activity')
        results.append({'size':size,'cord':False,'classes':'all','visible_pulses':len(app.visible_activity_ids),'full_firing':sum(c>0 for c in app._display_frame.snapshot.spike_counts),'enabled_firing':app.enabled_firing})
        odor=next(n['id'] for n in app.manifest['nodes'] if 'odor' in n['roles'])
        app.select(odor);app.set_activity_scope('selected')
        if app._selection_pending:app._selection_pending[1].result()
        app.load_geometry(100000);save('selected-circuit')
        app.activity_off();save('activity-off-selected')
        app.clear_selection();app.all_activity();app.toggle_vnc();app.load_geometry(100000)
        app.fit_view();app.brain_motion.settle();app._brain_camera();save('all-cord')
        results.append({'size':size,'cord':True,'classes':'all','visible_pulses':len(app.visible_activity_ids)})
        app.activity_off();app.toggle_activity_class('odor');save('odor-cord')
        results.append({'size':size,'cord':True,'classes':'odor','visible_pulses':len(app.visible_activity_ids)})
    finally:app.destroy()
clock.setMode(ClockObject.MNormal)
(out/'capture-summary.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
