"""Saved renderer views and bounded teacher display diagnostic; no learning."""
import json
from pathlib import Path
import torch
torch.set_num_threads(1)
from flybrain.flight3d.anatomical_viewer import FlightViewer
from flybrain.flight3d.activity import pulse_strengths
from panda3d.core import PNMImage,ClockObject,Filename

out=Path(__file__).resolve().parent
frames=out/'frames';frames.mkdir(exist_ok=True)
clock=ClockObject.getGlobalClock();clock.setMode(ClockObject.MNonRealTime);clock.setDt(1/60)
records=[]
for size in ((1440,900),(1000,700)):
    app=FlightViewer(970000,'teacher',offscreen=True,window_size=size)
    try:
        app.paused=True;app.simulation.pause();app.load_geometry(100000)
        def save(name):
            app.draw_frame(app.simulation.display_frame)
            for _ in range(3):app.taskMgr.step();app.graphicsEngine.syncFrame()
            image=PNMImage();assert app.win.getScreenshot(image);image.removeAlpha()
            path=frames/f'{size[0]}-{name}.png'
            assert image.write(Filename.fromOsSpecific(str(path)))
            assert path.is_file() and path.stat().st_size>0
        app.preset('front');save('idle-front')
        app.simulation.advance_fixed(100)
        frame=app.simulation.display_frame
        for name in ('front','side','top'):
            app.preset(name);app.brain_motion.settle();app._brain_camera();save('activity-'+name)
        hidden=app.visible_activity_ids
        app.toggle_vnc();app.load_geometry(100000);app.preset('front');app.fit_view();app.brain_motion.settle();app._brain_camera()
        save('brain-and-cord');shown=app.visible_activity_ids
        example='720575941546280072'
        app.select(example)
        if app._selection_pending:app._selection_pending[1].result()
        app.load_geometry(100000);save('inspector')
        app.select(sorted(app.source_missing)[0]);save('missing-geometry')
        if size[0]==1440:
            silent=[id for id,count in zip(frame.snapshot.neuron_ids,frame.cumulative_counts) if count==0]
            incoming={id:[e['pre'] for e in app.manifest['edges'] if e['post']==id and e['pre']!=id]
                      for id in silent}
            visual=[id for id in silent if 'vision' in app.nodes[id]['roles']]
            # Categories describe disjoint causes: visual inputs first. Some
            # visual cells also have only source self-edges; do not count them
            # again in the six remaining self-only nonvisual cells.
            selfonly=[id for id in silent if id not in visual and not incoming[id]
                      and any(e['post']==id and e['pre']==id for e in app.manifest['edges'])]
            remaining=[id for id in silent if id not in visual+selfonly]
            assert (len(silent),len(visual),len(selfonly),len(remaining))==(59,50,6,3)
            for id in remaining:
                assert incoming[id] and all(frame.cumulative_counts[app._owner_index[pre]]==0 for pre in incoming[id])
            record={'seed':970000,'mode':'teacher','simulation_seconds':frame.snapshot.time,'completed_intervals':frame.step_index,
                    'fired_since_reset':sum(v>0 for v in frame.cumulative_counts),'silent':len(silent),
                    'silence_categories':{'vision':visual,'self_only':selfonly,'silent_upstream':remaining},
                    'visible_pulses_cord_hidden':len(hidden),'visible_pulses_cord_shown':len(shown),
                    'example':{'id':example,'cumulative_spikes':frame.cumulative_counts[app._owner_index[example]]},
                    'cells':[{'id':id,'total':count,'latest':frame.snapshot.spike_counts[i],
                              'last_active':frame.last_active_times[i],'rate_hz':frame.firing_rates_hz[i],
                              'pulse':float(pulse_strengths(frame)[i])}
                             for i,(id,count) in enumerate(zip(frame.snapshot.neuron_ids,frame.cumulative_counts))]}
            (out/'activity-diagnostic.json').write_text(json.dumps(record,indent=2))
            app.select(selfonly[0]);save('silent-cell')
        records.append({'size':size,'snapshot_time':frame.snapshot.time,'hidden_pulses':len(hidden),'shown_pulses':len(shown)})
    finally:app.destroy()
clock.setMode(ClockObject.MNormal)
(out/'capture-summary.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
