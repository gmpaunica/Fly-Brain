"""Compare UI frame latency and simulation cost; no fitting or behavioral assay."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch


def benchmark(view: str, frames: int=120, cord: bool=False, activity: str='all') -> dict:
    """Warm identical geometry, then measure complete synchronized display frames."""
    torch.set_num_threads(1)
    if view=='previous-anatomy':
        from flybrain.flight3d.viewer import FlightViewer
    elif view=='network':
        from flybrain.flight3d.network_viewer import FlightViewer
    elif view=='repaired':
        from flybrain.flight3d.anatomical_viewer import FlightViewer
    else: raise ValueError(view)
    from panda3d.core import ClockObject
    from flybrain.flight3d.anatomy import CACHE
    app=FlightViewer(970000,'teacher',offscreen=True,window_size=(1440,900))
    clock=ClockObject.getGlobalClock();clock.setMode(ClockObject.MNonRealTime);clock.setDt(1/60)
    advance=app.session.advance; simulation_times=[]
    def measured_advance(*args,**kwargs):
        start=time.perf_counter();result=advance(*args,**kwargs)
        simulation_times.append(time.perf_counter()-start);return result
    app.session.advance=measured_advance
    try:
        app.paused=True
        if hasattr(app,'all_activity') and activity=='all':app.all_activity()
        if hasattr(app,'simulation'):app.simulation.pause()
        app.focus_brain();app.preset('front')
        if app.vnc!=cord:app.toggle_vnc()
        app.load_geometry(100000)
        if cord and hasattr(app,'fit_view'):
            app.fit_view();app.brain_motion.settle();app._brain_camera()
        if hasattr(app,'simulation'):
            app.simulation.advance_fixed(100);app.draw_frame(app.simulation.display_frame)
        else:
            for _ in range(50):advance()
        ids=[n['id'] for n in app.manifest['nodes'] if (CACHE/'skeletons'/f"{n['id']}.npz").exists()
             and n['metadata'].get('region')!='ventral_nerve_cord'][::30][:6]
        result={'view':view,'frames_per_condition':frames,'resolution':[1440,900],
                'seed':970000,'cord_visible':cord,'activity_classes':activity,'python':platform.python_version(),'torch':torch.__version__,
                'renderer':app.win.getGsg().getDriverRenderer(),'target_display_hz':60,'conditions':{}}
        for condition in ('paused','running','orbiting','selecting'):
            app.paused=condition!='running';simulation_times.clear()
            for _ in range(12):app.taskMgr.step();app.graphicsEngine.syncFrame()
            # Establish a clean measurement boundary for an in-flight worker.
            if hasattr(app,'simulation'):app.simulation.pause()
            timings=[];selection=[];start_time=app.session.world.time
            active=[]; full_firing=[]; enabled_firing=[]
            simulation_times.clear()
            wall_start=time.perf_counter()
            for frame in range(frames):
                start=time.perf_counter()
                if condition=='orbiting':
                    if hasattr(app.brain_motion,'rotate_pixels'):app.brain_motion.rotate_pixels(1.33,0)
                    else:app.azimuth+=.008
                    app._brain_camera()
                if condition=='selecting' and frame%20==0:
                    selection_start=time.perf_counter();app.select(ids[(frame//20)%len(ids)])
                    selection.append(time.perf_counter()-selection_start)
                app.taskMgr.step();app.graphicsEngine.syncFrame()
                timings.append(time.perf_counter()-start)
                if hasattr(app,'visible_activity_ids'):active.append(len(app.visible_activity_ids))
                if hasattr(app,'activity_mask'):
                    full_firing.append(sum(c>0 for c in app._display_frame.snapshot.spike_counts))
                    enabled_firing.append(app.enabled_firing)
                time.sleep(max(0.,1/60-(time.perf_counter()-start)))
            barrier_start=time.perf_counter()
            if hasattr(app,'simulation'):app.simulation.pause()
            barrier_ms=(time.perf_counter()-barrier_start)*1000
            elapsed=time.perf_counter()-wall_start
            total=sum(timings);values=np.array(timings)*1000
            result['conditions'][condition]={'mean_ms':float(values.mean()),'p95_ms':float(np.percentile(values,95)),
                'max_ms':float(values.max()),'frames_per_wall_second':frames/total,
                'paced_frames_per_wall_second':frames/elapsed,'pause_barrier_ms':barrier_ms,
                'simulation_seconds':app.session.world.time-start_time,'simulation_ticks':len(simulation_times),
                'simulation_compute_ms':sum(simulation_times)*1000,
                'simulation_step_p95_ms':float(np.percentile(simulation_times,95)*1000) if simulation_times else None,
                'simulation_seconds_per_wall_second':(app.session.world.time-start_time)/elapsed,
                'selection_ms':[v*1000 for v in selection],
                'visible_pulses_min':min(active) if active else None,
                'visible_pulses_max':max(active) if active else None,
                'full_model_firing_min':min(full_firing) if full_firing else None,
                'full_model_firing_max':max(full_firing) if full_firing else None,
                'enabled_firing_min':min(enabled_firing) if enabled_firing else None,
                'enabled_firing_max':max(enabled_firing) if enabled_firing else None,
                'source_batches':len(app.source_batches) if hasattr(app,'source_batches') else None,
                'geometry_nodes':app.brain_scene.findAllMatches('**/+GeomNode').getNumPaths()}
        result['gates']={'p95_below_33_ms':all(v['p95_ms']<33 for v in result['conditions'].values()),
                         'hundreds_visible_active':all(v['visible_pulses_min'] is not None and v['visible_pulses_min']>=200
                                                      for v in result['conditions'].values()) if view=='repaired' and activity=='all' else None}
        return result
    finally:app.destroy();clock.setMode(ClockObject.MNormal)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view',choices=['previous-anatomy','network','repaired'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=120)
    parser.add_argument('--cord',choices=('hidden','shown'),default='hidden')
    parser.add_argument('--activity',choices=('all','odor'),default='all')
    args=parser.parse_args();record=benchmark(args.view,args.frames,args.cord=='shown',args.activity)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
