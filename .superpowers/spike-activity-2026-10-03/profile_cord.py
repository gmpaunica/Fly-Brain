"""Isolate GPU branch coloring cost without changing geometry or simulation."""
import json,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from flybrain.flight3d.anatomical_viewer import FlightViewer
from panda3d.core import ClockObject
out=Path(__file__).resolve().parent
app=FlightViewer(970000,'teacher',offscreen=True,window_size=(1440,900))
clock=ClockObject.getGlobalClock();clock.setMode(ClockObject.MNonRealTime);clock.setDt(1/60)
try:
    app.paused=True;app.simulation.pause();app.toggle_vnc();app.load_geometry(100000)
    app.fit_view();app.brain_motion.settle();app._brain_camera();app.simulation.advance_fixed(100)
    app.draw_frame(app.simulation.display_frame)
    records={'segments':sum(n.node().getGeom(0).getVertexData().getNumRows()//2 for n in app.source_batches),
             'batches':len(app.source_batches),'conditions':{}}
    for variant in ('activity-shader','fixed-color','fragment-lookup'):
        if variant=='fixed-color':
            for node in app.source_batches:node.setShaderOff();node.setColor(.24,.30,.55,.22)
        if variant=='fragment-lookup':
            from panda3d.core import Shader
            shader=Shader.make(Shader.SLGLSL,'''#version 130
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex; in float activity_index; flat out float owner;
void main(){gl_Position=p3d_ModelViewProjectionMatrix*p3d_Vertex;owner=activity_index;}
''','''#version 130
uniform sampler2D activity_colors;uniform float activity_width;
flat in float owner;out vec4 fragment_color;
void main(){fragment_color=texture(activity_colors,vec2((owner+.5)/activity_width,.5));}
''')
            for node in app.source_batches:node.clearColor();node.setShader(shader)
        for _ in range(12):app.taskMgr.step();app.graphicsEngine.syncFrame()
        values=[]
        for _ in range(120):
            start=time.perf_counter();app.taskMgr.step();app.graphicsEngine.syncFrame()
            values.append((time.perf_counter()-start)*1000)
            time.sleep(max(0.,1/60-(time.perf_counter()-start)))
        records['conditions'][variant]={'mean_ms':float(np.mean(values)),'p95_ms':float(np.percentile(values,95))}
    (out/'cord-profile.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
finally:app.destroy();clock.setMode(ClockObject.MNormal)
