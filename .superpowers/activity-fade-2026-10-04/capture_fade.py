"""Actual GPU one-spike display fixture, explicitly not a neural experiment."""
from dataclasses import replace
import json
from pathlib import Path
import torch
torch.set_num_threads(1)
import numpy as np
from PIL import Image,ImageDraw
from panda3d.core import Camera,Filename,NodePath,OrthographicLens,PNMImage
from flybrain.flight3d.activity import SpikeRecorder,pulse_strengths
from flybrain.flight3d.activity_classes import CLASSES
from flybrain.flight3d.anatomical_viewer import FlightViewer
from flybrain.flight3d.branch_activity import BranchActivityTable
from flybrain.flight3d.world import FlightWorld,make_trial

out=Path(__file__).resolve().parent
frames=out/'frames';frames.mkdir(exist_ok=True)
app=FlightViewer(970000,'teacher',offscreen=True,window_size=(640,480))
buffer=None;root=None
try:
    app.paused=True;app.simulation.pause()
    initial=replace(FlightWorld(make_trial(970000)).snapshot(),neuron_ids=tuple(str(i) for i in range(10)),spike_counts=(0,)*10)
    recorder=SpikeRecorder(initial)
    frame=recorder.observe(replace(initial,time=.02,spike_counts=(1,)*10))
    table=BranchActivityTable(initial.neuron_ids,np.array([c.rgb for c in CLASSES]))
    root=NodePath('one-spike-display-fixture')
    node=table.make_batch(np.array([[[x,0,-.65],[x,0,.65]] for x in np.linspace(-.9,.9,10)]),np.arange(10),'class-fade')
    node.setRenderModeThickness(14);node.reparentTo(root)
    buffer=app.win.makeTextureBuffer('fade-visual-qa',600,180)
    lens=OrthographicLens();lens.setFilmSize(2,2)
    camera=root.attachNewNode(Camera('fade-qa-camera',lens));camera.setPos(0,-2,0)
    region=buffer.makeDisplayRegion();region.setCamera(camera)
    buffer.setClearColor((0,0,0,1));buffer.setClearColorActive(True)
    pictures=[];samples=[]
    for index,age in enumerate(np.arange(0,.801,.05)):
        if age>0:frame=recorder.observe(replace(initial,time=.02+float(age)))
        table.update(frame,False,(True,)*10)
        for _ in range(3):app.graphicsEngine.renderFrame();app.graphicsEngine.syncFrame()
        picture=PNMImage();assert buffer.getScreenshot(picture);picture.removeAlpha()
        path=frames/f'{index:02d}-{round(age*1000):03d}ms.png'
        assert picture.write(Filename.fromOsSpecific(str(path)))
        canvas=Image.new('RGB',(600,235),(8,7,14));canvas.paste(Image.open(path).convert('RGB'),(0,40))
        draw=ImageDraw.Draw(canvas)
        draw.text((10,7),'GPU display fixture: one spike, then silence (not a neural run)',fill=(220,215,230))
        draw.text((10,24),f'Age: {age:.2f} s | fade strength: {pulse_strengths(frame)[0]:.4f}',fill=(220,215,230))
        for i,c in enumerate(CLASSES):draw.text((i*60+12,219),str(i+1),fill=(220,215,230))
        pictures.append(canvas)
        samples.append({'age_seconds':float(age),'strength':float(pulse_strengths(frame)[0]),'rgba':table.colors.tolist()})
    pictures[0].save(out/'fade-demo.gif',save_all=True,append_images=pictures[1:],loop=0,
                     duration=[400]+[50]*15+[600])
    chosen=[0,3,6,9,12]
    sheet=Image.new('RGB',(600,235*len(chosen)),(8,7,14))
    for row,index in enumerate(chosen):sheet.paste(pictures[index],(0,row*235))
    sheet.save(out/'fade-contact-sheet.png')
    (out/'fade-samples.json').write_text(json.dumps({'fixture':'synthetic one-spike display only','classes':[c.label for c in CLASSES],'samples':samples},indent=2),encoding='utf-8')
    assert app.session.snapshot.time==0
    print('Saved actual GPU fade GIF, contact sheet, 17 frames and numerical samples.')
finally:
    if buffer is not None:app.graphicsEngine.removeWindow(buffer)
    if root is not None:root.removeNode()
    app.destroy()
