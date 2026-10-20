import torch
torch.set_num_threads(1)
from flybrain.encounters3d.viewer import EncounterViewer
from panda3d.core import GeomVertexReader, PNMImage
app=EncounterViewer(980001,'predator',offscreen=True)
try:
    app.paused=True;app.simulation.pause()
    node=app.flower.find('**/predator-proxy')
    reader=GeomVertexReader(node.node().getGeom(0).getVertexData(),'color')
    print('vertex',reader.getData4f(),'state',node.getNetState())
    for _ in range(3):app.graphicsEngine.renderFrame()
    pic=PNMImage();app.win.getScreenshot(pic)
    pixels=[tuple(pic.getXel(x,y)) for x in range(460,505) for y in range(350,400)]
    print('brightest',max(pixels,key=sum))
finally:app.destroy()
