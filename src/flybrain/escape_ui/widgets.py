"""Pixel-sized native DirectGUI controls with vector backgrounds and checkmarks."""
import math
from typing import Callable
import torch  # Initialize native DLLs before Panda3D.
from direct.gui import DirectGuiGlobals as DGG
from direct.gui.DirectGui import DirectButton, DirectCheckButton
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData,
                         GeomVertexFormat, GeomVertexWriter, LineSegs, NodePath, TextNode)

INK = (.92, .95, 1, 1)
MUTED = (.61, .68, .78, 1)
PANEL = (.045, .059, .083, 1)
BUTTON = (.095, .13, .18, 1)
ACCENT = (.18, .82, .86, 1)


def rounded(width: float, height: float, color: tuple, radius: float = 8) -> NodePath:
    """Build a small immutable rounded rectangle in the GUI's X/Z plane."""
    radius = min(radius, width/2, height/2)
    points = [(width/2, 0)]
    for x, z, start in ((width-radius, height/2-radius, 0),
                        (radius, height/2-radius, 90),
                        (radius, -height/2+radius, 180),
                        (width-radius, -height/2+radius, 270)):
        for step in range(6):
            angle = math.radians(start+step*18)
            points.append((x+radius*math.cos(angle), z+radius*math.sin(angle)))
    data = GeomVertexData('rounded-ui', GeomVertexFormat.getV3(), Geom.UHStatic)
    writer = GeomVertexWriter(data, 'vertex')
    for x, z in points:
        writer.addData3f(x, 0, z)
    triangles = GeomTriangles(Geom.UHStatic)
    for i in range(1, len(points)):
        triangles.addVertices(0, i, i+1 if i+1 < len(points) else 1)
    geom = Geom(data)
    geom.addPrimitive(triangles)
    node = GeomNode('rounded-ui')
    node.addGeom(geom)
    path = NodePath(node)
    path.setColor(*color)
    path.setLightOff()
    path.setTwoSided(True)
    return path


class Widgets:
    """All sizes are pixels; widgets share the dashboard's lifetime list."""
    def __init__(self, parent, nodes: list, font) -> None:
        self.parent, self.nodes, self.font = parent, nodes, font

    def label(self, text: str, x: float, y: float, size: int = 16,
              color: tuple = INK, parent=None, width: float | None = None,
              align=TextNode.ALeft):
        options = {} if width is None else {'wordwrap': width/size}
        node = OnscreenText(parent=parent if parent is not None else self.parent,
            text=text, pos=(x, -y), scale=size, font=self.font,
            align=align, fg=color, mayChange=True, **options)
        self.nodes.append(node)
        return node

    def panel(self, x: float, y: float, width: float, height: float,
              color: tuple = PANEL, parent=None, radius: float = 10):
        node = rounded(width, height, color, radius)
        node.reparentTo(parent if parent is not None else self.parent)
        node.setPos(x, 0, -y-height/2)
        self.nodes.append(node)
        return node

    def button(self, text: str, x: float, y: float, width: float,
               command: Callable, height: float = 44, size: int = 16,
               color: tuple = BUTTON, parent=None):
        colors = (color, tuple(min(1, c*1.35) for c in color[:3])+(1,),
                  tuple(c*.8 for c in color[:3])+(1,), BUTTON)
        node = DirectButton(parent=parent if parent is not None else self.parent,
            pos=(x, 0, -y), text=text, text_font=self.font, text_scale=size,
            text_pos=(width/2, -size*.32), text_fg=INK, command=command,
            frameSize=(0, width, -height/2, height/2), relief=None,
            geom=tuple(rounded(width, height, c) for c in colors))
        self.nodes.append(node)
        return node

    def check(self, text: str, x: float, y: float, width: float,
              checked: bool, command: Callable, parent=None, height: float = 30):
        node = DirectCheckButton(parent=parent if parent is not None else self.parent,
            pos=(x, 0, -y), text=text, text_font=self.font, text_scale=16,
            text_align=TextNode.ALeft, text_pos=(38, -5), text_fg=INK,
            frameSize=(0, width, -height/2, height/2), indicatorValue=int(checked),
            command=command, relief=None, boxRelief=None)
        indicator = node.component('indicator')
        tick = LineSegs('checkbox-tick')
        tick.setThickness(2.5)
        tick.setColor(*INK)
        tick.moveTo(-5, 0, 0)
        tick.drawTo(-1, 0, -4)
        tick.drawTo(6, 0, 5)
        indicator['text'] = ('', '')
        indicator['geom'] = (NodePath('unchecked'), NodePath(tick.create()))
        indicator['frameSize'] = (-10, 10, -10, 10)
        indicator['frameColor'] = ((.09, .13, .19, 1), (.05, .45, .51, 1))
        indicator['relief'] = DGG.FLAT
        indicator.setPos(13, 0, 0)
        node.setIndicatorValue()
        self.nodes.append(node)
        return node
