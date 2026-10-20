"""Display coordinates and camera interpolation, with no simulation imports."""
import math
import numpy as np


class PanelLayout:
    """Panel rectangles in window-normalized (-1..1) coordinates."""
    rectangles={'world':(-.97,-.03,-.38,.64),'brain':(.03,.97,-.38,.64),
                'help':(-.97,-.03,-.84,-.46),'details':(.03,.97,-.84,-.46)}

    def route(self,x: float,y: float) -> str | None:
        for name,(left,right,bottom,top) in self.rectangles.items():
            if left<=x<=right and bottom<=y<=top: return name
        return None


class CameraMotion:
    """Bounded zoom target and pan offset animated using display time only."""
    def __init__(self,distance: float,minimum: float,maximum: float):
        self.minimum=minimum; self.maximum=maximum
        self.reset(distance)

    def reset(self,distance: float) -> None:
        self.distance=self.target=float(np.clip(distance,self.minimum,self.maximum))
        self.offset=np.zeros(3)

    def zoom(self,factor: float) -> None:
        self.target=float(np.clip(self.target*factor,self.minimum,self.maximum))

    def advance(self,dt: float) -> None:
        self.distance+=(self.target-self.distance)*(-math.expm1(-max(0,dt)/.12))

    def pan(self,delta: np.ndarray) -> None:
        self.offset+=delta
        norm=float(np.linalg.norm(self.offset))
        if norm>.25: self.offset*=.25/norm
