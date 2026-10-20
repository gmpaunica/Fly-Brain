"""Pixel layout is the single source for drawing, camera bounds and mouse routing."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    """Screen pixels; origin top left, x rightward and y downward."""
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


class DashboardLayout:
    """Two scene cards and a 320px sidebar at the supported desktop sizes."""
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.sidebar = Rect(width-340, 148, width-20, height-20)
        scene_right = self.sidebar.left - 16
        card_width = (scene_right - 20 - 16) / 2
        self.cards = {
            'world': Rect(20, 148, 20+card_width, height-54),
            'brain': Rect(36+card_width, 148, scene_right, height-54),
        }
        self.viewports = {name: Rect(r.left+1, r.top+48, r.right-1, r.bottom-46)
                          for name, r in self.cards.items()}
        self.rectangles = {name: self.normalized_rect(r) for name, r in self.viewports.items()}
        self.rectangles.update(details=self.normalized_rect(self.sidebar),
                               help=self.normalized_rect(self.sidebar))
        self.row_height = min(38, max(30, (height-400)/10))
        self.group_rows = [(self.sidebar.left+16, 260+i*self.row_height) for i in range(10)]
        self.outline_y = self.group_rows[-1][1] + self.row_height + 8
        self.cord_y = self.outline_y + 30
        self.description_top = self.cord_y + 16

    def normalized(self, x: float, y: float) -> tuple[float, float]:
        return 2*x/self.width-1, 1-2*y/self.height

    def normalized_rect(self, rect: Rect) -> tuple[float, float, float, float]:
        left, top = self.normalized(rect.left, rect.top)
        right, bottom = self.normalized(rect.right, rect.bottom)
        return left, right, bottom, top

    def route(self, x: float, y: float) -> str | None:
        for name, (left, right, bottom, top) in self.rectangles.items():
            if left <= x <= right and bottom <= y <= top:
                return name
        return None
