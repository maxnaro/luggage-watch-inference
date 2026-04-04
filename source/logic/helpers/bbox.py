from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pyds import NvOSD_RectParams


@dataclass(frozen=True)
class BBox:
    """Bounding box represented by its top-left corner (x, y) and its width and height."""

    x: float
    y: float
    width: float
    height: float

    def __init__(self, rect_params: NvOSD_RectParams):
        object.__setattr__(self, "x", rect_params.left)
        object.__setattr__(self, "y", rect_params.top)
        object.__setattr__(self, "width", rect_params.width)
        object.__setattr__(self, "height", rect_params.height)

    @property
    def centre(self) -> tuple[float, float]:
        """Returns the centre point of the bounding box as (x_centre, y_centre)."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def left(self) -> float:
        return self.x

    @property
    def top(self) -> float:
        return self.y

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def distance_to(self, other: BBox) -> float:
        """Calculates the Euclidean distance between the centres of this bounding box and another."""
        x1, y1 = self.centre
        x2, y2 = other.centre
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def centre_to_box_distance(self, other: BBox) -> float:
        """Shortest distance from this box centre to the other box area."""
        x, y = self.centre
        nearest_x = min(max(x, other.left), other.right)
        nearest_y = min(max(y, other.top), other.bottom)
        return sqrt((nearest_x - x) ** 2 + (nearest_y - y) ** 2)
