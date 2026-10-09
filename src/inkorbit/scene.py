"""A backend-neutral description of a figure.

Geometry lives in *data* coordinates (kilometres, Earth radii, degrees...),
with x and y on the same scale. Anything with a physical size on the page
(text height, arrowheads, markers, pen widths, fill density) is in millimetres
and is resolved by :mod:`inkorbit.layout` when the figure is put on a page.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from . import style as _style

ArrayLike2 = Sequence[float] | np.ndarray

HAlign = Literal["left", "center", "right"]
VAlign = Literal["baseline", "bottom", "center", "top"]
MarkerKind = Literal["dot", "open", "star", "cross"]
Pattern = Literal["stipple", "hatch", "solid"]


def _xy(value: ArrayLike2) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"expected an (x, y) pair, got shape {arr.shape}")
    return arr


def _polyline(points) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
        raise ValueError(f"expected an (N, 2) array with N >= 2, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class Line:
    points: np.ndarray
    role: str = _style.ORBIT
    closed: bool = False


@dataclass(frozen=True)
class Fill:
    polygon: np.ndarray
    pattern: Pattern = "stipple"
    spacing_mm: float | None = None
    angle_deg: float | None = None
    seed: int | None = None  # stipple randomness; defaults to the fill's position in the scene


@dataclass(frozen=True)
class Arrowhead:
    at: np.ndarray
    direction: np.ndarray
    size_mm: float | None = None


@dataclass(frozen=True)
class Marker:
    at: np.ndarray
    kind: MarkerKind = "dot"
    size_mm: float | None = None


@dataclass(frozen=True)
class Text:
    text: str
    at: np.ndarray
    height_mm: float | None = None
    ha: HAlign = "left"
    va: VAlign = "baseline"
    rotation_deg: float = 0.0
    offset_mm: tuple[float, float] = (0.0, 0.0)
    bold: bool = False


Item = Line | Fill | Arrowhead | Marker | Text


@dataclass
class Scene:
    """An ordered list of drawing items plus an optional caption.

    Items are drawn in the order they were added, so add fills first.
    """

    items: list[Item] = field(default_factory=list)
    caption: str | None = None
    limits: tuple[float, float, float, float] | None = None  # xmin, xmax, ymin, ymax

    def add(self, item: Item) -> Item:
        self.items.append(item)
        return item

    def line(
        self,
        points,
        role: str = _style.ORBIT,
        *,
        closed: bool = False,
        arrows: Sequence[float] = (),
        arrow_size_mm: float | None = None,
    ) -> Line:
        """Add a polyline. ``arrows`` places direction arrowheads at fractions of its length."""
        pts = _polyline(points)
        line = Line(pts, role, closed)
        self.add(line)
        for fraction in arrows:
            at, direction = point_along(pts, fraction)
            self.add(Arrowhead(at, direction, arrow_size_mm))
        return line

    def fill(self, polygon, pattern: Pattern = "stipple", *, spacing_mm=None, angle_deg=None, seed=None) -> Fill:
        return self.add(Fill(_polyline(polygon), pattern, spacing_mm, angle_deg, seed))

    def arrowhead(self, at: ArrayLike2, direction: ArrayLike2, size_mm: float | None = None) -> Arrowhead:
        return self.add(Arrowhead(_xy(at), _xy(direction), size_mm))

    def vector(
        self, start: ArrayLike2, end: ArrayLike2, role: str = _style.ORBIT, size_mm: float | None = None
    ) -> Line:
        """A straight line ending in an arrowhead at ``end``."""
        start, end = _xy(start), _xy(end)
        line = self.add(Line(np.array([start, end]), role))
        self.add(Arrowhead(end, end - start, size_mm))
        return line

    def marker(self, at: ArrayLike2, kind: MarkerKind = "dot", size_mm: float | None = None) -> Marker:
        return self.add(Marker(_xy(at), kind, size_mm))

    def text(
        self,
        text: str,
        at: ArrayLike2,
        *,
        height_mm: float | None = None,
        ha: HAlign = "left",
        va: VAlign = "baseline",
        rotation_deg: float = 0.0,
        offset_mm: tuple[float, float] = (0.0, 0.0),
        bold: bool = False,
    ) -> Text:
        return self.add(Text(text, _xy(at), height_mm, ha, va, rotation_deg, offset_mm, bold))

    def bounds(self) -> tuple[float, float, float, float]:
        """Data extent of everything that has a data position."""
        if self.limits is not None:
            return self.limits
        chunks = []
        for item in self.items:
            if isinstance(item, (Line,)):
                chunks.append(item.points)
            elif isinstance(item, Fill):
                chunks.append(item.polygon)
            else:
                chunks.append(item.at[None, :])
        if not chunks:
            return (-1.0, 1.0, -1.0, 1.0)
        pts = np.vstack(chunks)
        (xmin, ymin), (xmax, ymax) = pts.min(axis=0), pts.max(axis=0)
        return (float(xmin), float(xmax), float(ymin), float(ymax))

    def render(self, page="figure", style: _style.Style | None = None, renderer: str = "matplotlib"):
        from .render import render

        return render(self, page=page, style=style, renderer=renderer)

    def _repr_png_(self) -> bytes:
        return self.render().to_png(dpi=150)


def point_along(points: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Point and travel direction at a fraction of a polyline's length."""
    seg = np.diff(points, axis=0)
    lengths = np.hypot(seg[:, 0], seg[:, 1])
    total = lengths.sum()
    if total == 0:
        raise ValueError("polyline has zero length")
    target = np.clip(fraction, 0.0, 1.0) * total
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    i = int(np.clip(np.searchsorted(cumulative, target, side="right") - 1, 0, len(seg) - 1))
    t = (target - cumulative[i]) / lengths[i] if lengths[i] else 0.0
    return points[i] + t * seg[i], seg[i]
