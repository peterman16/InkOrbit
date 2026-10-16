"""Automatic label placement.

Every label gets a ranked set of candidate positions around its anchor (the
cartographer's order: upper right first) at increasing distances. Each
candidate is scored by how much ink it would cover, using an occupancy grid
of everything already on the page; the cheapest wins and is added to the
grid so later labels avoid it too. Far placements get a thin leader line.
"""

from __future__ import annotations

import numpy as np

from .patterns import points_in_polygon

CELL_MM = 0.25
CLEARANCE_MM = 0.5
DESCENDER = 0.3  # descender depth as a fraction of cap height

_DIRECTIONS = {
    "ne": (1, 1),
    "se": (1, -1),
    "nw": (-1, 1),
    "sw": (-1, -1),
    "e": (1, 0),
    "n": (0, 1),
    "w": (-1, 0),
    "s": (0, -1),
}
_GAPS_MM = (1.0, 2.5, 5.0, 8.0)
_LEADER_FROM_GAP_MM = 5.0


# How much a label minds sitting on each kind of ink.
LIGHT = 1  # grid lines: labels may sit on them, as on a printed map
INK = 10  # lines, symbols, fills
TEXT = 50  # other labels: never overlap


class Occupancy:
    """A raster of the page recording how much it costs to letter over each cell."""

    def __init__(self, width_mm: float, height_mm: float, cell_mm: float = CELL_MM):
        self.cell = cell_mm
        self.width, self.height = width_mm, height_mm
        shape = (int(np.ceil(height_mm / cell_mm)) + 1, int(np.ceil(width_mm / cell_mm)) + 1)
        self.grid = np.zeros(shape, np.uint8)

    def _index(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.floor(points[:, 0] / self.cell).astype(int)
        iy = np.floor(points[:, 1] / self.cell).astype(int)
        return ix, iy

    def stamp_points(self, points: np.ndarray, radius_mm: float, weight: int = INK) -> None:
        if not len(points):
            return
        ix, iy = self._index(np.asarray(points, float))
        r = int(np.ceil(radius_mm / self.cell))
        offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= r * r]
        ny, nx = self.grid.shape
        for dx, dy in offsets:
            x, y = ix + dx, iy + dy
            ok = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
            cells = (y[ok], x[ok])
            self.grid[cells] = np.maximum(self.grid[cells], weight)

    def stamp_polyline(self, points: np.ndarray, radius_mm: float, weight: int = INK) -> None:
        self.stamp_points(densify(points, self.cell), radius_mm, weight)

    def stamp_polygon(self, polygon: np.ndarray, clearance_mm: float) -> None:
        self.stamp_polyline(np.vstack([polygon, polygon[:1]]), clearance_mm)
        lo = np.floor(polygon.min(axis=0) / self.cell).astype(int)
        hi = np.ceil(polygon.max(axis=0) / self.cell).astype(int)
        xs, ys = np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1)
        centers = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        inside = points_in_polygon((centers + 0.5) * self.cell, polygon)
        self.stamp_points((centers[inside] + 0.5) * self.cell, 0.0)

    def _box_slice(self, box):
        x0, y0, x1, y1 = box
        return (
            slice(max(int(np.floor(y0 / self.cell)), 0), max(int(np.ceil(y1 / self.cell)), 0)),
            slice(max(int(np.floor(x0 / self.cell)), 0), max(int(np.ceil(x1 / self.cell)), 0)),
        )

    def stamp_box(self, box, weight: int = TEXT) -> None:
        cells = self._box_slice(box)
        self.grid[cells] = np.maximum(self.grid[cells], weight)

    def line_cost(self, points: np.ndarray) -> float:
        pts = densify(points, self.cell)
        ix, iy = self._index(pts)
        ny, nx = self.grid.shape
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        return float(self.grid[iy[ok], ix[ok]].sum(dtype=np.int64))

    def cost(self, box) -> float:
        x0, y0, x1, y1 = box
        if x0 < 0 or y0 < 0 or x1 > self.width or y1 > self.height:
            return np.inf
        return float(self.grid[self._box_slice(box)].sum(dtype=np.int64))


def densify(points: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline so consecutive points are at most ``step`` apart."""
    pts = np.asarray(points, float)
    seg = np.diff(pts, axis=0)
    n = np.maximum(np.ceil(np.hypot(seg[:, 0], seg[:, 1]) / step).astype(int), 1)
    out = [pts[i] + np.outer(np.arange(k) / k, seg[i]) for i, k in enumerate(n)]
    return np.vstack([*out, pts[-1:]])


def candidates(anchor: np.ndarray, width: float, height: float, prefer: str | None = None):
    """Yield (rank, gap, box, baseline_origin) for each candidate position."""
    order = list(_DIRECTIONS)
    if prefer is not None:
        if prefer not in _DIRECTIONS:
            raise ValueError(f"unknown label direction {prefer!r}; use one of {sorted(_DIRECTIONS)}")
        order.remove(prefer)
        order.insert(0, prefer)
    depth = DESCENDER * height
    total_h = height + depth
    for gap in _GAPS_MM:
        for rank, name in enumerate(order):
            dx, dy = _DIRECTIONS[name]
            # Diagonals sit a little closer so the label still reads as attached.
            g = gap * (0.75 if dx and dy else 1.0)
            x0 = anchor[0] + (g if dx > 0 else -width - g if dx < 0 else -width / 2)
            y0 = anchor[1] + (g if dy > 0 else -total_h - g if dy < 0 else -total_h / 2)
            yield rank, gap, (x0, y0, x0 + width, y0 + total_h), np.array([x0, y0 + depth])


def place(occupancy: Occupancy, anchors, width: float, height: float, prefer: str | None = None):
    """Pick the best candidate over all anchors.

    Returns (anchor, box, baseline_origin, needs_leader).
    """
    best = None
    for k, anchor in enumerate(anchors):
        for rank, gap, box, origin in candidates(anchor, width, height, prefer):
            score = occupancy.cost(box) + rank + 4.0 * _GAPS_MM.index(gap) + 3.0 * k
            if gap >= _LEADER_FROM_GAP_MM and np.isfinite(score):
                # The leader must not cut through other labels either.
                score += occupancy.line_cost(leader(anchor, box))
            if best is None or score < best[0]:
                best = (score, anchor, box, origin, gap)
    _, anchor, box, origin, gap = best
    return anchor, box, origin, gap >= _LEADER_FROM_GAP_MM


def leader(anchor: np.ndarray, box) -> np.ndarray:
    """A short line from near the anchor to near the closest point of the label box."""
    x0, y0, x1, y1 = box
    target = np.array([np.clip(anchor[0], x0, x1), np.clip(anchor[1], y0, y1)])
    d = target - anchor
    length = np.hypot(*d)
    u = d / length
    return np.array([anchor + 0.8 * u, target - 0.6 * u])
