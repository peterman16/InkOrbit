"""Fill patterns (stipple and hatching) computed in page millimetres."""

from __future__ import annotations

import numpy as np


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test, vectorised over points."""
    px, py = points[:, 0:1], points[:, 1:2]
    x0, y0 = polygon[:, 0], polygon[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0 > py) != (y1 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    return np.count_nonzero(crosses & (px < x_at), axis=1) % 2 == 1


def stipple(polygon: np.ndarray, spacing: float, seed: int = 0) -> np.ndarray:
    """Hand-stipple-like dots: a jittered grid clipped to the polygon."""
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    xs = np.arange(lo[0], hi[0] + spacing, spacing)
    ys = np.arange(lo[1], hi[1] + spacing, spacing)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    grid = grid + rng.uniform(-0.5, 0.5, grid.shape) * spacing
    return grid[points_in_polygon(grid, polygon)]


def hatch(polygon: np.ndarray, spacing: float, angle_deg: float) -> list[np.ndarray]:
    """Parallel hatch segments clipped to the polygon (even-odd)."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    to_local = np.array([[c, s], [-s, c]])  # rotates by -angle: hatch lines become horizontal
    local = polygon @ to_local.T
    x0, y0 = local[:, 0], local[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    segments = []
    lo, hi = y0.min(), y0.max()
    for y in np.arange(lo + spacing / 2, hi, spacing):
        crosses = (y0 > y) != (y1 > y)
        xs = np.sort(x0[crosses] + (y - y0[crosses]) * (x1[crosses] - x0[crosses]) / (y1[crosses] - y0[crosses]))
        for xa, xb in zip(xs[::2], xs[1::2], strict=True):
            segments.append(np.array([[xa, y], [xb, y]]) @ to_local)
    return segments
