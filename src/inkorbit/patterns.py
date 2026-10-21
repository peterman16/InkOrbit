"""Fill patterns (stipple and hatching) computed in page millimetres."""

from __future__ import annotations

import numpy as np

from .geometry import points_in_polygon

# A fill too small or thin for a few dots at its spacing (a little crater shadow) is stippled finer instead,
# by this factor at a time, down to this spacing: it stays dots, never turns solid.
_MIN_DOTS, _FINER, _FINEST_MM = 5, 0.75, 0.15


def stipple(polygon: np.ndarray, spacing: float, seed: int = 0) -> np.ndarray:
    """Hand-stipple-like dots: a jittered grid clipped to the polygon, finer if it would hold fewer than 5 dots."""
    dots = _jittered(polygon, spacing, seed)
    while len(dots) < _MIN_DOTS and spacing > _FINEST_MM:
        spacing = max(spacing * _FINER, _FINEST_MM)
        dots = _jittered(polygon, spacing, seed)
    return dots


def _jittered(polygon: np.ndarray, spacing: float, seed: int) -> np.ndarray:
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
