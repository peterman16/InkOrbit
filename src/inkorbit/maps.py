"""Map data and geometry on an equirectangular (longitude, latitude) map.

Coastlines and land come from Natural Earth 1:110m, which is in the public
domain ("Made with Natural Earth").
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np

from . import geometry


@lru_cache(maxsize=1)
def _earth() -> dict[str, np.ndarray]:
    with resources.files("inkorbit").joinpath("data/earth_110m.npz").open("rb") as f:
        data = np.load(f)
        return {key: data[key].astype(float) for key in data.files}


def _unpack(points: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [points[int(a) : int(b)] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


def coastlines() -> list[np.ndarray]:
    """Coastline polylines as (N, 2) arrays of (longitude, latitude) in degrees."""
    data = _earth()
    return _unpack(data["coastline_points"], data["coastline_offsets"])


def land() -> list[np.ndarray]:
    """Land outlines (outer rings) as (N, 2) arrays of (longitude, latitude) in degrees."""
    data = _earth()
    return _unpack(data["land_points"], data["land_offsets"])


def wrap_longitude(lon) -> np.ndarray:
    """Longitudes into [-180, 180)."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def split_at_antimeridian(lon, lat, jump: float = 180.0) -> list[np.ndarray]:
    """Split a track into (lon, lat) runs wherever it crosses the map's left/right edge.

    The crossing point is interpolated so each run reaches the border instead
    of stopping short of it.
    """
    lon = wrap_longitude(lon)
    lat = np.asarray(lat, dtype=float)
    runs: list[np.ndarray] = []
    current = [(lon[0], lat[0])]
    for k in range(1, len(lon)):
        l0, l1 = lon[k - 1], lon[k]
        if abs(l1 - l0) > jump:
            edge = 180.0 if l0 > l1 else -180.0
            l1_unwrapped = l1 + 360.0 if l0 > l1 else l1 - 360.0
            t = (edge - l0) / (l1_unwrapped - l0)
            y = lat[k - 1] + t * (lat[k] - lat[k - 1])
            current.append((edge, y))
            runs.append(np.array(current))
            current = [(-edge, y)]
        current.append((l1, lat[k]))
    runs.append(np.array(current))
    return [r for r in runs if len(r) >= 2]


def small_circle(lat_deg: float, lon_deg: float, radius_deg: float, n: int = 121) -> list[np.ndarray]:
    """A circle of given angular radius around a point on the sphere, as map polylines."""
    lat, lon = geometry.destination(lat_deg, lon_deg, np.linspace(0.0, 360.0, n), radius_deg)
    return split_at_antimeridian(lon, lat)
