"""Surface features per body: (lon, lat) polylines in degrees, east longitude in [-180, 180)."""

from __future__ import annotations

import numpy as np

from .bodies import Body, get_body
from .maps import coastlines

# Dark cloud belts as (south, north) planetographic latitudes in degrees. Stylised and approximate, not data.
_BELTS = {
    "jupiter": ((-90, -45), (-30, -26), (-20, -7), (7, 17), (24, 28), (45, 90)),
    "saturn": ((-65, -50), (-35, -20), (20, 35), (50, 65)),
}


def belts(body: str | Body) -> list[tuple[float, float]]:
    """Latitude ranges (south, north) in degrees of ``body``'s dark cloud belts; empty if none."""
    return list(_BELTS.get(get_body(body).name.lower(), ()))


def _band(lat: float) -> np.ndarray:
    return np.column_stack([np.linspace(-180, 180, 361), np.full(361, float(lat))])


def _great_red_spot() -> np.ndarray:
    # Stylised: centred at lat -23, lon 0 (the real spot's longitude drifts); semi-axes 7 deg lon, 3 deg lat,
    # so it sits between the -20 and -26 band edges.
    t = np.linspace(0, 2 * np.pi, 73)
    return np.column_stack([7 * np.cos(t), -23 + 3 * np.sin(t)])


def features(body: str | Body) -> list[np.ndarray]:
    """Surface polylines of ``body`` as (N, 2) arrays of (longitude, latitude); empty if none are known."""
    body = get_body(body)
    if body.name == "Earth":
        return coastlines()
    edges = sorted({e for belt in belts(body) for e in belt if abs(e) < 90})  # belt edges, minus the poles
    bands = [_band(lat) for lat in edges]
    return bands + [_great_red_spot()] if body.name == "Jupiter" else bands
