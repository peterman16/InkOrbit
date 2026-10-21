"""Surface features per body: (lon, lat) polylines in degrees, east longitude in [-180, 180)."""

from __future__ import annotations

import zlib
from functools import lru_cache
from importlib import resources

import numpy as np

from .bodies import Body, get_body
from .maps import coastlines, small_circle

# USGS Gazetteer features are drawn as rings (centre and diameter). Thousands of rings turn to mud, so keep
# only those whose diameter is between these fractions of the body's, and the largest _MAX_RINGS of them.
# Bigger than half the body, a ring is a circle across the disc and no longer reads as a place.
_MIN_FRACTION, _MAX_FRACTION = 0.03, 0.5
_MAX_RINGS = 100
_GAZETTEER = ("moon", "mercury", "venus", "mars")

# Dark cloud belts as (south, north) planetographic latitudes in degrees. Stylised and approximate, not data.
_BELTS = {
    "jupiter": ((-90, -45), (-30, -26), (-20, -7), (7, 17), (24, 28), (45, 90)),
    "saturn": ((-65, -50), (-35, -20), (20, 35), (50, 65)),
}


def belts(body: str | Body) -> list[tuple[float, float]]:
    """Latitude ranges (south, north) in degrees of ``body``'s dark cloud belts; empty if none."""
    return list(_BELTS.get(get_body(body).name.lower(), ()))


def edge(body: str | Body, lat: float, lon: np.ndarray) -> np.ndarray:
    """Latitude of a belt edge (degrees) at longitudes ``lon`` (degrees): ``lat`` plus a few fixed sine waves.

    Stylised: the phases come from a seed per body and edge, so every render is the same. Jupiter's edges
    wave about twice as much as Saturn's, the poles stay put, and the two edges that frame the Great Red
    Spot flatten near longitude 0 to keep it clear.
    """
    name = get_body(body).name.lower()
    if abs(lat) >= 90:
        return np.full(np.shape(lon), float(lat))
    rng = np.random.default_rng(zlib.crc32(f"{name}{lat}".encode()))
    k = rng.choice(np.arange(3, 13), size=rng.integers(2, 5), replace=False)
    amp = rng.uniform(0.6, 1.2) * (1.0 if name == "jupiter" else 0.5) / len(k)
    x = np.radians(lon)
    wave = sum(amp * np.sin(kk * x + rng.uniform(0, 2 * np.pi)) for kk in k)
    if name == "jupiter" and lat in (-26, -20):
        wave = wave * (1 - np.exp(-((np.asarray(lon) / 10) ** 2)))
    return lat + wave


def _great_red_spot() -> np.ndarray:
    # Stylised: centred at lat -23, lon 0 (the real spot's longitude drifts); semi-axes 7 deg lon, 3 deg lat,
    # so it sits between the -20 and -26 band edges.
    t = np.linspace(0, 2 * np.pi, 73)
    return np.column_stack([7 * np.cos(t), -23 + 3 * np.sin(t)])


@lru_cache(maxsize=8)
def _rings(body: Body) -> list[np.ndarray]:
    with resources.files("inkorbit").joinpath(f"data/{body.name.lower()}_surface.npz").open("rb") as f:
        data = np.load(f)
        lat, lon, diameter = (data[k].astype(float) for k in ("lat", "lon", "diameter_km"))
    fraction = diameter / (2 * body.radius_km)
    big = np.argsort(-diameter)
    big = big[(fraction[big] >= _MIN_FRACTION) & (fraction[big] <= _MAX_FRACTION)][:_MAX_RINGS]
    radius = np.degrees(diameter[big] / 2 / body.radius_km)
    return [run for k, r in zip(big, radius, strict=True) for run in small_circle(lat[k], lon[k], r)]


def features(body: str | Body) -> list[np.ndarray]:
    """Surface polylines of ``body`` as (N, 2) arrays of (longitude, latitude); empty if none are known."""
    body = get_body(body)
    if body.name == "Earth":
        return coastlines()
    if body.name.lower() in _GAZETTEER:
        return _rings(body)
    edges = sorted({e for belt in belts(body) for e in belt if abs(e) < 90})  # belt edges, minus the poles
    lon = np.linspace(-180, 180, 361)
    bands = [np.column_stack([lon, edge(body, lat, lon)]) for lat in edges]
    return bands + [_great_red_spot()] if body.name == "Jupiter" else bands
