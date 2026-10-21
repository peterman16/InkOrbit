"""Surface features per body: (lon, lat) polylines in degrees, east longitude in [-180, 180)."""

from __future__ import annotations

import zlib
from functools import lru_cache
from importlib import resources

import numpy as np

from . import geometry
from .bodies import Body, get_body
from .maps import coastlines, small_circle

# USGS Gazetteer features are rings (centre and diameter), by kind: 0 crater, 1 mare, 2 mons, 3 planitia,
# 4 terra, 5 regio. Craters and mountains stay outlines; the broad kinds are areas, which draw stipples them.
# Thousands of rings turn to mud, so keep those whose diameter is between these fractions of the body's,
# and the largest _MAX_RINGS. Bigger than half (lines) or 60% (areas), a ring is a circle across the disc.
_LINE_KINDS, _AREA_KINDS = (0, 2), (1, 3, 4, 5)
_LINE_FRACTION, _AREA_FRACTION = (0.03, 0.5), (0.03, 0.6)
_MAX_RINGS = 100
_CRATER_FRACTION = (0.02, 0.15)
_RIM_HARMONICS, _RIM_AMPLITUDE = (2, 3, 5, 7), (0.01, 0.035)  # of the radius: slightly polygonal, hand-drawn rims
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


@lru_cache(maxsize=16)
def _caps(body: Body, kinds: tuple[int, ...], fraction: tuple[float, float]) -> list[tuple[float, float, float]]:
    """(lat, lon, angular radius) in degrees of the largest gazetteer features of these kinds and sizes."""
    with resources.files("inkorbit").joinpath(f"data/{body.name.lower()}_surface.npz").open("rb") as f:
        data = np.load(f)
        lat, lon, diameter, kind = (data[k].astype(float) for k in ("lat", "lon", "diameter_km", "kind"))
    size = diameter / (2 * body.radius_km)
    keep = np.argsort(-diameter)
    keep = keep[np.isin(kind[keep], kinds) & (size[keep] >= fraction[0]) & (size[keep] <= fraction[1])][:_MAX_RINGS]
    return [(lat[k], lon[k], np.degrees(diameter[k] / 2 / body.radius_km)) for k in keep]


def areas(body: str | Body) -> list[tuple[float, float, float]]:
    """Broad surface areas (maria, plains, highlands) as (lat, lon, angular radius) caps in degrees."""
    body = get_body(body)
    return _caps(body, _AREA_KINDS, _AREA_FRACTION) if body.name.lower() in _GAZETTEER else []


def craters(body: str | Body) -> list[tuple[float, float, float, bool]]:
    """Craters and mountains as (lat, lon, angular radius, is_mons) in degrees, largest first; empty if none."""
    body = get_body(body)
    if body.name.lower() not in _GAZETTEER:
        return []
    found = [(*c, mons) for mons in (False, True) for c in _caps(body, (2 if mons else 0,), _CRATER_FRACTION)]
    return sorted(found, key=lambda c: -c[2])[:_MAX_RINGS]


def rim(body: str | Body, i: int, lat: float, lon: float, r: float) -> np.ndarray:
    """Hand-drawn rim of feature ``i`` of ``body``: a closed (N, 2) ring of (lon, lat) degrees, cap radius ``r``.

    The radius gets a few low harmonics with seeded phases, so every render is the same.
    """
    rng = np.random.default_rng(zlib.crc32(f"{get_body(body).name}{i}".encode()))
    t = np.linspace(0.0, 360.0, 181)
    rr = np.full_like(t, r)
    for k in _RIM_HARMONICS:
        rr += r * rng.uniform(*_RIM_AMPLITUDE) * np.cos(k * np.radians(t) + rng.uniform(0, 2 * np.pi))
    return np.column_stack(geometry.destination(lat, lon, t, rr)[::-1])


def shadow(lat: float, lon: float, r: float, mons: bool = False) -> np.ndarray:
    """Crescent of shadow as an (N, 2) ring of (lon, lat) degrees, for light from the west.

    A crater is shaded inside its west wall; a mountain outside its east flank. The outer edge is half the
    rim, the other a half-ellipse of the same north-south extent (0.35 r or 1.25 r east-west).
    """
    t = np.linspace(-np.pi / 2, np.pi / 2, 40) + (0.0 if mons else np.pi)  # east half, or west half
    x, y = np.cos(t), np.sin(t)
    x = np.concatenate([r * x, (1.25 if mons else 0.35) * r * x[::-1]])
    y = r * np.concatenate([y, y[::-1]])
    # local tangent offsets (x east, y north) as bearing and distance
    return np.column_stack(geometry.destination(lat, lon, np.degrees(np.arctan2(x, y)), np.hypot(x, y))[::-1])


def features(body: str | Body) -> list[np.ndarray]:
    """Surface polylines of ``body`` as (N, 2) arrays of (longitude, latitude); empty if none are known."""
    body = get_body(body)
    if body.name == "Earth":
        return coastlines()
    if body.name.lower() in _GAZETTEER:
        return [run for cap in _caps(body, _LINE_KINDS, _LINE_FRACTION) for run in small_circle(*cap)]
    edges = sorted({e for belt in belts(body) for e in belt if abs(e) < 90})  # belt edges, minus the poles
    lon = np.linspace(-180, 180, 361)
    bands = [np.column_stack([lon, edge(body, lat, lon)]) for lat in edges]
    return bands + [_great_red_spot()] if body.name == "Jupiter" else bands
