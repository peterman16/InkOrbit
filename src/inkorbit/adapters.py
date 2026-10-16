"""Turn other libraries' objects into what inkorbit draws.

Everything here is duck-typed: inkorbit never imports poliastro, hapsira or
astropy itself, so it works with whichever of them (or none) is installed.
Skyfield is imported only to convert positions to latitude/longitude.

Lengths come out in kilometres and angles in radians unless stated.
"""

from __future__ import annotations

import numpy as np

from .bodies import BODIES, Body, get_body
from .elements import Elements


def _is_quantity(x) -> bool:
    return hasattr(x, "to_value") and hasattr(x, "unit")


def length_km(x) -> float | np.ndarray:
    """A plain number (taken as km) or an astropy Quantity of length."""
    if _is_quantity(x):
        return x.to_value("km")
    return np.asarray(x, dtype=float) if np.ndim(x) else float(x)


def angle_rad(x) -> float:
    """A plain number (taken as radians) or an astropy angle Quantity / Skyfield Angle."""
    if _is_quantity(x):
        return float(x.to_value("rad"))
    if hasattr(x, "radians"):  # skyfield Angle
        return float(x.radians)
    return float(x)


def _is_twobody_orbit(obj) -> bool:
    """poliastro and hapsira Orbit objects share this interface."""
    return all(hasattr(obj, name) for name in ("a", "ecc", "inc", "raan", "argp", "nu", "attractor"))


def as_elements(obj) -> Elements:
    """Elements from :class:`Elements` or a poliastro / hapsira ``Orbit`` (lengths in km)."""
    if isinstance(obj, Elements):
        return obj
    if _is_twobody_orbit(obj):
        ecc = obj.ecc.to_value("") if _is_quantity(obj.ecc) else obj.ecc
        return Elements(
            a=float(length_km(obj.a)),
            e=float(ecc),
            i=angle_rad(obj.inc),
            raan=angle_rad(obj.raan),
            argp=angle_rad(obj.argp),
            nu=angle_rad(obj.nu),
        )
    raise TypeError(f"cannot read orbital elements from {type(obj).__name__}; pass inkorbit.Elements or an Orbit")


def body_of(obj) -> Body | None:
    """The central body of a poliastro / hapsira ``Orbit``, if it has one."""
    attractor = getattr(obj, "attractor", None)
    if attractor is None:
        return None
    name = str(getattr(attractor, "name", "")).lower()
    known = BODIES.get(name)
    radius = float(length_km(attractor.R)) if hasattr(attractor, "R") else (known.radius_km if known else np.nan)
    mu = float(attractor.k.to_value("km3 / s2")) if hasattr(attractor, "k") else (known.mu_km3_s2 if known else np.nan)
    names = known.apsis_names if known else ("periapsis", "apoapsis")
    return Body(str(getattr(attractor, "name", "body")), radius, mu, names)


def resolve_body(body, orbits=()) -> Body:
    """An explicit body wins; otherwise the first orbit that knows its attractor; else Earth."""
    if body is not None:
        return get_body(body)
    for orbit in orbits:
        found = body_of(orbit)
        if found is not None:
            return found
    return get_body("earth")


def as_positions(obj) -> np.ndarray:
    """(N, 2) or (N, 3) positions in km from arrays, astropy Quantities/representations or Skyfield positions."""
    if hasattr(obj, "position") and hasattr(obj.position, "km"):  # skyfield ICRF / Geocentric
        pts = np.asarray(obj.position.km, dtype=float)
        return pts.T if pts.ndim == 2 and pts.shape[0] == 3 else pts
    if hasattr(obj, "xyz"):  # astropy CartesianRepresentation
        return np.asarray(length_km(obj.xyz), dtype=float).T
    if _is_twobody_orbit(obj):
        raise TypeError("an Orbit has elements, not sampled positions; pass it as the orbit itself")
    pts = np.asarray(length_km(obj), dtype=float)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        if pts.ndim == 2 and pts.shape[0] in (2, 3):
            return pts.T
        raise ValueError(f"expected positions shaped (N, 2) or (N, 3), got {pts.shape}")
    return pts


def as_lat_lon(obj, lon=None) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude in degrees from arrays, Quantities or a Skyfield geocentric position."""
    if lon is not None:
        lat = obj
        lat = lat.to_value("deg") if _is_quantity(lat) else np.asarray(lat, dtype=float)
        lon = lon.to_value("deg") if _is_quantity(lon) else np.asarray(lon, dtype=float)
        return np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if hasattr(obj, "position") and hasattr(obj, "center"):
        from skyfield.api import wgs84

        lat, lon = wgs84.latlon_of(obj)
        return np.asarray(lat.degrees, dtype=float), np.asarray(lon.degrees, dtype=float)
    raise TypeError("pass latitude and longitude arrays (degrees), or a Skyfield geocentric position")


def is_orbit_like(obj) -> bool:
    return isinstance(obj, Elements) or _is_twobody_orbit(obj)
