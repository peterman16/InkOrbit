"""Plane geometry for drawing. No time propagation lives here."""

from __future__ import annotations

import numpy as np


def conic(p: float, e: float, nu: np.ndarray | None = None, n: int = 721) -> np.ndarray:
    """Points of a conic in the perifocal frame (periapsis along +x).

    ``p`` is the semi-latus rectum and ``nu`` the true anomalies in radians.
    By default a full ellipse is sampled; for e >= 1 pass the anomaly range.
    """
    if nu is None:
        if e >= 1:
            raise ValueError("open conics need an explicit range of true anomaly")
        nu = np.linspace(0.0, 2 * np.pi, n)
    nu = np.asarray(nu, dtype=float)
    r = p / (1 + e * np.cos(nu))
    return np.column_stack([r * np.cos(nu), r * np.sin(nu)])


def ellipse(a: float, e: float, n: int = 721) -> np.ndarray:
    """A closed Keplerian ellipse with the focus at the origin."""
    return conic(a * (1 - e**2), e, n=n)


def circle(center: tuple[float, float], radius: float, n: int = 361) -> np.ndarray:
    return arc(center, radius, 0.0, 2 * np.pi, n)


def arc(center: tuple[float, float], radius: float, start: float, stop: float, n: int = 181) -> np.ndarray:
    """A circular arc from angle ``start`` to ``stop`` (radians, counter-clockwise if stop > start)."""
    t = np.linspace(start, stop, n)
    return np.column_stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)])


def rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(points, dtype=float) @ np.array([[c, s], [-s, c]])


# --- Globes in orthographic view -------------------------------------------------
#
# View frame: x to the right, y up, z towards the viewer. A globe of radius R is
# centred on ``center``; only the hemisphere with z >= 0 is visible.


def view_rotation(tilt: float = 0.0, spin: float = 0.0) -> np.ndarray:
    """Body-fixed -> view rotation.

    With no rotation the north pole points up (+y) and longitude 0 faces the
    viewer. ``spin`` turns the globe about its axis (east comes towards the
    viewer for positive spin) and ``tilt`` leans the north pole towards the
    viewer, both in radians.
    """
    # Body frame: x through (lat 0, lon 0), y through (0, 90E), z north.
    to_view = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    cs, ss = np.cos(spin), np.sin(spin)
    spin_m = np.array([[cs, ss, 0.0], [-ss, cs, 0.0], [0.0, 0.0, 1.0]])
    ct, st = np.cos(tilt), np.sin(tilt)
    tilt_m = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    return tilt_m @ to_view @ spin_m


def lat_lon_to_view(lat: np.ndarray, lon: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Unit vectors in the view frame for latitudes/longitudes in radians."""
    lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
    body = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    return body @ rotation.T


def visible_runs(points3: np.ndarray) -> list[np.ndarray]:
    """Split a polyline of view-frame points into runs on the visible side (z >= 0)."""
    visible = points3[:, 2] >= 0
    runs, start = [], None
    for i, v in enumerate(visible):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > 1:
                runs.append(points3[start:i])
            start = None
    if start is not None and len(points3) - start > 1:
        runs.append(points3[start:])
    return runs


def graticule(
    rotation: np.ndarray, lat_step_deg: float = 30.0, lon_step_deg: float = 30.0, n: int = 241
) -> list[np.ndarray]:
    """Visible parts of the latitude/longitude grid of a unit globe, as 2D view polylines."""
    lines: list[np.ndarray] = []
    t = np.linspace(-np.pi, np.pi, n)
    for lat in np.arange(-90 + lat_step_deg, 90, lat_step_deg):
        lines += [r[:, :2] for r in visible_runs(lat_lon_to_view(np.radians(lat), t, rotation))]
    m = np.linspace(-np.pi / 2, np.pi / 2, n)
    for lon in np.arange(-180, 180, lon_step_deg):
        lines += [r[:, :2] for r in visible_runs(lat_lon_to_view(m, np.radians(lon), rotation))]
    return lines


def night_side(sun_view: np.ndarray, n: int = 181) -> np.ndarray | None:
    """Outline of the visible night side of a unit globe, as a closed 2D polygon.

    ``sun_view`` is the direction to the Sun in the view frame. Returns None
    when no night side is visible.
    """
    s = np.asarray(sun_view, dtype=float)
    s = s / np.linalg.norm(s)
    z = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(s, z)
    if np.linalg.norm(e1) < 1e-9:
        # Sun straight behind or in front of the viewer.
        return circle((0.0, 0.0), 1.0, n) if s[2] < 0 else None
    e1 /= np.linalg.norm(e1)
    w = np.cross(s, e1)
    if w[2] < 0:
        w = -w
    # Visible half of the terminator, from e1 through w to -e1.
    t = np.linspace(0.0, np.pi, n)
    terminator = np.outer(np.cos(t), e1) + np.outer(np.sin(t), w)
    # Night half of the limb, from -e1 back to e1, centred on the anti-sun side.
    a_start = np.arctan2(-e1[1], -e1[0])
    anti = np.arctan2(-s[1], -s[0]) if np.hypot(s[0], s[1]) > 1e-12 else a_start + np.pi / 2
    sweep = (anti - a_start) % (2 * np.pi)
    direction = 1.0 if sweep < np.pi else -1.0
    a = a_start + direction * np.linspace(0.0, np.pi, n)
    limb = np.column_stack([np.cos(a), np.sin(a)])
    return np.vstack([terminator[:, :2], limb[1:-1]])
