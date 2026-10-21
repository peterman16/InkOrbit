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


def project_to_plane(points: np.ndarray) -> np.ndarray:
    """Project 3D points onto their best-fit plane through the origin.

    The result is 2D, with +x towards the closest point (periapsis for an orbit
    around the origin) and the motion counter-clockwise, i.e. the perifocal view.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected an (N, 3) array, got shape {pts.shape}")
    normal = np.linalg.svd(pts, full_matrices=False)[2][-1]
    # Orient the normal along the angular momentum so the motion is counter-clockwise.
    h = np.cross(pts[:-1], np.diff(pts, axis=0)).sum(axis=0)
    if np.dot(normal, h) < 0:
        normal = -normal
    x = pts[np.argmin(np.linalg.norm(pts, axis=1))]
    x = x - np.dot(x, normal) * normal
    if np.linalg.norm(x) < 1e-12:
        raise ValueError("cannot orient the plane: closest point is at the origin")
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    return np.column_stack([pts @ x, pts @ y])


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
    ct, st = np.cos(tilt), np.sin(tilt)
    tilt_m = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    return tilt_m @ to_view @ _r3(spin)


def orbit_view_rotation(i: float, raan: float, argp: float, in_plane: float = 0.0) -> np.ndarray:
    """Body -> view rotation for a drawing that looks down the orbit normal.

    The body frame is inertial (longitude 0 at the vernal equinox), so the view
    is the perifocal frame R3(argp) R1(i) R3(raan), turned in the drawing by
    ``in_plane`` (e.g. raan + argp for the "plane" frame). Radians.
    """
    ci, si = np.cos(i), np.sin(i)
    r1 = np.array([[1.0, 0.0, 0.0], [0.0, ci, si], [0.0, -si, ci]])
    return _r3(argp - in_plane) @ r1 @ _r3(raan)


def _r3(angle: float) -> np.ndarray:
    """Frame rotation by ``angle`` about z (coordinates of the same vector in the turned frame)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


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


# --- Clipping to a rectangle -------------------------------------------------------


def clip_polyline(points: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float) -> list[np.ndarray]:
    """Clip a polyline to a rectangle, returning the runs that lie inside it."""
    pts = np.asarray(points, dtype=float)
    runs: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    for p0, p1 in zip(pts[:-1], pts[1:], strict=True):
        clipped = _clip_segment(p0, p1, xmin, xmax, ymin, ymax)
        if clipped is None:
            if current:
                runs.append(current)
                current = []
            continue
        a, b = clipped
        if current and not np.allclose(current[-1], a):
            runs.append(current)
            current = []
        if not current:
            current.append(a)
        current.append(b)
        if not np.allclose(b, p1):  # left the rectangle
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [np.array(r) for r in runs if len(r) >= 2]


def _clip_segment(p0, p1, xmin, xmax, ymin, ymax):
    """Liang-Barsky segment clipping; None if the segment is entirely outside."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], p0[0] - xmin), (d[0], xmax - p0[0]), (-d[1], p0[1] - ymin), (d[1], ymax - p0[1])):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return p0 + t0 * d, p0 + t1 * d


def clip_polygon(polygon: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray | None:
    """Sutherland-Hodgman clipping of a polygon to a rectangle; None if nothing is left."""
    out = np.asarray(polygon, dtype=float)
    for axis, bound, keep_below in ((0, xmin, False), (0, xmax, True), (1, ymin, False), (1, ymax, True)):
        if len(out) == 0:
            return None
        inside = out[:, axis] <= bound if keep_below else out[:, axis] >= bound
        result = []
        for k in range(len(out)):
            cur, prev = out[k], out[k - 1]
            cur_in, prev_in = inside[k], inside[k - 1]
            if cur_in != prev_in:
                t = (bound - prev[axis]) / (cur[axis] - prev[axis])
                result.append(prev + t * (cur - prev))
            if cur_in:
                result.append(cur)
        out = np.array(result)
    return out if len(out) >= 3 else None


def _signed_area(poly: np.ndarray) -> float:
    return 0.5 * float(np.sum(poly[:-1, 0] * poly[1:, 1] - poly[1:, 0] * poly[:-1, 1]))


def with_holes(outer: np.ndarray, holes: list[np.ndarray]) -> np.ndarray:
    """One ring filling ``outer`` minus ``holes`` under the even-odd rule, for a renderer that fills a single polygon.

    Each ring is closed, and after each hole the path returns to the start of ``outer``; the two bridge edges to
    and from a hole coincide and cancel. Holes must lie inside ``outer`` and must not overlap each other.
    """
    closed = [r if np.allclose(r[0], r[-1]) else np.vstack([r, r[:1]]) for r in (outer, *holes)]
    return np.vstack([closed[0], *[np.vstack([h, closed[0][:1]]) for h in closed[1:]]])


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test, vectorised over points."""
    px, py = points[:, 0:1], points[:, 1:2]
    x0, y0 = polygon[:, 0], polygon[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0 > py) != (y1 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    return np.count_nonzero(crosses & (px < x_at), axis=1) % 2 == 1


def _limb_pieces(points3: np.ndarray) -> list[np.ndarray]:
    """Visible runs of a closed loop of view points that crosses the limb, in 2D with their ends snapped to the limb."""
    pts = np.roll(points3, -int(np.argmax(points3[:, 2] < 0)), axis=0)  # start hidden, so no run wraps the seam
    pieces = []
    for run in visible_runs(np.vstack([pts, pts[:1]])):
        run = run[:, :2].copy()
        run[[0, -1]] /= np.linalg.norm(run[[0, -1]], axis=1, keepdims=True)
        pieces.append(run)
    return pieces


def _join_on_limb(pieces: list[np.ndarray], inside) -> list[np.ndarray]:
    """Close pieces ending on the limb into polygons, with the limb arcs between neighbouring ends that are inside.

    ``inside`` maps (N, 3) unit view points to booleans. Coincident ends (a boundary that runs out to the limb
    and back along itself) are never joined to each other.
    """
    if not pieces:
        return []
    ends = np.array([p[i] for p in pieces for i in (0, -1)])
    ang = np.arctan2(ends[:, 1], ends[:, 0])
    order = np.argsort(ang)
    nxt = np.roll(order, -1)
    span = np.diff(ang[order], append=ang[order[0]] + 2 * np.pi)
    mid = ang[order] + span / 2
    keep = inside(np.column_stack([np.cos(mid), np.sin(mid), np.zeros_like(mid)])) & (span > 1e-9)
    joined: dict[int, tuple[int, np.ndarray]] = {}
    for a, b, s in zip(order[keep], nxt[keep], span[keep], strict=True):
        arc = ang[a] + np.linspace(0.0, s, 91)
        joined[a] = (b, np.column_stack([np.cos(arc), np.sin(arc)]))
        joined[b] = (a, np.column_stack([np.cos(arc[::-1]), np.sin(arc[::-1])]))
    polys, used = [], set()
    for first in range(len(pieces)):
        if first in used:
            continue
        parts, e = [], 2 * first
        while e // 2 not in used:
            used.add(e // 2)
            parts.append(pieces[e // 2] if e % 2 == 0 else pieces[e // 2][::-1])
            if (e ^ 1) not in joined:
                break
            e, arc = joined[e ^ 1][0], joined[e ^ 1][1]
            parts.append(arc)
        poly = np.vstack(parts)
        polys.append(np.vstack([poly, poly[:1]]))
    return polys


def belt_polygons(rotation: np.ndarray, south, north, n: int = 361) -> list[np.ndarray]:
    """Visible part of the latitude belt ``south``..``north`` (degrees) of a unit globe, as 2D polygons.

    Each edge is a latitude or an array of ``n`` latitudes at longitudes ``linspace(-180, 180, n, endpoint=False)``.

    Each polygon is closed (first point repeated) and lies in the unit disc. The result has 0, 1 or
    more polygons. Edges are the visible arcs of the two latitude circles, joined by arcs of the
    limb that lie inside the belt; an edge seen whole (pole-on) is a loop, and a loop whose far
    side is the belt becomes a hole, bridged into the first polygon.
    """
    pole = rotation[:, 2]  # the north pole in the view frame

    t = np.linspace(-np.pi, np.pi, n, endpoint=False)
    south, north = (np.broadcast_to(np.asarray(e, float), t.shape) for e in (south, north))

    def in_belt(v: np.ndarray) -> np.ndarray:
        x, y, z = (v @ rotation).T  # body frame
        lon, lat = np.degrees(np.arctan2(y, x)), np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
        lon_t = np.degrees(t)
        return (np.interp(lon, lon_t, south, period=360) <= lat) & (lat <= np.interp(lon, lon_t, north, period=360))

    pieces: list[np.ndarray] = []
    outers: list[np.ndarray] = []
    holes: list[np.ndarray] = []
    for lat, is_south in ((south, True), (north, False)):
        if np.all(np.abs(lat) >= 90):
            continue
        pts = lat_lon_to_view(np.radians(lat), t, rotation)
        if np.all(pts[:, 2] >= 0):
            # Seen whole: its inside holds the visible pole, so the belt is inside iff that pole is on the belt's side.
            (outers if (pole[2] > 0) == is_south else holes).append(np.vstack([pts[:, :2], pts[:1, :2]]))
            continue
        pieces += _limb_pieces(pts)
    polys = _join_on_limb(pieces, in_belt)
    polys += outers
    if not polys and (holes or in_belt(np.array([[0.0, 0.0, 1.0]]))[0]):
        a = np.linspace(-np.pi, np.pi, 361)
        polys = [np.column_stack([np.cos(a), np.sin(a)])]
    if polys:
        holes = [h[::-1] if _signed_area(h) * _signed_area(polys[0]) > 0 else h for h in holes]
        polys[0] = with_holes(polys[0], holes)
    return polys


def ring_polygons(rotation: np.ndarray, lonlat_ring) -> list[np.ndarray]:
    """Visible part of the region inside a closed lon/lat ring of a unit globe, as closed 2D polygons in the unit disc.

    ``lonlat_ring`` is an (N, 2) array of east longitude in -180..180 and latitude, in degrees. The region is the
    even-odd inside of the ring in the lon/lat plane, so a ring round a pole must close along that pole, e.g.
    ``(-180, 60), (180, 60), (180, 90), (-180, 90)``. Edges are straight in lon/lat. Where the region runs past
    the limb it is closed along the limb rather than by pushing hidden points onto it.
    """
    ring = np.asarray(lonlat_ring, dtype=float)
    ring = np.vstack([ring, ring[:1]])
    seg = np.diff(ring, axis=0)
    steps = np.maximum(1, np.ceil(np.abs(seg).max(axis=1))).astype(int)  # at most a degree per step
    dense = np.vstack([a + np.outer(np.arange(m) / m, d) for a, d, m in zip(ring[:-1], seg, steps, strict=True)])
    pts = lat_lon_to_view(np.radians(dense[:, 1]), np.radians(dense[:, 0]), rotation)

    def inside(v: np.ndarray) -> np.ndarray:
        x, y, z = (v @ rotation).T  # body frame
        lonlat = np.column_stack([np.arctan2(y, x), np.arcsin(np.clip(z, -1.0, 1.0))])
        return points_in_polygon(np.degrees(lonlat), ring)

    if np.all(pts[:, 2] >= 0):
        return [np.vstack([pts[:, :2], pts[:1, :2]])]
    if np.all(pts[:, 2] < 0):
        return []
    return _join_on_limb(_limb_pieces(pts), inside)


def outside_caps(points3: np.ndarray, caps: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Mask of unit 3D ``points3`` lying outside every cap (unit centre vector, angular radius in radians)."""
    mask = np.ones(len(points3), bool)
    for centre, radius in caps:
        mask &= points3 @ centre < np.cos(radius)
    return mask


def destination(lat, lon, bearing, dist):
    """Point reached from (lat, lon) along an initial bearing for an angular distance on the unit sphere.

    All arguments and results are in degrees (broadcast as numpy arrays); the bearing is
    clockwise from north. The returned east longitude is wrapped to [-180, 180).
    """
    lat1, lon1, brg, d = np.radians(
        np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (lat, lon, bearing, dist)))
    )
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(brg))
    lon2 = lon1 + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), (np.degrees(lon2) + 180.0) % 360.0 - 180.0
