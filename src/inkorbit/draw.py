"""Drawing building blocks: orbits, bodies, apse and node lines, angles, vectors.

Each function adds items to a :class:`~inkorbit.scene.Scene` and returns what
it drew. Text passed as ``label=`` is placed automatically (see
:mod:`inkorbit.labels`); pass ``label=None`` to leave it off, or set
``scene.labels = False`` to hide every automatic label at once.

Angles are in radians. Lengths are in the scene's data units.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import geometry as geo
from . import style as st
from .elements import Elements, Frame
from .scene import Direction, Line, Scene, point_along

# Greek symbols on angles read better a little larger than running text.
SYMBOL_HEIGHT_MM = 3.2

# Sentinel meaning "use the conventional label for this thing".
DEFAULT = "__default__"

_APSIS_NAMES = {
    "earth": ("perigee", "apogee"),
    "moon": ("perilune", "apolune"),
    "sun": ("perihelion", "aphelion"),
}


def _label(text, default: str | None) -> str | None:
    return default if text == DEFAULT else text


def orbit(
    scene: Scene,
    orbit,
    *,
    role: str = st.ORBIT,
    arrows: Sequence[float] = (0.3, 0.8),
    frame: Frame = "perifocal",
    label: str | None = None,
    label_at: float = 0.35,
    max_radius: float | None = None,
) -> Line:
    """Draw an orbit from :class:`Elements` or from an (N, 2) / (N, 3) array of positions.

    3D positions are projected onto their own orbital plane, periapsis to the
    right (the perifocal view). ``arrows`` are fractions of the path length
    where direction arrowheads go; ``label_at`` is where the label attaches.
    """
    if isinstance(orbit, Elements):
        points = orbit.points(frame=frame, max_radius=max_radius)
    else:
        points = np.asarray(orbit, dtype=float)
        if points.ndim == 2 and points.shape[1] == 3:
            points = geo.project_to_plane(points)
    closed = isinstance(orbit, Elements) and orbit.is_closed
    line = scene.line(points, role, arrows=arrows)
    if label:
        # Let the label slide along the orbit if its first choice is crowded.
        fractions = [label_at + k * 0.06 * sign for k in range(1, 8) for sign in (1, -1)]
        anchors = [point_along(points, f % 1.0 if closed else f)[0] for f in fractions if closed or 0 < f < 1]
        scene.label(label, point_along(points, label_at)[0], priority=2, alternatives=anchors)
    return line


def body(
    scene: Scene,
    radius: float = 1.0,
    center: Sequence[float] = (0.0, 0.0),
    *,
    tilt: float = np.radians(20),
    spin: float = 0.0,
    sun: float | None = np.pi,
    sun_elevation: float = np.radians(25),
    grid_step: float = np.radians(30),
    label: str | None = None,
) -> None:
    """A globe: stippled night side, latitude/longitude grid and outline.

    ``sun`` is the direction to the Sun in the drawing (radians, 0 = right),
    ``sun_elevation`` lifts it towards the viewer, which narrows the night side.
    ``sun=None`` draws no night side. ``tilt`` leans the north pole towards the
    viewer and ``spin`` turns the globe about its axis.
    """
    c = np.asarray(center, dtype=float)
    if sun is not None:
        direction = np.array(
            [np.cos(sun_elevation) * np.cos(sun), np.cos(sun_elevation) * np.sin(sun), np.sin(sun_elevation)]
        )
        night = geo.night_side(direction)
        if night is not None:
            scene.fill(c + radius * night, "stipple")
    if grid_step:
        step = np.degrees(grid_step)
        for line in geo.graticule(geo.view_rotation(tilt, spin), step, step):
            scene.line(c + radius * line, st.GRATICULE)
    scene.line(geo.circle(tuple(c), radius), st.OUTLINE)
    if label:
        # Attach to the rim, upper right, where it reads as naming the globe.
        scene.label(label, c + radius * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)]), prefer="ne", priority=3)


def _broken_line(scene: Scene, start, end, role: str, gap_radius: float | None, center=(0.0, 0.0)) -> None:
    """A straight line with a gap where it would cross a body of ``gap_radius``."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    if not gap_radius:
        scene.line([start, end], role)
        return
    c = np.asarray(center, float)
    d = end - start
    # Parameters where the line enters and leaves the (slightly enlarged) disc.
    r = gap_radius * 1.08
    f = start - c
    a, b, cc = d @ d, 2 * f @ d, f @ f - r * r
    disc = b * b - 4 * a * cc
    if disc <= 0:
        scene.line([start, end], role)
        return
    t1, t2 = sorted(((-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)))
    if t1 > 0:
        scene.line([start, start + min(t1, 1) * d], role)
    if t2 < 1:
        scene.line([start + max(t2, 0) * d, end], role)


def apsides(
    scene: Scene,
    elements: Elements,
    *,
    body_radius: float | None = None,
    names: tuple[str, str] | str = "earth",
    frame: Frame = "perifocal",
    line_label: str | None = DEFAULT,
    overshoot: float = 0.12,
) -> None:
    """Line of apsides with markers and names at periapsis (and apoapsis for ellipses).

    ``names`` is a body name ("earth", "moon", "sun") or a pair of words;
    anything else gives "periapsis"/"apoapsis".
    """
    if isinstance(names, str):
        names = _APSIS_NAMES.get(names.lower(), ("periapsis", "apoapsis"))
    w = elements.angle_of(0.0, frame)
    u = np.array([np.cos(w), np.sin(w)])
    rp = elements.periapsis
    ext = overshoot * (rp + (elements.apoapsis if elements.is_closed else rp))
    far = -(elements.apoapsis + ext) if elements.is_closed else -0.3 * rp
    _broken_line(scene, far * u, (rp + ext) * u, st.APSE, body_radius)

    scene.marker(rp * u, "dot")
    scene.label(names[0], rp * u, prefer=_outward(u, "below"), priority=4)
    if elements.is_closed:
        scene.marker(-elements.apoapsis * u, "dot")
        scene.label(names[1], -elements.apoapsis * u, prefer=_outward(-u, "below"), priority=4)
    text = _label(line_label, "line of apsides")
    if text and elements.is_closed:
        scene.label(text, (far * 0.55) * u, prefer="n", priority=1)


def _outward(u: np.ndarray, side: str) -> Direction:
    """Compass direction pointing away from the centre along ``u``, biased below the line."""
    east = u[0] >= 0
    if side == "below":
        return "se" if east else "sw"
    return "ne" if east else "nw"


def nodes(
    scene: Scene,
    elements: Elements,
    *,
    body_radius: float | None = None,
    frame: Frame = "perifocal",
    labels: tuple[str, str] | None = ("ascending node", "descending node"),
) -> None:
    """Line of nodes. In the orbit's own plane the ascending node lies at true anomaly -argp."""
    nu_asc = -elements.argp
    nu_desc = np.pi - elements.argp
    a = elements.position(nu_asc, frame) if _reachable(elements, nu_asc) else None
    d = elements.position(nu_desc, frame) if _reachable(elements, nu_desc) else None
    w = elements.angle_of(nu_asc, frame)
    u = np.array([np.cos(w), np.sin(w)])
    reach_a = np.linalg.norm(a) * 1.12 if a is not None else elements.periapsis
    reach_d = np.linalg.norm(d) * 1.12 if d is not None else elements.periapsis
    _broken_line(scene, -reach_d * u, reach_a * u, st.APSE, body_radius)
    for point, text in zip((a, d), labels or (None, None), strict=True):
        if point is not None:
            scene.marker(point, "open")
            if text:
                scene.label(text, point, priority=3)


def _reachable(elements: Elements, nu: float) -> bool:
    wrapped = (nu + np.pi) % (2 * np.pi) - np.pi
    return elements.is_closed or abs(wrapped) < elements.nu_limit * 0.98


def angle(
    scene: Scene,
    start: float,
    stop: float,
    radius: float,
    *,
    center: Sequence[float] = (0.0, 0.0),
    label: str | None = None,
    role: str = st.CONSTRUCTION,
    arrow: bool = True,
) -> Line:
    """An angle arc from ``start`` to ``stop`` (radians), with an arrowhead and a label at mid-arc."""
    inset = min(np.radians(3), abs(stop - start) / 6)
    sign = 1 if stop >= start else -1
    pts = geo.arc(tuple(center), radius, start + sign * inset, stop - sign * inset)
    line = scene.line(pts, role)
    if arrow:
        scene.arrowhead(pts[-1], pts[-1] - pts[-2], size_mm=2.0)
    if label:
        mid = (start + stop) / 2
        at = np.asarray(center, float) + radius * np.array([np.cos(mid), np.sin(mid)])
        scene.label(label, at, prefer=_compass(mid), priority=5, height_mm=SYMBOL_HEIGHT_MM)
    return line


def _compass(theta: float) -> Direction:
    names: list[Direction] = ["e", "ne", "n", "nw", "w", "sw", "s", "se"]
    return names[int(np.round((theta % (2 * np.pi)) / (np.pi / 4))) % 8]


def vector(
    scene: Scene,
    start: Sequence[float],
    end: Sequence[float],
    *,
    label: str | None = None,
    role: str = st.ORBIT,
) -> Line:
    """An arrow from ``start`` to ``end``, labelled beyond its tip."""
    line = scene.vector(start, end, role)
    if label:
        d = np.asarray(end, float) - np.asarray(start, float)
        scene.label(label, end, prefer=_compass(np.arctan2(d[1], d[0])), priority=5)
    return line


def spacecraft(
    scene: Scene,
    elements: Elements,
    nu: float | None = None,
    *,
    frame: Frame = "perifocal",
    marker: str = "open",
    label: str | None = None,
) -> np.ndarray:
    """Mark the object's position at true anomaly ``nu`` (defaults to ``elements.nu``)."""
    at = elements.position(nu, frame)
    scene.marker(at, marker, size_mm=1.8 if marker == "open" else None)
    if label:
        scene.label(label, at, prefer=_compass(np.arctan2(at[1], at[0])), priority=4)
    return at


def radius_vector(
    scene: Scene,
    elements: Elements,
    nu: float | None = None,
    *,
    body_radius: float | None = None,
    frame: Frame = "perifocal",
    label: str | None = "r̄",
    role: str = st.SECONDARY,
) -> Line:
    """Position vector from the focus (or the body's surface) to the object."""
    at = elements.position(nu, frame)
    r = np.linalg.norm(at)
    u = at / r
    start = u * (body_radius * 1.08 if body_radius else 0.0)
    line = scene.vector(start, at - u * 0.03 * r, role)
    if label:
        mid = (start + at) / 2
        side = np.array([-u[1], u[0]])
        scene.label(label, mid, prefer=_compass(np.arctan2(side[1], side[0]) + np.pi), priority=3)
    return line


def velocity_vector(
    scene: Scene,
    elements: Elements,
    nu: float | None = None,
    *,
    length: float | None = None,
    frame: Frame = "perifocal",
    label: str | None = "v̄",
    role: str = st.ORBIT,
) -> Line:
    """Velocity direction at the object (drawn to ``length`` data units, default 0.35 periapsis radii)."""
    at = elements.position(nu, frame)
    length = length or 0.35 * elements.periapsis
    return vector(scene, at, at + length * elements.direction_of_motion(nu, frame), label=label, role=role)


def true_anomaly(
    scene: Scene,
    elements: Elements,
    nu: float | None = None,
    *,
    radius: float | None = None,
    frame: Frame = "perifocal",
    label: str | None = "ν",
) -> Line:
    """The true-anomaly arc from the periapsis direction to the object."""
    nu = elements.nu if nu is None else nu
    radius = radius or 0.45 * elements.periapsis
    w = elements.angle_of(0.0, frame)
    return angle(scene, w, w + nu, radius, label=label)
