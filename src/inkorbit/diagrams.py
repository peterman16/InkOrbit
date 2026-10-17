"""Ready-made diagrams. Each returns a :class:`~inkorbit.plate.Plate` you can
keep drawing on, save, or display in a notebook.

Use them directly (``inkorbit.diagrams.hohmann(...)``) or by name through
``inkorbit.diagram("hohmann", ...)``.

Orbits can be :class:`~inkorbit.Elements` or poliastro / hapsira ``Orbit``
objects; lengths are in kilometres unless the inputs carry astropy units.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import draw, geometry, maps
from . import style as st
from .adapters import as_elements, as_lat_lon, length_km, resolve_body
from .adapters import is_orbit_like as _orbit_like
from .bodies import Body
from .elements import Elements
from .plate import Plate
from .printed import Printed
from .scene import point_along


def _deg(x: float) -> str:
    return f"{np.degrees(x):.1f}°"


def _km(x: float) -> str:
    return f"{x:,.0f} km"


# --- 1. A single orbit with its elements ---------------------------------------------


def orbit(
    orbit,
    *,
    body: str | Body | None = None,
    body_radius: float | None = None,
    caption: str | None = None,
    name: str = "spacecraft",
    vectors: bool = True,
    elements_table: bool = True,
    coastlines: bool = False,
    page="figure",
    style=None,
    labels: bool = True,
    printed: bool | Printed = False,
) -> Plate:
    """One orbit in its own plane, periapsis to the right, with its elements labelled.

    Shows the body, the orbit with direction arrows, the line of apsides, the
    object with radius and velocity vectors and its true anomaly, and, for an
    inclined orbit, the line of nodes and the argument of periapsis. The
    element values are listed under the figure (``elements_table=False`` to omit).
    ``body_radius`` overrides the body's size, e.g. when the elements are not in km.
    """
    el = as_elements(orbit)
    central = resolve_body(body, [orbit])
    radius = body_radius if body_radius is not None else central.radius_km
    plate = Plate(caption, page=page, style=style, labels=labels, printed=printed)
    s = plate.scene

    draw.body(s, radius, coastlines=coastlines, orbit=el)
    draw.orbit(s, el, max_radius=4 * el.periapsis)
    draw.apsides(s, el, body_radius=radius, names=central.apsis_names)
    if not np.isclose(el.i, 0.0) and el.argp:
        draw.nodes(s, el, body_radius=radius)
        draw.angle(s, -el.argp, 0.0, _arc_radius(el, radius) * 1.25, label="ω")
    if vectors:
        draw.radius_vector(s, el, body_radius=radius)
        draw.velocity_vector(s, el, length=0.45 * max(el.periapsis, radius))
        draw.true_anomaly(s, el, radius=_arc_radius(el, radius))
    draw.spacecraft(s, el, label=name)

    if elements_table:
        s.notes = [
            f"a = {_km(el.a)}",
            f"e = {el.e:.3f}",
            f"i = {_deg(el.i)}",
            f"Ω = {_deg(el.raan)}",
            f"ω = {_deg(el.argp)}",
            f"ν = {_deg(el.nu)}",
        ]
    return plate


def _arc_radius(el: Elements, body_radius: float) -> float:
    """An angle-arc radius between the body and the orbit."""
    return max(body_radius * 1.3, 0.55 * el.periapsis)


# --- 2. Several orbits about one body ------------------------------------------------

_ROLES = (st.ORBIT, st.SECONDARY, st.TRANSFER)


def orbits(
    *orbits,
    names: Sequence[str] | None = None,
    roles: Sequence[str] | None = None,
    body: str | Body | None = None,
    body_radius: float | None = None,
    caption: str | None = None,
    coastlines: bool = False,
    page="figure",
    style=None,
    labels: bool = True,
    axes: bool = False,
    grid: bool = False,
    printed: bool | Printed = False,
) -> Plate:
    """Several orbits about one body, drawn in their common plane.

    Orbits may be elements / ``Orbit`` objects or arrays of positions. Each
    orbit is rotated by its longitude of periapsis, so coplanar orbits keep
    their true orientation. Line styles cycle through solid heavy, solid light
    and dashed unless ``roles`` says otherwise.
    """
    if not orbits:
        raise ValueError("pass at least one orbit")
    central = resolve_body(body, orbits)
    radius = body_radius if body_radius is not None else central.radius_km
    plate = Plate(
        caption,
        page=page,
        style=style,
        labels=labels,
        printed=printed,
        axes=axes,
        grid=grid,
        units="km" if axes else None,
    )
    s = plate.scene
    sources = [as_elements(item) if _orbit_like(item) else item for item in orbits]
    view = next((o for o in sources if isinstance(o, Elements)), None)
    draw.body(s, radius, coastlines=coastlines, orbit=view, frame="plane")
    names = list(names or [])
    roles = list(roles or [])
    for k, source in enumerate(sources):
        role = roles[k] if k < len(roles) else _ROLES[k % len(_ROLES)]
        label = names[k] if k < len(names) else None
        max_radius = 4 * source.periapsis if isinstance(source, Elements) else None
        draw.orbit(s, source, role=role, frame="plane", label=label, max_radius=max_radius)
    return plate


# --- 3. Hohmann transfer -------------------------------------------------------------


def hohmann(
    r_initial,
    r_final,
    *,
    body: str | Body | None = None,
    caption: str | None = None,
    names: tuple[str, str, str] = ("initial orbit", "final orbit", "transfer orbit"),
    delta_v: bool = True,
    page="figure",
    style=None,
    labels: bool = True,
    printed: bool | Printed = False,
) -> Plate:
    """A Hohmann transfer between two circular coplanar orbits.

    Radii are numbers in km, astropy lengths, or circular ``Orbit`` objects.
    With ``delta_v`` the two burns are drawn as vectors with their Δv, and the
    totals and time of flight are listed under the figure.
    """
    r1, r2 = _radius(r_initial), _radius(r_final)
    if np.isclose(r1, r2):
        raise ValueError("initial and final radii are the same: there is no transfer")
    given = [o for o in (r_initial, r_final) if _orbit_like(o)]
    central = resolve_body(body, given)
    radius, mu = central.radius_km, central.mu_km3_s2

    a = (r1 + r2) / 2
    transfer = Elements(a=a, e=abs(r2 - r1) / (r1 + r2))
    if r1 < r2:
        half = transfer.points(nu_range=(0.0, np.pi), n=361)
    else:
        # Start at apoapsis on the right and descend to periapsis on the left.
        half = geometry.rotate(transfer.points(nu_range=(np.pi, 2 * np.pi), n=361), np.pi)

    plate = Plate(caption, page=page, style=style, labels=labels, printed=printed)
    s = plate.scene
    draw.body(s, radius, orbit=as_elements(given[0]) if given else Elements(a=r1), frame="plane")
    draw.orbit(s, Elements(a=r1), role=st.SECONDARY, frame="plane", label=names[0], label_at=0.62)
    draw.orbit(s, Elements(a=r2), role=st.SECONDARY, frame="plane", label=names[1], label_at=0.88)
    s.line(half, st.TRANSFER, arrows=(0.5,))
    s.label(names[2], point_along(half, 0.3)[0], priority=2)

    start, end = half[0], half[-1]
    s.marker(start, "dot")
    s.marker(end, "dot")
    if delta_v and np.isfinite(mu):
        dv1 = np.sqrt(mu * (2 / r1 - 1 / a)) - np.sqrt(mu / r1)
        dv2 = np.sqrt(mu / r2) - np.sqrt(mu * (2 / r2 - 1 / a))
        tof = np.pi * np.sqrt(a**3 / mu)
        # Burns are tangential: +y at the start (right), -y at the end (left), scaled for visibility.
        burn = 0.25 * max(r1, r2)
        draw.vector(s, start, start + np.array([0.0, burn * np.sign(dv1)]), label="Δv1")
        draw.vector(s, end, end + np.array([0.0, -burn * np.sign(dv2)]), label="Δv2")
        s.notes = [
            f"Δv1 = {dv1:+.3f} km/s",
            f"Δv2 = {dv2:+.3f} km/s",
            f"total Δv = {abs(dv1) + abs(dv2):.3f} km/s",
            f"time of flight = {_duration(tof)}",
        ]
    return plate


def _radius(value) -> float:
    if _orbit_like(value):
        el = as_elements(value)
        if not np.isclose(el.e, 0.0, atol=1e-3):
            raise ValueError("a Hohmann transfer needs circular orbits (e = 0)")
        return float(el.a)
    return float(length_km(value))


def _duration(seconds: float) -> str:
    hours = seconds / 3600
    if hours < 48:
        return f"{hours:.2f} h"
    return f"{hours / 24:.1f} days"


# --- 4. Ground track ---------------------------------------------------------------


def ground_track(
    track,
    lon=None,
    *,
    stations: Sequence = (),
    events: Sequence = (),
    caption: str | None = None,
    name: str | None = None,
    latitude_limit: float = 80.0,
    land: str | None = None,
    page="map",
    style=None,
    labels: bool = True,
    printed: bool | Printed = False,
) -> Plate:
    """A ground track on a ruled equirectangular map with coastlines.

    ``track`` is latitudes (with ``lon``) in degrees, astropy angles, or a
    Skyfield geocentric position over several times. ``stations`` are
    ``(name, lat, lon)`` or ``(name, lat, lon, radius_deg)``: each gets a
    numbered marker, a visibility circle when a radius is given, and a line in
    the legend under the map. ``events`` are ``(text, lat, lon)`` points along
    the track. ``land="stipple"`` stipples the continents.
    """
    lat, lon = as_lat_lon(track, lon)
    plate = Plate(caption, page=page, style=style, labels=labels, printed=printed, axes=True, grid=True)
    s = plate.scene
    lim = float(latitude_limit)
    box = (-180.0, 180.0, -lim, lim)
    s.limits = box
    s.numbers_all_sides = True
    s.xticks = np.arange(-180, 181, 30)
    s.yticks = np.arange(-80, 81, 20)
    s.tick_format = lambda v: f"{abs(v):g}°"

    if land == "stipple":
        for ring in maps.land():
            clipped = geometry.clip_polygon(ring, *box)
            if clipped is not None:
                s.fill(clipped, "stipple", spacing_mm=0.7)
    elif land is not None:
        raise ValueError("land must be None or 'stipple'")
    for coast in maps.coastlines():
        for run in geometry.clip_polyline(coast, *box):
            s.line(run, st.COASTLINE)

    runs = []
    for run in maps.split_at_antimeridian(lon, lat):
        runs += geometry.clip_polyline(run, *box)
    for run in runs:
        s.line(run, st.SECONDARY, arrows=(0.5,) if _length(run) > 60 else ())
    if name and runs:
        longest = max(runs, key=_length)
        anchors = [point_along(longest, f)[0] for f in (0.35, 0.45, 0.25, 0.55, 0.15, 0.65)]
        s.label(name, anchors[0], priority=3, alternatives=anchors[1:])

    notes = []
    for k, station in enumerate(stations, start=1):
        station_name, slat, slon, *rest = station
        if rest:
            for ring in maps.small_circle(slat, slon, rest[0]):
                for run in geometry.clip_polyline(ring, *box):
                    s.line(run, st.OUTLINE)
        s.marker((maps.wrap_longitude(slon), slat), "open", size_mm=1.4)
        s.label(str(k), (maps.wrap_longitude(slon), slat), priority=5)
        notes.append(f"{k}. {station_name}")
    for text, elat, elon in events:
        s.marker((maps.wrap_longitude(elon), elat), "dot")
        s.label(text, (maps.wrap_longitude(elon), elat), priority=4)
    s.notes = notes
    return plate


def _length(points: np.ndarray) -> float:
    return float(np.hypot(*np.diff(points, axis=0).T).sum())


# --- By name -------------------------------------------------------------------------

DIAGRAMS = {
    "orbit": orbit,
    "orbits": orbits,
    "hohmann": hohmann,
    "ground_track": ground_track,
}


def diagram(kind: str, *args, **kwargs) -> Plate:
    """Make a diagram by name: one of ``"orbit"``, ``"orbits"``, ``"hohmann"``, ``"ground_track"``."""
    try:
        fn = DIAGRAMS[kind]
    except KeyError:
        raise ValueError(f"unknown diagram {kind!r}; choose from {sorted(DIAGRAMS)}") from None
    return fn(*args, **kwargs)
