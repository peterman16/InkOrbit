import numpy as np
import pytest

import inkorbit as ink
from inkorbit import draw
from inkorbit import geometry as geo
from inkorbit import style as st
from inkorbit.scene import Arrowhead, Fill, Label, Line, Marker

ELLIPSE = ink.Elements(a=2.6, e=0.55, nu=np.radians(110))


def _kinds(scene):
    return [type(i) for i in scene.items]


def test_orbit_from_elements_has_arrows_and_label():
    scene = ink.Scene()
    draw.orbit(scene, ELLIPSE, arrows=(0.2, 0.7), label="orbit")
    kinds = _kinds(scene)
    assert kinds.count(Line) == 1 and kinds.count(Arrowhead) == 2 and kinds.count(Label) == 1
    assert scene.items[-1].alternatives is not None


def test_orbit_from_3d_positions_is_projected():
    scene = ink.Scene()
    pts = ELLIPSE.points(n=100)
    pts3 = np.column_stack([pts, np.zeros(len(pts))])
    line = draw.orbit(scene, pts3, arrows=())
    assert line.points.shape == (100, 2)


def test_body_draws_night_grid_and_outline():
    scene = ink.Scene()
    draw.body(scene, 2.0, center=(1.0, 1.0), label="EARTH")
    assert isinstance(scene.items[0], Fill)
    roles = {i.role for i in scene.items if isinstance(i, Line)}
    assert roles == {st.GRATICULE, st.OUTLINE}
    for item in scene.items:
        if isinstance(item, Line):
            assert np.all(np.hypot(*(item.points - 1.0).T) <= 2.0 + 1e-9)


def _line_points(scene):
    return [i.points for i in scene.items if isinstance(i, Line)]


def test_body_without_orbit_keeps_tilt_and_spin():
    # Reference: the globe as drawn before orbits could orient it.
    tilt, spin = np.radians(20), np.radians(15)
    step = np.degrees(np.radians(30))  # the default grid step, as draw.body converts it
    expected = [2.0 * line for line in geo.graticule(geo.view_rotation(tilt, spin), step, step)]
    scene = ink.Scene()
    draw.body(scene, 2.0, spin=spin, sun=None)
    lines = _line_points(scene)
    assert len(lines) == len(expected) + 1  # plus the outline
    assert all(np.allclose(a, b) for a, b in zip(lines, expected, strict=False))


def test_body_seen_down_an_equatorial_orbit_has_concentric_latitudes():
    scene = ink.Scene()
    draw.body(scene, sun=None, orbit=ink.Elements(a=2.0))
    for line in _line_points(scene):
        # Each line is a circle about the pole or a radial meridian.
        r = np.hypot(*line.T)
        ux, uy = line[np.argmax(r)] / r.max()
        assert np.allclose(r, r[0]) or np.allclose(line[:, 0] * uy - line[:, 1] * ux, 0)


def test_body_orbit_replaces_tilt_and_spin_adds_a_turn():
    el = ink.Elements(a=2.0, i=0.5, raan=0.3, argp=0.6)
    a, b = ink.Scene(), ink.Scene()
    draw.body(a, sun=None, orbit=el, tilt=1.0, spin=0.2)
    draw.body(b, sun=None, orbit=ink.Elements(a=2.0, i=0.5, raan=0.5, argp=0.6))
    assert all(np.allclose(p, q) for p, q in zip(_line_points(a), _line_points(b), strict=True))


def test_body_without_sun_has_no_night_side():
    scene = ink.Scene()
    draw.body(scene, sun=None)
    assert Fill not in _kinds(scene)


def test_apse_line_breaks_around_the_body():
    scene = ink.Scene()
    draw.apsides(scene, ELLIPSE, body_radius=1.0)
    apse = [i for i in scene.items if isinstance(i, Line) and i.role == st.APSE]
    assert len(apse) == 2
    for line in apse:
        assert np.all(np.abs(line.points[:, 0]) >= 1.0)
    names = [i.text for i in scene.items if isinstance(i, Label)]
    assert names[:2] == ["perigee", "apogee"]


def test_apsis_names_follow_the_body():
    scene = ink.Scene()
    draw.apsides(scene, ELLIPSE, names="moon", line_label=None)
    assert [i.text for i in scene.items if isinstance(i, Label)] == ["perilune", "apolune"]


def test_hyperbola_has_only_periapsis():
    scene = ink.Scene()
    draw.apsides(scene, ink.Elements(a=-2.0, e=1.5), names="sun")
    assert [i.text for i in scene.items if isinstance(i, Label)] == ["perihelion"]
    assert _kinds(scene).count(Marker) == 1


def test_vectors_and_angle():
    scene = ink.Scene()
    draw.radius_vector(scene, ELLIPSE, body_radius=1.0)
    draw.velocity_vector(scene, ELLIPSE)
    draw.true_anomaly(scene, ELLIPSE)
    draw.spacecraft(scene, ELLIPSE, label="spacecraft")
    labels = [i.text for i in scene.items if isinstance(i, Label)]
    assert labels == ["r̄", "v̄", "ν", "spacecraft"]
    assert _kinds(scene).count(Arrowhead) == 3


def test_nodes_line():
    scene = ink.Scene()
    draw.nodes(scene, ink.Elements(a=2.0, e=0.3, i=0.5, argp=np.radians(40)), body_radius=0.5)
    assert [i.text for i in scene.items if isinstance(i, Label)] == ["ascending node", "descending node"]


def _surface_lines(scene):
    return [i for i in scene.items if isinstance(i, Line) and i.role == st.SURFACE]


def test_surface_is_drawn_only_for_bodies_with_features():
    earth, moon = ink.Scene(), ink.Scene()
    draw.body(earth, 1.0, body="earth", surface=True)
    draw.body(moon, 1.0, body="moon", surface=True)
    assert _surface_lines(earth)
    assert not _surface_lines(moon)  # never falls back to Earth's coastlines


def test_surface_needs_a_body():
    with pytest.raises(ValueError, match="surface needs body="):
        draw.body(ink.Scene(), 1.0, surface=True)


def test_gas_giant_bands_and_great_red_spot():
    from inkorbit import surfaces

    assert surfaces.features("saturn") and not surfaces.features("neptune")
    jupiter = surfaces.features("jupiter")
    assert jupiter and np.allclose(jupiter[-1][0], jupiter[-1][-1])  # the spot outline is closed


def _area(poly):
    return abs(geo._signed_area(poly))


def _strip(y):  # area of the unit disc below the chord at height y, plus pi/2 (analytic orthographic belt area)
    return y * np.sqrt(1 - y * y) + np.arcsin(y)


def test_equatorial_belt_at_zero_tilt_is_the_analytic_strip():
    (poly,) = geo.belt_polygons(geo.view_rotation(0.0), 7, 17)
    expected = _strip(np.sin(np.radians(17))) - _strip(np.sin(np.radians(7)))  # y = sin(lat) when seen edge-on
    assert _area(poly) == pytest.approx(expected, rel=2e-3)


def test_belt_polygons_stay_in_the_disc_for_every_view():
    for tilt in np.radians([0, 20, 60, 89, 90, -90, -45]):
        for belt in [(7, 17), (-20, -7), (45, 90), (-90, -45), (20, 35), (-7, 7)]:
            for poly in geo.belt_polygons(geo.view_rotation(tilt, 0.7), *belt):
                assert np.all(np.hypot(*poly.T) <= 1 + 1e-9)


def test_pole_on_cap_is_a_disc_and_pole_on_belt_is_a_ring():
    pole_on = geo.view_rotation(np.pi / 2)
    (cap,) = geo.belt_polygons(pole_on, 45, 90)
    assert np.hypot(*cap.T).max() == pytest.approx(np.cos(np.radians(45)))
    assert geo.belt_polygons(pole_on, -90, -45) == []  # the far pole is hidden
    (ring,) = geo.belt_polygons(pole_on, 20, 35)
    assert _area(ring) == pytest.approx(np.pi * (np.cos(np.radians(20)) ** 2 - np.cos(np.radians(35)) ** 2), rel=2e-3)


def test_belt_crossing_the_limb_is_one_polygon_with_both_edges():
    (poly,) = geo.belt_polygons(geo.view_rotation(np.radians(20)), 45, 90)
    assert np.isclose(np.hypot(*poly.T).max(), 1.0)  # reaches the limb
    assert poly[:, 1].max() > 0.9  # and the cap near the pole


def test_belts_stipple_jupiter_but_not_neptune():
    def fills(name):
        s = ink.Scene()
        draw.body(s, 1.0, body=name, surface=True, sun=None)
        return [i for i in s.items if isinstance(i, Fill)]

    assert fills("jupiter") and not fills("neptune")


def test_belt_edges_are_deterministic_wavy_and_clear_of_the_red_spot():
    from inkorbit import surfaces

    lon = np.linspace(-180, 180, 361)
    first, again = surfaces.edge("jupiter", 17, lon), surfaces.edge("jupiter", 17, lon)
    assert np.array_equal(first, again) and np.ptp(first) > 0.5
    assert np.allclose(surfaces.edge("jupiter", 90, lon), 90)  # the pole side stays put
    assert np.ptp(surfaces.edge("saturn", 35, lon)) < np.ptp(first)
    spot = np.abs(lon) < 7
    half = 3 * np.sqrt(1 - (lon[spot] / 7) ** 2)  # the spot's half-height, centred at -23
    assert np.all(surfaces.edge("jupiter", -26, lon)[spot] <= -23 - half)
    assert np.all(surfaces.edge("jupiter", -20, lon)[spot] >= -23 + half)
