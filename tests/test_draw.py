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
    earth, neptune = ink.Scene(), ink.Scene()
    draw.body(earth, 1.0, body="earth", surface=True)
    draw.body(neptune, 1.0, body="neptune", surface=True)
    assert _surface_lines(earth)
    assert not _surface_lines(neptune)  # no features for Neptune: never falls back to Earth's coastlines


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


def _fills(name, spacing, spin=0.0):
    s = ink.Scene()
    draw.body(s, 1.0, body=name, surface=True, sun=None, spin=spin)
    return [i for i in s.items if isinstance(i, Fill) and i.spacing_mm == spacing]


def test_moon_regions_are_stippled_in_two_tones_and_mars_has_none():
    for spacing in (st.DEFAULT_STYLE.region_stipple_mm, st.DEFAULT_STYLE.region_core_stipple_mm):
        assert _fills("moon", spacing)
        assert not _fills("mars", spacing)  # no region data for Mars


@pytest.mark.parametrize("name", ["moon", "mercury", "venus", "mars"])
def test_rocky_bodies_get_crater_shadows_and_rims(name):
    assert _fills(name, st.DEFAULT_STYLE.crater_shadow_stipple_mm)
    s = ink.Scene()
    draw.body(s, 1.0, body=name, surface=True)
    assert _surface_lines(s)


def test_crater_shadows_stay_small_across_the_antimeridian():
    # Facing longitude 180: a shadow ring crossing it must not become a band around the globe.
    shadows = _fills("moon", st.DEFAULT_STYLE.crater_shadow_stipple_mm, spin=np.pi)
    assert shadows and max(_area(f.polygon) for f in shadows) < 0.05


def test_some_crater_floors_are_cut_out_of_the_maria():
    maria = _fills("moon", st.DEFAULT_STYLE.region_stipple_mm)
    assert any(len(f.polygon) > 1 + len(np.unique(f.polygon, axis=0)) for f in maria)  # a hole repeats the start


def test_rocky_globe_renders_the_same_twice():
    def png():
        s = ink.Scene()
        draw.body(s, 1.0, body="moon", surface=True)
        return s.render().to_array(dpi=72)

    assert np.array_equal(png(), png())


def _ring_stipples(scene):
    return [i for i in scene.items if isinstance(i, Fill) and i.spacing_mm in st.DEFAULT_STYLE.ring_stipple_mm]


def test_saturn_surface_adds_ring_bands_and_other_bodies_are_unchanged():
    plain, ringed, jupiter = ink.Scene(), ink.Scene(), ink.Scene()
    draw.body(plain, 1.0, body="saturn")
    draw.body(ringed, 1.0, body="saturn", surface=True)
    draw.body(jupiter, 1.0, body="jupiter", surface=True)
    assert len(_ring_stipples(ringed)) == 6  # near and far halves of C, B and A
    # Paper behind the globe and under the three near halves; none without rings.
    assert sum(i.pattern == "paper" for i in ringed.items if isinstance(i, Fill)) == 4
    for scene in (plain, jupiter):
        assert not _ring_stipples(scene) and not any(isinstance(i, Fill) and i.pattern == "paper" for i in scene.items)


def test_ring_label_sits_beyond_the_a_ring():
    scene = ink.Scene()
    draw.body(scene, 1.0, body="saturn", surface=True, label="SATURN")
    (label,) = [i for i in scene.items if isinstance(i, Label)]
    assert np.linalg.norm(label.at) > 2.0  # out at the A ring's edge (2.27 radii), not on the bands


@pytest.mark.parametrize(("tilt", "halves"), [(0.0, 6), (np.pi / 2, 3)])
def test_edge_on_and_pole_on_rings_draw(tilt, halves):
    scene = ink.Scene()
    draw.body(scene, 1.0, body="saturn", surface=True, tilt=tilt, label="SATURN")
    scene.render()
    assert len(_ring_stipples(scene)) == halves  # pole-on there is no far half


def _lines(scene, role):
    return [i for i in scene.items if isinstance(i, Line) and i.role == role]


@pytest.mark.parametrize("elevation", [0.0, 0.4, 1.2])
def test_oblique_orbit_projects_the_apogee(elevation):
    # Equatorial, argp = 0: apogee at inertial (-a(1 + e), 0, 0). Azimuth 90 deg puts the -x axis to the right
    # at any elevation (view_rotation: longitude 90E faces the viewer), so apogee is drawn at a(1 + e) = 3.
    line = draw.orbit(ink.Scene(), ink.Elements(a=2.0, e=0.5), view=(elevation, np.pi / 2), arrows=())
    assert np.hypot(*line.points.T).max() == pytest.approx(3.0, rel=1e-5)
    # Azimuth 0: the apse line points at the viewer and tilts up by the elevation.
    line = draw.orbit(ink.Scene(), ink.Elements(a=2.0, e=0.5), view=(elevation, 0.0), arrows=())
    assert line.points[:, 1].max() == pytest.approx(3.0 * np.sin(elevation), abs=1e-4)


def test_oblique_orbit_behind_the_globe_is_hidden_without_arrowheads():
    # Circular equatorial orbit of radius 2 at azimuth 0: half way round (nu = pi) it is at view
    # (0, 2 sin el, -2 cos el), behind the unit globe; a quarter round (nu = pi/2) it is at (2, 0, 0), clear of it.
    scene = ink.Scene()
    line = draw.orbit(scene, ink.Elements(a=2.0), view=(0.3, 0.0), body_radius=1.0, arrows=(0.25, 0.5))
    (front,), (hidden,) = _lines(scene, st.ORBIT), _lines(scene, st.HIDDEN)
    assert line is front
    assert np.all(np.hypot(*hidden.points.T) <= 1.0 + 1e-9)
    assert np.allclose(front.points[[0, -1]], hidden.points[[-1, 0]])
    (arrow,) = [i for i in scene.items if isinstance(i, Arrowhead)]
    assert np.allclose(arrow.at, [2.0, 0.0], atol=1e-3)


def test_orbit_without_view_is_unchanged():
    a, b = ink.Scene(), ink.Scene()
    draw.orbit(a, ELLIPSE, label="orbit")
    draw.orbit(b, ELLIPSE, label="orbit", view=None, body_radius=1.0)
    assert len(a.items) == len(b.items)
    for p, q in zip(a.items, b.items, strict=True):
        assert type(p) is type(q) and all(np.array_equal(getattr(p, k), getattr(q, k)) for k in vars(p))


# Seen from azimuth 0 the inertial +x axis points at the viewer, tilted down by the elevation: with argp = raan = 0
# a point at inertial (-r, 0, 0) is drawn at (0, r sin el) and lies behind the globe when r sin el < R.
OBLIQUE = (0.2, 0.0)


def _texts(scene):
    return [i.text for i in scene.items if isinstance(i, Label)]


def test_oblique_apse_line_through_the_body_is_hidden_and_the_apsis_behind_is_dropped():
    scene = ink.Scene()
    draw.apsides(scene, ink.Elements(a=4.0, e=0.5), body_radius=1.5, view=OBLIQUE)  # apogee 6 sin 0.2 = 1.19 < 1.5
    assert _lines(scene, st.APSE) and _lines(scene, st.HIDDEN)
    for line in _lines(scene, st.HIDDEN):
        assert np.all(np.hypot(*line.points.T) <= 1.5 + 1e-9)
    assert _texts(scene) == ["perigee", "line of apsides"]
    assert _kinds(scene).count(Marker) == 1


def test_oblique_nodes_drop_the_node_behind_the_globe():
    scene = ink.Scene()
    draw.nodes(scene, ink.Elements(a=2.0, i=0.5), body_radius=1.0, view=OBLIQUE)  # descending node at (-2, 0, 0)
    assert _texts(scene) == ["ascending node"]
    assert _lines(scene, st.HIDDEN)


def test_oblique_spacecraft_behind_the_globe_is_not_drawn():
    el = ink.Elements(a=4.0, e=0.5)
    behind, front = ink.Scene(), ink.Scene()
    at = draw.spacecraft(behind, el, np.pi, label="sc", view=OBLIQUE, body_radius=1.5)
    assert behind.items == [] and np.allclose(at, [0.0, 6 * np.sin(0.2)])
    draw.spacecraft(front, el, 0.0, label="sc", view=OBLIQUE, body_radius=1.5)
    assert _kinds(front) == [Marker, Label]
