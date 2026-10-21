import numpy as np

from inkorbit import geometry as geo
from inkorbit import maps


def _area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_ellipse_apsides():
    pts = geo.ellipse(2.0, 0.5)
    r = np.hypot(pts[:, 0], pts[:, 1])
    assert np.isclose(r.min(), 1.0, rtol=1e-4)
    assert np.isclose(r.max(), 3.0, rtol=1e-4)
    assert np.allclose(pts[0], [1.0, 0.0])


def test_night_side_half_disc_when_sun_is_sideways():
    night = geo.night_side(np.array([-1.0, 0.0, 0.0]))
    assert np.isclose(_area(night), np.pi / 2, rtol=1e-3)
    assert night[:, 0].min() > -1e-9  # the night is on the right


def test_night_side_edge_cases():
    assert geo.night_side(np.array([0.0, 0.0, 1.0])) is None
    assert np.isclose(_area(geo.night_side(np.array([0.0, 0.0, -1.0]))), np.pi, rtol=1e-3)


def test_tilt_leans_north_pole_towards_viewer():
    north = geo.lat_lon_to_view(np.pi / 2, 0.0, geo.view_rotation(tilt=np.radians(30)))
    assert north[1] > 0 and north[2] > 0


def test_graticule_is_on_the_visible_disc():
    for line in geo.graticule(geo.view_rotation(np.radians(20), np.radians(10))):
        assert np.all(np.hypot(line[:, 0], line[:, 1]) <= 1 + 1e-9)


def _pole(rotation):
    return geo.lat_lon_to_view(np.pi / 2, 0.0, rotation)


def test_equatorial_orbit_shows_the_pole_to_the_viewer():
    assert np.allclose(_pole(geo.orbit_view_rotation(0.0, 0.0, 0.0)), [0, 0, 1])
    # Equatorial in the plane frame is the inertial frame itself: lon 0 (vernal equinox) along +x.
    rotation = geo.orbit_view_rotation(0.0, 0.4, 0.7, in_plane=1.1)
    assert np.allclose(rotation, np.eye(3))


def test_pole_in_perifocal_frame():
    # Third row of the perifocal-from-inertial matrix R3(w) R1(i) R3(raan) (Vallado, sec. 2.6).
    i, w = np.radians(60), np.radians(30)
    expected = [np.sin(i) * np.sin(w), np.sin(i) * np.cos(w), np.cos(i)]
    assert np.allclose(_pole(geo.orbit_view_rotation(i, np.radians(50), w)), expected)


def test_plane_frame_turns_the_pole_in_plane():
    i, raan, w = np.radians(60), np.radians(50), np.radians(30)
    perifocal = _pole(geo.orbit_view_rotation(i, raan, w))
    plane = _pole(geo.orbit_view_rotation(i, raan, w, in_plane=raan + w))
    c, s = np.cos(raan + w), np.sin(raan + w)
    assert np.allclose(plane, [c * perifocal[0] - s * perifocal[1], s * perifocal[0] + c * perifocal[1], perifocal[2]])


def test_orbit_view_rotation_is_a_rotation():
    rotation = geo.orbit_view_rotation(0.3, 1.2, -0.8, 0.5)
    assert np.allclose(rotation @ rotation.T, np.eye(3)) and np.isclose(np.linalg.det(rotation), 1.0)


def test_with_holes_even_odd_fill():
    # Outer 4x4 square with a 1x1 hole in the middle: stipple and hatch must avoid the hole, not the rest.
    from inkorbit import patterns

    outer = np.array([[0, 0], [4, 0], [4, 4], [0, 4.0]])
    hole = np.array([[1.5, 1.5], [2.5, 1.5], [2.5, 2.5], [1.5, 2.5]])
    ring = geo.with_holes(outer, [hole])
    dots = patterns.stipple(ring, 0.2)
    in_hole = (np.abs(dots - 2) < 0.5).all(axis=1)
    assert len(dots) > 0 and not in_hole.any() and (~in_hole).any()
    for seg in patterns.hatch(ring, 0.25, 0):
        assert not (seg[:, 0].min() < 2.5 and seg[:, 0].max() > 1.5 and 1.5 < seg[0, 1] < 2.5)


def _lonlat_rect(lon0, lon1, lat0, lat1):
    return np.array([[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]], float)


def _visible_area(rotation, ring, n=600):
    """Area of the region's visible part, from a grid of visible points tested in lon/lat."""
    g = np.linspace(-1, 1, n)
    x, y = np.meshgrid(g, g)
    disc = x**2 + y**2 < 1
    body = np.column_stack([x[disc], y[disc], np.sqrt(1 - x[disc] ** 2 - y[disc] ** 2)]) @ rotation
    lonlat = np.degrees(np.column_stack([np.arctan2(body[:, 1], body[:, 0]), np.arcsin(body[:, 2])]))
    return geo.points_in_polygon(lonlat, ring).mean() * np.pi


def test_ring_crossing_the_limb_is_closed_along_the_limb():
    rotation, ring = geo.view_rotation(), _lonlat_rect(60, 120, -20, 20)  # lon 90 is the limb
    (poly,) = geo.ring_polygons(rotation, ring)
    r = np.hypot(*poly.T)
    assert r.max() <= 1 + 1e-9 and np.count_nonzero(np.isclose(r, 1.0)) > 10  # follows a limb arc
    assert np.isclose(_area(poly), _visible_area(rotation, ring), rtol=0.02)


def test_ring_round_the_visible_pole():
    # A polar cap as a lon/lat ring closes along the pole; seen with tilt it crosses the limb on the far side.
    rotation, ring = geo.view_rotation(np.radians(20)), _lonlat_rect(-180, 180, 60, 90)
    (poly,) = geo.ring_polygons(rotation, ring)
    assert np.hypot(*poly.T).max() <= 1 + 1e-9
    assert np.isclose(_area(poly), _visible_area(rotation, ring), rtol=0.02)


def test_ring_containing_the_whole_limb_fills_the_disc():
    # Everything south of 10N, seen from below the south pole: the edge is hidden all round.
    (poly,) = geo.ring_polygons(geo.view_rotation(np.radians(-90)), _lonlat_rect(-180, 180, -90, 10))
    assert np.isclose(_area(poly), np.pi, rtol=0.01)


def test_ring_fully_visible_or_hidden():
    rotation, ring = geo.view_rotation(), _lonlat_rect(-20, 20, -10, 10)
    (poly,) = geo.ring_polygons(rotation, ring)
    assert np.allclose(poly[0], poly[-1]) and np.hypot(*poly.T).max() < 1
    assert np.isclose(_area(poly), _visible_area(rotation, ring), rtol=0.02)
    assert geo.ring_polygons(rotation, _lonlat_rect(160, 180, -10, 10)) == []


def test_destination_round_trip():
    # Going out and coming straight back: the final bearing at the far point is not the reverse of the
    # initial one, so go out on a meridian (bearing 0), where it is exactly 180.
    lat, lon = geo.destination(20.0, 30.0, 0.0, 50.0)
    assert np.allclose([lat, lon], [70.0, 30.0])
    assert np.allclose(geo.destination(lat, lon, 180.0, 50.0), (20.0, 30.0))


def test_destination_along_equator_moves_longitude_by_distance():
    lat, lon = geo.destination(0.0, 10.0, 90.0, np.array([5.0, 100.0, 175.0]))
    assert np.allclose(lat, 0.0)
    assert np.allclose(lon, [15.0, 110.0, -175.0])  # last one wraps past 180


def test_destination_poles():
    # Due north a quarter circle from the equator is the north pole; due south from it, any longitude.
    lat, _ = geo.destination(0.0, 40.0, 0.0, 90.0)
    assert np.isclose(lat, 90.0)
    lat, _ = geo.destination(90.0, 0.0, 180.0, 30.0)
    assert np.isclose(lat, 60.0)
    lat, _ = geo.destination(-90.0, 0.0, 0.0, 30.0)
    assert np.isclose(lat, -60.0)


def test_small_circle_unchanged():
    # Copy of the formula small_circle used before it called geometry.destination.
    lat1, lon1, d = np.radians(35.0), np.radians(-170.0), np.radians(25.0)
    brg = np.linspace(0.0, 2 * np.pi, 121)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(brg))
    lon2 = lon1 + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2))
    expected = maps.split_at_antimeridian(np.degrees(lon2), np.degrees(lat2))
    got = maps.small_circle(35.0, -170.0, 25.0)
    assert len(got) == len(expected) > 1
    for g, e in zip(got, expected, strict=True):
        assert np.allclose(g, e)


def test_ring_arcs_tilted_halves_mirror_and_far_passes_behind_disc():
    rot = geo.view_rotation(tilt=np.radians(20))
    near, far = geo.ring_arcs(rot, 2.0)
    assert near.shape == far.shape == (181, 2)
    assert np.allclose(near[-1], far[0]) and np.allclose(far[-1], near[0])  # arcs join end to end
    assert np.allclose(near, -far)  # opposite points of the circle
    assert np.hypot(*far.T).min() < 1.0  # the far half runs behind the disc
    # radius scales the arc and keeps the ordering, so outer + reversed inner makes a band
    assert np.allclose(geo.ring_arcs(rot, 1.0)[0] * 2.0, near)


def test_ring_arcs_pole_on_is_all_near():
    near, far = geo.ring_arcs(geo.view_rotation(tilt=np.pi / 2), 1.5)  # pole facing the viewer: circle lies in z = 0
    assert far.shape == (0, 2)
    assert np.allclose(np.hypot(*near.T), 1.5) and np.allclose(near[0], near[-1])


def test_ring_arcs_edge_on_is_a_line():
    near, far = geo.ring_arcs(geo.view_rotation(tilt=0.0), 2.0)  # equator seen edge-on
    assert np.allclose(near[:, 1], 0.0) and np.allclose(far[:, 1], 0.0)
    assert np.isclose(np.abs(near[:, 0]).max(), 2.0)
