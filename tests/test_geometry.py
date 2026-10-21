import numpy as np

from inkorbit import geometry as geo


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
