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
