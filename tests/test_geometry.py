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
