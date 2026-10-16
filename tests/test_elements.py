import numpy as np
import pytest

import inkorbit as ink
from inkorbit import geometry as geo


def test_ellipse_apsides_and_positions():
    el = ink.Elements(a=2.0, e=0.5)
    assert np.isclose(el.periapsis, 1.0) and np.isclose(el.apoapsis, 3.0)
    assert np.allclose(el.position(0.0), [1.0, 0.0])
    assert np.allclose(el.position(np.pi), [-3.0, 0.0])


def test_direction_of_motion_is_tangent_and_prograde():
    el = ink.Elements(a=2.0, e=0.5)
    assert np.allclose(el.direction_of_motion(0.0), [0.0, 1.0])
    nu = 1.0
    eps = 1e-6
    numeric = el.position(nu + eps) - el.position(nu - eps)
    assert np.allclose(el.direction_of_motion(nu), numeric / np.linalg.norm(numeric), atol=1e-6)


def test_plane_frame_rotates_by_longitude_of_periapsis():
    el = ink.Elements(a=2.0, e=0.5, raan=0.2, argp=0.3)
    expected = geo.rotate(np.array([[1.0, 0.0]]), 0.5)[0]
    assert np.allclose(el.position(0.0, frame="plane"), expected)


def test_hyperbola_points_stop_at_max_radius():
    el = ink.Elements(a=-2.0, e=1.5)
    pts = el.points(max_radius=10.0)
    r = np.hypot(pts[:, 0], pts[:, 1])
    assert np.isclose(r.max(), 10.0, rtol=1e-6)
    assert np.isclose(r.min(), el.periapsis, rtol=1e-4)


def test_invalid_elements():
    with pytest.raises(ValueError, match="parabolic"):
        ink.Elements(a=1.0, e=1.0)
    with pytest.raises(ValueError, match="eccentricity"):
        ink.Elements(a=1.0, e=-0.1)


def test_project_3d_orbit_to_perifocal_plane():
    el = ink.Elements(a=2.0, e=0.4)
    flat = el.points(n=181)[:-1]
    tilt = np.array([[1, 0, 0], [0, np.cos(0.7), -np.sin(0.7)], [0, np.sin(0.7), np.cos(0.7)]])
    spun = geo.rotate(flat, 0.9)
    pts3 = np.column_stack([spun, np.zeros(len(spun))]) @ tilt.T
    projected = geo.project_to_plane(pts3)
    assert np.allclose(projected, flat, atol=1e-9)
