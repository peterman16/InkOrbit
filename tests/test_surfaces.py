import numpy as np
import pytest

from inkorbit import geometry, surfaces


def _sep(lat, lon, ring):
    """Angular distance in degrees from (lat, lon) to each (lon, lat) point of ``ring``."""
    la, lo, pl, pn = np.radians(lat), np.radians(lon), np.radians(ring[:, 1]), np.radians(ring[:, 0])
    return np.degrees(np.arccos(np.clip(np.sin(la) * np.sin(pl) + np.cos(la) * np.cos(pl) * np.cos(pn - lo), -1, 1)))


def test_craters_only_for_bodies_with_data():
    for name in ("moon", "mercury", "venus", "mars"):
        found = surfaces.craters(name)
        assert found and len(found) <= 100
        assert all(len(c) == 4 for c in found)
    assert surfaces.craters("jupiter") == []


def test_rim_is_seeded_and_stays_near_the_circle():
    ring = surfaces.rim("moon", 3, 10.0, 20.0, 5.0)
    assert np.array_equal(ring, surfaces.rim("moon", 3, 10.0, 20.0, 5.0))
    assert not np.array_equal(ring, surfaces.rim("moon", 4, 10.0, 20.0, 5.0))
    assert ring.shape == (181, 2)
    # four harmonics of at most 3.5% each
    assert np.all(np.abs(_sep(10.0, 20.0, ring) / 5.0 - 1) <= 0.14)


@pytest.mark.parametrize("mons, side", [(False, -1), (True, 1)])
def test_shadow_lies_on_the_west_wall_of_a_crater_and_the_east_flank_of_a_mons(mons, side):
    lat, lon, r = 5.0, 30.0, 4.0
    ring = surfaces.shadow(lat, lon, r, mons)
    dist = _sep(lat, lon, ring)
    assert dist.min() > 0 and (dist.max() <= r * 1.25 + 1e-6)
    assert np.all((ring[:, 0] - lon) * side >= -1e-9)  # west of the centre for a crater, east for a mons
    assert (dist.max() > r + 0.5) == mons  # a mons shadow reaches beyond its rim, a crater's stays inside


def test_outside_caps_masks_points_inside_any_cap():
    pts = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    caps = [(np.array([1.0, 0, 0]), np.radians(10)), (np.array([0, 1.0, 0]), np.radians(60))]
    assert geometry.outside_caps(pts, caps).tolist() == [False, False, True]
    assert geometry.outside_caps(pts, []).all()
