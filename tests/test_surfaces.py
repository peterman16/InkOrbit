from importlib import resources

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


def test_mountains_only_on_mars_and_venus():
    for name, has_mons in (("moon", False), ("mercury", False), ("venus", True), ("mars", True)):
        assert any(c[3] for c in surfaces.craters(name)) == has_mons


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


@pytest.mark.parametrize("name", ["moon", "mercury", "venus"])
def test_regions_are_closed_rings_that_never_jump_across_the_map(name):
    base, core = surfaces.regions(name)
    assert base and core
    for ring in (*base, *core):
        assert ring.dtype == np.float32 and ring.ndim == 2 and ring.shape[1] == 2
        assert np.array_equal(ring[0], ring[-1])
        assert np.all(np.abs(ring[:, 0]) <= 180) and np.all(np.abs(ring[:, 1]) <= 90)
        assert np.abs(np.diff(ring[:, 0])).max() <= 10  # a ring around a pole runs along it in steps


@pytest.mark.parametrize("name", ["mars", "jupiter"])
def test_bodies_without_regions_get_empty_lists(name):
    assert surfaces.regions(name) == ([], [])


def test_the_moons_maria_are_where_they_are():
    # Oceanus Procellarum (~ -57 lon, 18 lat) is inside a base ring; the far side's highlands (~ 180, 0) are not.
    from matplotlib.path import Path

    base, _ = surfaces.regions("moon")
    inside = lambda lon, lat: any(Path(r).contains_point((lon, lat)) for r in base)  # noqa: E731
    assert inside(-57, 18) and not inside(180, 0)


@pytest.mark.parametrize("name", ["moon", "mercury", "venus", "mars"])
def test_surface_files_stay_small(name):
    assert resources.files("inkorbit").joinpath(f"data/{name}_surface.npz").stat().st_size < 60_000
