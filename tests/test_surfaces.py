from importlib import resources

import numpy as np
import pytest

from inkorbit import surfaces


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
