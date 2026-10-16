import numpy as np
import pytest

from inkorbit import printed

DPI = 300


def _rgba(ink_mask):
    out = np.full((*ink_mask.shape, 4), 255, np.uint8)
    out[ink_mask, :3] = 0
    return out


@pytest.fixture
def page():
    """A 25 mm square page with a thick bar and a thin hairline, like a real figure."""
    mask = np.zeros((295, 295), bool)
    mask[40:60, 20:270] = True  # ~1.7 mm bar
    mask[150:152, 20:270] = True  # ~0.17 mm hairline
    return _rgba(mask), mask


def test_output_is_bool_and_seeded(page):
    rgba, _ = page
    a = printed.apply(rgba, DPI, seed=1)
    assert a.dtype == bool and a.shape == rgba.shape[:2]
    assert np.array_equal(a, printed.apply(rgba, DPI, seed=1))
    assert not np.array_equal(a, printed.apply(rgba, DPI, seed=2))
    assert np.array_equal(printed.apply(rgba, DPI), printed.apply(rgba, DPI, seed=0))


def test_ink_coverage_stays_close_to_clean(page):
    rgba, mask = page
    got = printed.apply(rgba, DPI).sum()
    assert abs(got - mask.sum()) < 0.15 * mask.sum()


def test_edges_are_ragged_but_hairline_survives(page):
    rgba, mask = page
    out = printed.apply(rgba, DPI)
    assert (out != mask).any()
    assert out[148:154, 20:270].any(axis=0).mean() > 0.95  # the hairline is still there along its length
    # No noise and no spread reproduces the clean render.
    clean = printed.apply(rgba, DPI, printed.Printed(ink_spread_mm=0, edge_noise=0))
    assert np.array_equal(clean, mask)
