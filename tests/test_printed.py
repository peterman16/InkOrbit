import numpy as np
import pytest

from inkorbit import printed

DPI = 300


def _rgba(ink_mask):
    out = np.full((*ink_mask.shape, 4), 255, np.uint8)
    out[ink_mask, :3] = 0
    return out


def _plain(**kw):
    """Printed with every effect off except the ones given."""
    off = dict(
        ink_spread_mm=0, edge_noise=0, dropout_density=0, speck_density=0, skew_deg=0, warp_mm=0
    )  # fmt: skip
    return printed.Printed(**{**off, **kw})


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
    clean = printed.apply(rgba, DPI, _plain())
    assert np.array_equal(clean, mask)


def test_specks_on_blank_paper_match_density():
    blank = _rgba(np.zeros((1200, 1200), bool))  # 101.6 mm square at 300 dpi
    params = _plain(speck_density=0.05, speck_mm=0.17)
    out = printed.apply(blank, DPI, params)
    side_mm = 2 * 25.4 / DPI  # 0.17 mm is two pixels
    expected = 0.05 * side_mm**2  # ink fraction of the page
    assert out.mean() == pytest.approx(expected, rel=0.15)


def test_dropouts_open_up_solid_ink(page):
    rgba, mask = page
    params = _plain(dropout_density=5)
    out = printed.apply(rgba, DPI, params)
    assert 0 < (mask & ~out).sum() < 0.2 * mask.sum() and not (out & ~mask).any()


def test_zero_densities_leave_no_grain(page):
    rgba, mask = page
    off = _plain()
    assert np.array_equal(printed.apply(rgba, DPI, off), mask)


def test_no_skew_no_warp_leaves_image_unchanged(page):
    rgba, mask = page
    assert np.array_equal(printed.apply(rgba, DPI, _plain()), mask)


def test_skew_moves_a_point_by_distance_times_angle():
    mask = np.zeros((600, 1000), bool)
    mask[298:302, 898:902] = True  # a dot 400 px right of the page centre
    out = printed.apply(_rgba(mask), DPI, _plain(skew_deg=1.0))
    ys, xs = np.nonzero(out)
    # The scan is turned by +1 degree: content at distance d lands d*theta away, perpendicular to the radius.
    assert ys.mean() - 299.5 == pytest.approx(-400 * np.radians(1.0), abs=1)  # rows grow downwards
    assert xs.mean() == pytest.approx(899.5, abs=1)


def test_warp_moves_content_by_at_most_its_amplitude(page):
    rgba, mask = page
    out = printed.apply(rgba, DPI, _plain(warp_mm=0.5))
    assert (out != mask).any()
    rows = np.nonzero(out.any(axis=1))[0]
    assert abs(rows.min() - 40) <= 0.5 * DPI / 25.4 + 1
