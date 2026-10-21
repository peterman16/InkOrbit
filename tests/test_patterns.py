import numpy as np

from inkorbit import patterns

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


def test_points_in_polygon():
    pts = np.array([[5.0, 5.0], [11.0, 5.0], [-1.0, -1.0], [9.9, 0.1]])
    assert patterns.points_in_polygon(pts, SQUARE).tolist() == [True, False, False, True]


def test_stipple_stays_inside_and_has_expected_density():
    dots = patterns.stipple(SQUARE, 0.5, seed=1)
    assert patterns.points_in_polygon(dots, SQUARE).all()
    assert 300 < len(dots) < 500  # ~ area / spacing^2


def test_stipple_is_deterministic():
    assert np.array_equal(patterns.stipple(SQUARE, 0.5, seed=3), patterns.stipple(SQUARE, 0.5, seed=3))


def test_small_fill_is_stippled_finer_but_never_below_the_floor():
    tiny = SQUARE * 0.06  # 0.6 mm square: about one dot at 0.55 mm
    assert len(patterns.stipple(tiny, 0.55)) >= 5
    assert len(patterns.stipple(SQUARE * 0.01, 0.55)) < 5  # 0.1 mm: too small even at 0.15 mm, stays sparse


def test_hatch_segments_span_the_square():
    segments = patterns.hatch(SQUARE, 1.0, 0.0)
    assert len(segments) == 10
    for seg in segments:
        assert np.allclose(sorted(seg[:, 0]), [0.0, 10.0])
