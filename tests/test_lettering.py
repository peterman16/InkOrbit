import numpy as np
import pytest

from inkorbit.lettering import letter


def test_width_grows_with_text():
    assert letter("apogee", 2.5).width > letter("apo", 2.5).width > 0


def test_height_scales_strokes():
    small, big = letter("H", 2.0), letter("H", 4.0)
    assert np.isclose(np.vstack(big.strokes)[:, 1].max(), 2 * np.vstack(small.strokes)[:, 1].max())
    assert np.isclose(np.vstack(small.strokes)[:, 1].max(), 2.0)


def test_greek_letters_are_drawn():
    assert letter("ν", 3.0).strokes
    assert letter("ω", 3.0).width > 0


def test_combining_macron_adds_an_overbar():
    plain, barred = letter("r", 2.5), letter("r̄", 2.5)
    assert len(barred.strokes) == len(plain.strokes) + 1
    bar = barred.strokes[-1]
    assert np.allclose(bar[:, 1], bar[0, 1]) and bar[0, 1] > 2.5


def test_degree_sign():
    assert letter("40°", 2.5).width > letter("40", 2.5).width


def test_unknown_character_raises():
    with pytest.raises(ValueError, match="no Hershey glyph"):
        letter("☃", 2.5)
