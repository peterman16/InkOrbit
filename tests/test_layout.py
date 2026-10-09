import numpy as np
import pytest

import inkorbit as ink
from inkorbit.layout import Dots, Solid, Stroke


def _scene():
    scene = ink.Scene()
    scene.line([[0, 0], [10, 0], [10, 5]], ink.style.ORBIT)
    return scene


def test_data_fits_inside_plot_box():
    d = ink.layout(_scene(), "figure")
    x0, y0, x1, y1 = d.plot_box
    pts = d.items[0].points
    assert pts[:, 0].min() >= x0 and pts[:, 0].max() <= x1
    assert pts[:, 1].min() >= y0 and pts[:, 1].max() <= y1


def test_scale_is_uniform():
    d = ink.layout(_scene(), (100, 100))
    pts = d.items[0].points
    assert np.isclose(np.ptp(pts[:, 0]) / 10, np.ptp(pts[:, 1]) / 5)


def test_pen_comes_from_role():
    d = ink.layout(_scene())
    assert d.items[0].width_mm == ink.DEFAULT_STYLE.pen(ink.style.ORBIT).width_mm


def test_caption_reserves_space():
    plain = ink.layout(_scene())
    scene = _scene()
    scene.caption = "Figure 1"
    captioned = ink.layout(scene)
    assert captioned.plot_box[1] > plain.plot_box[1]


def test_items_resolve_to_ink_primitives():
    scene = _scene()
    scene.fill([[0, 0], [5, 0], [5, 5], [0, 5]], "stipple")
    scene.fill([[0, 0], [5, 0], [5, 5]], "hatch")
    scene.marker((1, 1), "open")
    scene.vector((0, 0), (3, 3))
    scene.text("ν = 110°", (2, 2), bold=True)
    kinds = {type(i) for i in ink.layout(scene).items}
    assert kinds == {Stroke, Solid, Dots}


def test_unknown_role_and_page():
    scene = ink.Scene()
    scene.line([[0, 0], [1, 1]], "nonsense")
    with pytest.raises(KeyError, match="nonsense"):
        ink.layout(scene)
    with pytest.raises(ValueError, match="unknown page"):
        ink.layout(_scene(), "napkin")


def test_stipple_does_not_depend_on_float_noise():
    square = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    dots = []
    for poly in (square, square + 1e-13):
        scene = _scene()
        scene.fill(poly, "stipple")
        dots.append(next(i for i in ink.layout(scene).items if isinstance(i, Dots)).points)
    assert np.allclose(dots[0], dots[1], atol=1e-9)
