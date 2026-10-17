import numpy as np
import pytest
from diagrams_gallery import PLATES
from orbits_with_axes import orbits_scene

import inkorbit as ink


@pytest.mark.mpl_image_compare(tolerance=2, savefig_kwargs={"dpi": 100})
def test_orbits_with_axes():
    return orbits_scene().render().figure


@pytest.mark.parametrize("name", sorted(PLATES))
@pytest.mark.mpl_image_compare(tolerance=2, savefig_kwargs={"dpi": 100})
def test_diagram(name):
    return PLATES[name]().render().figure


def test_data_axes_match_scene_coordinates():
    scene = ink.Scene()
    scene.line([[0, 0], [10, 5]])
    fig = scene.render()
    lo, hi = fig.data_ax.get_xlim()
    assert lo < 0 < 10 < hi
    # A data point lands at the same page position through either axes.
    page_pt = fig.drawing.to_page(np.array([10.0, 5.0]))
    disp_data = fig.data_ax.transData.transform([10.0, 5.0])
    disp_page = fig.page_ax.transData.transform(page_pt)
    assert np.allclose(disp_data, disp_page, atol=0.5)


def test_png_repr():
    scene = ink.Scene()
    scene.line([[0, 0], [1, 1]])
    assert scene._repr_png_().startswith(b"\x89PNG")


def test_unknown_renderer():
    with pytest.raises(ValueError, match="unknown renderer"):
        ink.Scene().render(renderer="plotter")


def test_to_array_matches_page_and_inks_strokes():
    scene = ink.Scene()
    scene.line([[0, 0], [10, 0]], role="orbit")
    fig = scene.render()
    arr = fig.to_array(dpi=100)
    page = fig.drawing.page
    assert arr.dtype == np.uint8 and arr.shape[2] == 4
    assert abs(arr.shape[1] - page.width_mm / 25.4 * 100) <= 1
    assert abs(arr.shape[0] - page.height_mm / 25.4 * 100) <= 1
    assert tuple(arr[0, 0]) == (255, 255, 255, 255)  # blank corner is paper
    x, y = (fig.drawing.to_page(np.array([5.0, 0.0])) / 25.4 * 100).astype(int)
    assert arr[arr.shape[0] - 1 - y, x, 0] < 40  # on the stroke
