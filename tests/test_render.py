import numpy as np
import pytest
from diagrams_gallery import PLATES
from orbits_with_axes import orbits_scene
from proof_of_style import proof_scene

import inkorbit as ink


@pytest.mark.mpl_image_compare(tolerance=2, savefig_kwargs={"dpi": 100})
def test_proof_of_style():
    return proof_scene().render().figure


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
