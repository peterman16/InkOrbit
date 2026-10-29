import itertools
import warnings

import numpy as np
import pytest

import inkorbit as ink
from inkorbit import geometry, maps
from inkorbit import style as st
from inkorbit.scene import Arrowhead, Label, Line


def _labels(plate):
    return [i.text for i in plate.scene.items if isinstance(i, Label)]


def test_orbit_diagram_lists_elements_and_marks_nodes_when_inclined():
    el = ink.Elements(a=24_400, e=0.72, i=np.radians(28.5), argp=np.radians(35), nu=np.radians(125))
    plate = ink.diagram("orbit", el)
    assert plate.scene.notes[:2] == ["a = 24,400 km", "e = 0.720"]
    assert {"perigee", "apogee", "ascending node", "ω", "ν"} <= set(_labels(plate))
    flat = ink.diagram("orbit", ink.Elements(a=24_400, e=0.72), elements_table=False)
    assert "ascending node" not in _labels(flat) and not flat.scene.notes


def test_orbit_diagram_names_apsides_after_the_body():
    plate = ink.diagram("orbit", ink.Elements(a=3000, e=0.3), body="moon")
    assert {"perilune", "apolune"} <= set(_labels(plate))


def test_orbits_diagram_cycles_roles_and_accepts_arrays():
    pts = ink.Elements(a=9000, e=0.2).points(n=50)
    plate = ink.diagram("orbits", ink.Elements(a=7000), ink.Elements(a=20000, e=0.6), pts, names=["a", "b", "c"])
    roles = [i.role for i in plate.scene.items if isinstance(i, Line) and i.role in ("orbit", "secondary", "transfer")]
    assert roles[-3:] == ["orbit", "secondary", "transfer"]
    assert {"a", "b", "c"} <= set(_labels(plate))
    with pytest.raises(ValueError):
        ink.diagram("orbits")


def test_hohmann_delta_v_matches_textbook_leo_to_geo():
    plate = ink.diagram("hohmann", 6678, 42_164)
    notes = dict(n.split(" = ") for n in plate.scene.notes)
    assert notes["total Δv"] == "3.893 km/s"  # LEO (300 km) to GEO is the classic ~3.9 km/s
    assert notes["time of flight"] == "5.28 h"


def test_hohmann_initial_orbit_hugging_the_body_has_no_arrows():
    items = ink.diagram("hohmann", 6678, 42_164).scene.items
    # Arrowheads follow the line they belong to.
    initial, final, transfer = (
        sum(1 for _ in itertools.takewhile(lambda i: isinstance(i, Arrowhead), items[k + 1 :]))
        for k, item in enumerate(items)
        if isinstance(item, Line) and item.role in (st.SECONDARY, st.TRANSFER)
    )
    assert initial == 0  # 300 km up: within 10% of the Earth's radius
    assert final > 0 and transfer > 0


def test_hohmann_downward_and_errors():
    plate = ink.diagram("hohmann", 42_164, 6678)
    assert plate.scene.notes[0].startswith("Δv1 = -")
    with pytest.raises(ValueError, match="no transfer"):
        ink.diagram("hohmann", 7000, 7000)
    with pytest.raises(ValueError, match="circular"):
        ink.diagram("hohmann", ink.Elements(a=7000, e=0.1), 42_164)


def test_ground_track_stations_and_clipping():
    lon = np.linspace(-170, 530, 300)
    lat = 40 * np.sin(np.radians(lon))
    plate = ink.diagram("ground_track", lat, lon, stations=[("Cape", 28.4, -80.6, 15), ("Perth", -31.9, 115.9)])
    assert plate.scene.notes == ["1. Cape", "2. Perth"]
    for item in plate.scene.items:
        if isinstance(item, Line):
            assert np.all(np.abs(item.points[:, 0]) <= 180 + 1e-9)
            assert np.all(np.abs(item.points[:, 1]) <= 80 + 1e-9)
    stippled = ink.diagram("ground_track", lat, lon, land="stipple")
    assert len(stippled.scene.items) > len(plate.scene.items)


def test_unknown_diagram():
    with pytest.raises(ValueError, match="unknown diagram"):
        ink.diagram("porkchop")


def test_plate_delegates_to_draw():
    plate = ink.Plate("Figure 1")
    plate.body(1.0)
    plate.orbit(ink.Elements(a=2.0, e=0.3))
    assert plate._repr_png_().startswith(b"\x89PNG")
    with pytest.raises(AttributeError):
        plate.not_a_drawing_function  # noqa: B018


def test_split_at_antimeridian_interpolates_the_crossing():
    runs = maps.split_at_antimeridian([170, 175, -175, -170], [0, 10, 20, 30])
    assert len(runs) == 2
    assert np.allclose(runs[0][-1], [180, 15]) and np.allclose(runs[1][0], [-180, 15])


def test_small_circle_radius():
    (ring,) = maps.small_circle(0.0, 0.0, 10.0)
    assert np.allclose(np.hypot(ring[:, 0], ring[:, 1]), 10.0, atol=0.05)


def test_coastlines_load():
    coasts = maps.coastlines()
    assert len(coasts) > 100 and all(c.shape[1] == 2 for c in coasts)
    assert len(maps.land()) > 100


def test_clip_polyline_and_polygon():
    runs = geometry.clip_polyline(np.array([[-2.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 0.5]]), -1, 1, -1, 1)
    assert len(runs) == 2
    assert np.allclose(runs[0], [[-1, 0], [1, 0]])
    poly = geometry.clip_polygon(np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]), -1, 1, -1, 1)
    assert np.allclose(sorted(map(tuple, poly)), [(-1, -1), (-1, 1), (1, -1), (1, 1)])
    assert geometry.clip_polygon(np.array([[5.0, 5.0], [6.0, 5.0], [6.0, 6.0]]), -1, 1, -1, 1) is None


# Seeded, so the same pixels everywhere; the tolerance only absorbs float differences between
# numpy/matplotlib builds, which can flip a few threshold pixels.
@pytest.mark.mpl_image_compare(tolerance=1, savefig_kwargs={"dpi": 300})
def test_printed_hohmann_diagram():
    from diagrams_gallery import hohmann_plate

    return hohmann_plate(printed=True)._printed_figure(300)


def test_printed_plate_writes_two_colours(tmp_path):
    from matplotlib.image import imread

    plate = ink.Plate("x", printed=True)
    plate.scene.line([[0, 0], [10, 5]])
    for ext in ("png", "svg", "pdf"):
        plate.savefig(tmp_path / f"p.{ext}", dpi=150)
    px = imread(tmp_path / "p.png")
    assert len(np.unique(px.reshape(-1, px.shape[2]), axis=0)) == 2
    clean = ink.Plate("x")
    assert plate.printed is not None and clean.printed is None and ink.Plate(printed=ink.Printed()).printed
    assert plate._repr_png_()[:4] == b"\x89PNG"


# The KAN-24 mock-up orbit: a Molniya-like ellipse, km and radians.
MOCKUP = ink.Elements(
    a=16_000, e=0.55, i=np.radians(63.4), raan=np.radians(40), argp=np.radians(270), nu=np.radians(150)
)


@pytest.mark.mpl_image_compare(tolerance=2, savefig_kwargs={"dpi": 100})
def test_oblique_orbit_diagram():
    return ink.diagram("orbit", MOCKUP, view=np.radians([20, -20])).render().figure


@pytest.mark.mpl_image_compare(tolerance=2, savefig_kwargs={"dpi": 100})
def test_oblique_orbits_diagram():
    plate = ink.diagram(
        "orbits", ink.Elements(a=10_000), MOCKUP, names=["equatorial", "inclined"], view=np.radians([45, -20])
    )
    return plate.render().figure


def test_oblique_orbit_diagram_drops_angles_and_vectors_but_keeps_the_table():
    plate = ink.diagram("orbit", MOCKUP, view=np.radians([20, -20]))
    assert not {"ω", "ν", "r̄", "v̄"} & set(_labels(plate))
    assert plate.scene.notes == ink.diagram("orbit", MOCKUP).scene.notes


def test_oblique_orbits_diagram_has_no_axes():
    with pytest.raises(ValueError, match="axes"):
        ink.diagram("orbits", MOCKUP, axes=True, view=(0.3, 0.0))


def test_camera_near_the_orbit_plane_warns():
    # For an equatorial orbit the camera's angle above the orbit plane is the elevation itself.
    with pytest.warns(UserWarning, match="edge-on"):
        ink.diagram("orbit", ink.Elements(a=10_000, e=0.2), view=np.radians([4, 30]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ink.diagram("orbit", ink.Elements(a=10_000, e=0.2), view=np.radians([6, 30]))


@pytest.mark.mpl_image_compare(tolerance=2, savefig_kwargs={"dpi": 100})
def test_oblique_hohmann_diagram():
    return ink.diagram("hohmann", 6678, 42164, view=np.radians([25, -20])).render().figure


def test_oblique_hohmann_keeps_the_delta_v_table_without_burn_vectors():
    plate = ink.diagram("hohmann", 6678, 42164, view=np.radians([25, -20]))
    assert plate.scene.notes == ink.diagram("hohmann", 6678, 42164).scene.notes
    assert not {"Δv1", "Δv2"} & set(_labels(plate))
    assert [i.role for i in plate.scene.items if isinstance(i, Line) and i.role == st.TRANSFER]
