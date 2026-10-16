import numpy as np
import pytest

import inkorbit as ink
from inkorbit import geometry, maps
from inkorbit.scene import Label, Line


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


def _printed_hohmann():
    from diagrams_gallery import hohmann_plate

    plate = hohmann_plate()
    plate.printed = ink.Printed()
    return plate


# Seeded, so the same pixels everywhere; the tolerance only absorbs float differences between
# numpy/matplotlib builds, which can flip a few threshold pixels.
@pytest.mark.mpl_image_compare(tolerance=1, savefig_kwargs={"dpi": 300})
def test_printed_hohmann_diagram():
    return _printed_hohmann()._printed_figure(300)


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
