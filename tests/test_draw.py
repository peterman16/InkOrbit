import numpy as np

import inkorbit as ink
from inkorbit import draw
from inkorbit import style as st
from inkorbit.scene import Arrowhead, Fill, Label, Line, Marker

ELLIPSE = ink.Elements(a=2.6, e=0.55, nu=np.radians(110))


def _kinds(scene):
    return [type(i) for i in scene.items]


def test_orbit_from_elements_has_arrows_and_label():
    scene = ink.Scene()
    draw.orbit(scene, ELLIPSE, arrows=(0.2, 0.7), label="orbit")
    kinds = _kinds(scene)
    assert kinds.count(Line) == 1 and kinds.count(Arrowhead) == 2 and kinds.count(Label) == 1
    assert scene.items[-1].alternatives is not None


def test_orbit_from_3d_positions_is_projected():
    scene = ink.Scene()
    pts = ELLIPSE.points(n=100)
    pts3 = np.column_stack([pts, np.zeros(len(pts))])
    line = draw.orbit(scene, pts3, arrows=())
    assert line.points.shape == (100, 2)


def test_body_draws_night_grid_and_outline():
    scene = ink.Scene()
    draw.body(scene, 2.0, center=(1.0, 1.0), label="EARTH")
    assert isinstance(scene.items[0], Fill)
    roles = {i.role for i in scene.items if isinstance(i, Line)}
    assert roles == {st.GRATICULE, st.OUTLINE}
    for item in scene.items:
        if isinstance(item, Line):
            assert np.all(np.hypot(*(item.points - 1.0).T) <= 2.0 + 1e-9)


def test_body_without_sun_has_no_night_side():
    scene = ink.Scene()
    draw.body(scene, sun=None)
    assert Fill not in _kinds(scene)


def test_apse_line_breaks_around_the_body():
    scene = ink.Scene()
    draw.apsides(scene, ELLIPSE, body_radius=1.0)
    apse = [i for i in scene.items if isinstance(i, Line) and i.role == st.APSE]
    assert len(apse) == 2
    for line in apse:
        assert np.all(np.abs(line.points[:, 0]) >= 1.0)
    names = [i.text for i in scene.items if isinstance(i, Label)]
    assert names[:2] == ["perigee", "apogee"]


def test_apsis_names_follow_the_body():
    scene = ink.Scene()
    draw.apsides(scene, ELLIPSE, names="moon", line_label=None)
    assert [i.text for i in scene.items if isinstance(i, Label)] == ["perilune", "apolune"]


def test_hyperbola_has_only_periapsis():
    scene = ink.Scene()
    draw.apsides(scene, ink.Elements(a=-2.0, e=1.5), names="sun")
    assert [i.text for i in scene.items if isinstance(i, Label)] == ["perihelion"]
    assert _kinds(scene).count(Marker) == 1


def test_vectors_and_angle():
    scene = ink.Scene()
    draw.radius_vector(scene, ELLIPSE, body_radius=1.0)
    draw.velocity_vector(scene, ELLIPSE)
    draw.true_anomaly(scene, ELLIPSE)
    draw.spacecraft(scene, ELLIPSE, label="spacecraft")
    labels = [i.text for i in scene.items if isinstance(i, Label)]
    assert labels == ["r̄", "v̄", "ν", "spacecraft"]
    assert _kinds(scene).count(Arrowhead) == 3


def test_nodes_line():
    scene = ink.Scene()
    draw.nodes(scene, ink.Elements(a=2.0, e=0.3, i=0.5, argp=np.radians(40)), body_radius=0.5)
    assert [i.text for i in scene.items if isinstance(i, Label)] == ["ascending node", "descending node"]
