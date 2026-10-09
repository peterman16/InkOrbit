"""Proof of style: an elliptical orbit about the Earth, drawn by hand from primitives.

Run with ``python examples/proof_of_style.py`` to write docs/images/proof_of_style.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import inkorbit as ink
from inkorbit import geometry as geo
from inkorbit import style as st

# Distances in Earth radii.
A, E = 2.6, 0.55
P = A * (1 - E**2)
R_PERIGEE, R_APOGEE = A * (1 - E), A * (1 + E)


def proof_scene() -> ink.Scene:
    scene = ink.Scene(caption="Figure 1.1-1  Elliptical orbit about the Earth")

    # The Earth: stippled night side, latitude/longitude grid, outline.
    rotation = geo.view_rotation(tilt=np.radians(25), spin=np.radians(-20))
    night = geo.night_side(np.array([-1.0, 0.25, 0.45]))
    scene.fill(night, "stipple")
    for line in geo.graticule(rotation):
        scene.line(line, st.GRID)
    scene.line(geo.circle((0, 0), 1.0), st.OUTLINE)

    # The orbit, with direction arrows along it.
    scene.line(geo.ellipse(A, E), st.ORBIT, arrows=(0.4, 0.8))

    # Line of apsides, broken where it passes the Earth.
    scene.line([[-R_APOGEE - 0.5, 0], [-1.08, 0]], st.APSE)
    scene.line([[1.08, 0], [R_PERIGEE + 0.6, 0]], st.APSE)
    scene.marker((R_PERIGEE, 0), "dot")
    scene.marker((-R_APOGEE, 0), "dot")
    scene.text("perigee", (R_PERIGEE, 0), va="top", offset_mm=(1.5, -1.8))
    scene.text("apogee", (-R_APOGEE, 0), ha="right", va="top", offset_mm=(-1.5, -1.8))
    scene.text("line of apsides", (-R_APOGEE + 0.15, 0), offset_mm=(0, 1.6))

    # The spacecraft at true anomaly nu, with radius and velocity vectors.
    nu = np.radians(110)
    r = P / (1 + E * np.cos(nu))
    sat = r * np.array([np.cos(nu), np.sin(nu)])
    radial = sat / r
    scene.vector(radial * 1.08, sat - radial * 0.08, st.SECONDARY)
    scene.text("r̄", radial * (1.08 + r) / 2, offset_mm=(2.0, 0))

    v_dir = np.array([-np.sin(nu), E + np.cos(nu)])
    v_dir /= np.linalg.norm(v_dir)
    scene.vector(sat, sat + 1.0 * v_dir, st.ORBIT)
    scene.text("v̄", sat + 1.0 * v_dir, offset_mm=(-3.5, -1.0))
    scene.marker(sat, "open", size_mm=1.8)
    scene.text("spacecraft", sat, offset_mm=(2.5, 1.8))

    # True anomaly arc.
    arc = geo.arc((0, 0), 1.5, np.radians(4), nu - np.radians(3))
    scene.line(arc, st.CONSTRUCTION)
    scene.arrowhead(arc[-1], arc[-1] - arc[-2], size_mm=2.0)
    mid = np.radians(55)
    scene.text("ν", 1.7 * np.array([np.cos(mid), np.sin(mid)]), ha="center", va="center", height_mm=3.2)

    # Direction of the Sun, Apollo-report style.
    scene.vector((-2.6, 2.55), (-3.9, 2.55), st.SECONDARY)
    scene.text("SUN", (-2.6, 2.55), va="center", offset_mm=(1.5, 0), bold=True)
    return scene


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "docs" / "images" / "proof_of_style.png"
    proof_scene().render().savefig(out, dpi=200)
    print(f"wrote {out}")
