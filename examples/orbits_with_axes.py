"""Several orbits about the Earth on ruled axes, labelled automatically.

Run with ``python examples/orbits_with_axes.py`` to write docs/images/orbits_with_axes.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import inkorbit as ink
from inkorbit import draw
from inkorbit import style as st

R_EARTH = 6378.0  # km
PARKING = ink.Elements(a=R_EARTH + 500)
TRANSFER = ink.Elements(
    a=(2 * R_EARTH + 500 + 35786) / 2,
    e=(35786 - 500) / (2 * R_EARTH + 500 + 35786),
    argp=np.radians(30),
    nu=np.radians(70),
)
ESCAPE = ink.Elements(a=-20000, e=1.6, argp=np.radians(200))


def orbits_scene() -> ink.Scene:
    scene = ink.Scene(caption="Figure 1.2-1  Orbits about the Earth", axes=True, grid=True, units="km")
    draw.body(scene, R_EARTH, label="EARTH")
    draw.orbit(scene, PARKING, role=st.SECONDARY, frame="plane", label="parking orbit")
    draw.orbit(scene, TRANSFER, role=st.TRANSFER, frame="plane", label="transfer orbit")
    draw.orbit(scene, ESCAPE, frame="plane", max_radius=45000, label="escape hyperbola")
    draw.apsides(scene, TRANSFER, body_radius=R_EARTH, frame="plane")
    draw.spacecraft(scene, TRANSFER, frame="plane", label="S-IVB")
    draw.velocity_vector(scene, TRANSFER, frame="plane", length=8000)
    return scene


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "docs" / "images" / "orbits_with_axes.png"
    orbits_scene().render().savefig(out, dpi=200)
    print(f"wrote {out}")
