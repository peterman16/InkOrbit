"""Proof of style: an elliptical orbit about the Earth, built from inkorbit's primitives.

Run with ``python examples/proof_of_style.py`` to write docs/images/proof_of_style.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import inkorbit as ink
from inkorbit import draw

# Distances in Earth radii.
ORBIT = ink.Elements(a=2.6, e=0.55, nu=np.radians(110))


def proof_scene() -> ink.Scene:
    scene = ink.Scene(caption="Figure 1.1-1  Elliptical orbit about the Earth")

    draw.body(scene, radius=1.0, tilt=np.radians(25), spin=np.radians(-20), sun=np.pi + 0.25)
    draw.orbit(scene, ORBIT, arrows=(0.4, 0.8))
    draw.apsides(scene, ORBIT, body_radius=1.0)
    draw.radius_vector(scene, ORBIT, body_radius=1.0)
    draw.velocity_vector(scene, ORBIT, length=1.0)
    draw.spacecraft(scene, ORBIT, label="spacecraft")
    draw.true_anomaly(scene, ORBIT, radius=1.5)

    # Direction of the Sun, Apollo-report style.
    scene.vector((-2.6, 2.55), (-3.9, 2.55), ink.style.SECONDARY)
    scene.text("SUN", (-2.6, 2.55), va="center", offset_mm=(1.5, 0), bold=True)
    return scene


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "docs" / "images" / "proof_of_style.png"
    proof_scene().render().savefig(out, dpi=200)
    print(f"wrote {out}")
