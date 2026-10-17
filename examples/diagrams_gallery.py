"""The four ready-made diagrams.

Run with ``python examples/diagrams_gallery.py`` to write the images in docs/images/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import inkorbit as ink

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"


def orbit_plate() -> ink.Plate:
    gto = ink.Elements(a=24_400, e=0.72, i=np.radians(28.5), argp=np.radians(35), nu=np.radians(125))
    return ink.diagram("orbit", gto, caption="Figure 3.2-1  Geosynchronous transfer orbit")


def orbits_plate() -> ink.Plate:
    leo = ink.Elements(a=6_778)
    molniya = ink.Elements(a=26_600, e=0.74, argp=np.radians(-90))
    gto = ink.Elements(a=24_400, e=0.72, argp=np.radians(150))
    return ink.diagram(
        "orbits",
        leo,
        molniya,
        gto,
        names=["low Earth orbit", "Molniya", "transfer orbit"],
        caption="Figure 3.3-1  Three orbits about the Earth",
    )


def hohmann_plate() -> ink.Plate:
    return ink.diagram("hohmann", 6_678, 42_164, caption="Figure 4.1-1  Hohmann transfer to geosynchronous orbit")


def circular_ground_track(inclination_deg=31.0, period_min=88.0, revolutions=3.0, lon0=-80.0):
    """A simple circular-orbit ground track over a rotating Earth (for the example only)."""
    t = np.linspace(0.0, revolutions * period_min * 60, 2000)
    u = 2 * np.pi * t / (period_min * 60)  # argument of latitude
    i = np.radians(inclination_deg)
    lat = np.degrees(np.arcsin(np.sin(i) * np.sin(u)))
    lon = lon0 + np.degrees(np.arctan2(np.cos(i) * np.sin(u), np.cos(u))) - np.degrees(7.2921159e-5 * t)
    return lat, lon


def ground_track_plate() -> ink.Plate:
    lat, lon = circular_ground_track()
    stations = [
        ("Cape Canaveral", 28.4, -80.6, 15),
        ("Bermuda", 32.3, -64.8, 15),
        ("Canary Islands", 27.9, -15.4, 15),
        ("Carnarvon", -24.9, 113.7, 15),
        ("Hawaii", 21.3, -157.9, 15),
        ("Guaymas", 27.9, -110.9, 15),
    ]
    return ink.diagram(
        "ground_track",
        lat,
        lon,
        stations=stations,
        name="revolution 1",
        caption="FIGURE 17.  ORBITAL GROUND PROJECTION",
    )


PLATES = {
    "orbit_diagram": orbit_plate,
    "orbits_diagram": orbits_plate,
    "hohmann_diagram": hohmann_plate,
    "ground_track_diagram": ground_track_plate,
}

if __name__ == "__main__":
    for name, make in PLATES.items():
        out = OUT / f"{name}.png"
        make().savefig(out, dpi=200)
        print(f"wrote {out}")
