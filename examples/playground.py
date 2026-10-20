"""A playground: change the knobs below, run the file, look at playground.png.

Run with ``uv run python examples/playground.py``. Pick a figure with WHAT, then edit the
settings in its section. Distances are in km, angles in degrees (converted below).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import inkorbit as ink
from inkorbit import style as st

# ---------------------------------------------------------------- what to draw
WHAT = "custom"  # "orbit", "orbits", "hohmann", "ground_track" or "custom"
OUT = Path("playground.png")  # written in the folder you run from
DPI = 300  # keep 300 when PRINTED is on; lower pens drown in the grain

# ---------------------------------------------------------------- printed look
PRINTED = False  # True for the default worn print, or a Printed(...) to tune it:
# PRINTED = ink.Printed(ink_spread_mm=0.05, edge_noise=0.2, skew_deg=0.3, speck_density=0.02)

# ---------------------------------------------------------------- the orbit
R_EARTH = 6378.0
A = 24_400  # semi-major axis (negative with E > 1 for a hyperbola)
E = 0.72  # eccentricity
INC = 28.5  # inclination: tilts the globe, the orbit is drawn in its own plane
RAAN = 0.0  # right ascension of the ascending node: turns the globe's meridians
ARGP = 35.0  # argument of periapsis: turns the orbit on the page
NU = 125.0  # true anomaly of the spacecraft

ORBIT = ink.Elements(a=A, e=E, i=np.radians(INC), raan=np.radians(RAAN), argp=np.radians(ARGP), nu=np.radians(NU))


def orbit() -> ink.Plate:
    return ink.diagram(
        "orbit",
        ORBIT,
        caption="Figure 1  Playground orbit",
        name="spacecraft",
        vectors=True,  # radius and velocity vectors
        elements_table=True,  # the table of elements under the figure
        surface=False,  # surface features on the globe
        printed=PRINTED,
    )


def orbits() -> ink.Plate:
    leo = ink.Elements(a=R_EARTH + 400)
    molniya = ink.Elements(a=26_600, e=0.74, i=np.radians(63.4), argp=np.radians(-90))
    return ink.diagram(
        "orbits",
        leo,
        molniya,
        ORBIT,
        names=["low Earth orbit", "Molniya", "playground orbit"],
        caption="Figure 2  Three orbits",
        axes=False,  # ruled axes in km
        grid=False,
        printed=PRINTED,
    )


def hohmann() -> ink.Plate:
    r1, r2 = R_EARTH + 300, 42_164  # initial and final orbit radii
    return ink.diagram("hohmann", r1, r2, caption="Figure 3  Hohmann transfer", delta_v=True, printed=PRINTED)


def ground_track() -> ink.Plate:
    inclination, period_min, revolutions, lon0 = 51.6, 92.0, 3.0, -80.0
    t = np.linspace(0.0, revolutions * period_min * 60, 2000)
    u = 2 * np.pi * t / (period_min * 60)
    i = np.radians(inclination)
    lat = np.degrees(np.arcsin(np.sin(i) * np.sin(u)))
    lon = lon0 + np.degrees(np.arctan2(np.cos(i) * np.sin(u), np.cos(u))) - np.degrees(7.2921159e-5 * t)
    stations = [("Cape Canaveral", 28.4, -80.6, 15), ("Hawaii", 21.3, -157.9, 15)]  # name, lat, lon, cone
    return ink.diagram(
        "ground_track",
        lat,
        lon,
        stations=stations,
        name="revolution 1",
        caption="Figure 4  Ground track",
        printed=PRINTED,
    )


def custom() -> ink.Plate:
    """Build a figure piece by piece. Every inkorbit.draw function is a method on Plate."""
    plate = ink.Plate("Figure 5  Built by hand", axes=True, grid=True, units="km", printed=PRINTED)
    plate.body(R_EARTH, orbit=ORBIT, frame="plane", spin=np.radians(0), body="earth", surface=False, label="EARTH")
    plate.orbit(ORBIT, frame="plane", label="transfer orbit")
    plate.apsides(ORBIT, body_radius=R_EARTH, frame="plane")
    plate.spacecraft(ORBIT, frame="plane", label="S-IVB")
    plate.radius_vector(ORBIT, frame="plane")
    plate.velocity_vector(ORBIT, frame="plane", length=8000)
    # A second orbit in another pen role: ORBIT, TRANSFER, SECONDARY, CONSTRUCTION ...
    parking = ink.Elements(a=R_EARTH + 3000, i=ORBIT.i, raan=ORBIT.raan, argp=ORBIT.argp)
    plate.orbit(parking, role=st.SECONDARY, frame="plane", arrows=(), label="parking orbit")
    return plate


if __name__ == "__main__":
    figures = {"orbit": orbit, "orbits": orbits, "hohmann": hohmann, "ground_track": ground_track, "custom": custom}
    plate = figures[WHAT]()
    plate.savefig(OUT, dpi=DPI)
    print(f"wrote {OUT.resolve()}")
