"""Central bodies: sizes, gravitational parameters and what their apsides are called."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Body:
    name: str
    radius_km: float
    mu_km3_s2: float  # gravitational parameter GM
    apsis_names: tuple[str, str] = ("periapsis", "apoapsis")


SUN = Body("Sun", 695_700.0, 1.32712440018e11, ("perihelion", "aphelion"))
MERCURY = Body("Mercury", 2_439.7, 2.2032e4)
VENUS = Body("Venus", 6_051.8, 3.24859e5)
EARTH = Body("Earth", 6_378.137, 398_600.4418, ("perigee", "apogee"))
MOON = Body("Moon", 1_737.4, 4_902.800066, ("perilune", "apolune"))
MARS = Body("Mars", 3_396.19, 42_828.37)
JUPITER = Body("Jupiter", 71_492.0, 1.26686534e8, ("perijove", "apojove"))
SATURN = Body("Saturn", 60_268.0, 3.7931187e7)
URANUS = Body("Uranus", 25_559.0, 5.793939e6)
NEPTUNE = Body("Neptune", 24_764.0, 6.836529e6)

BODIES = {b.name.lower(): b for b in (SUN, MERCURY, VENUS, EARTH, MOON, MARS, JUPITER, SATURN, URANUS, NEPTUNE)}


def get_body(body: str | Body) -> Body:
    """Look a body up by name (case-insensitive), or pass a :class:`Body` through."""
    if isinstance(body, Body):
        return body
    try:
        return BODIES[body.lower()]
    except KeyError:
        raise ValueError(f"unknown body {body!r}; known: {sorted(BODIES)} (or pass a Body)") from None
