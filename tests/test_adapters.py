"""Adapters: duck-typed fakes always; the real libraries when they are installed."""

from types import SimpleNamespace

import numpy as np
import pytest

import inkorbit as ink
from inkorbit import adapters


class FakeQuantity:
    """Just enough of astropy's Quantity: a value, a unit and to_value()."""

    _TO_SI = {"km": 1e3, "m": 1.0, "rad": 1.0, "deg": np.pi / 180, "": 1.0, "km3 / s2": 1e9, "m3 / s2": 1.0}

    def __init__(self, value, unit):
        self.value, self.unit = np.asarray(value, dtype=float), unit

    def to_value(self, unit):
        return self.value * self._TO_SI[self.unit] / self._TO_SI[unit]


def _fake_orbit(name="Earth"):
    attractor = SimpleNamespace(name=name, R=FakeQuantity(6_378_136.6, "m"), k=FakeQuantity(3.986004418e14, "m3 / s2"))
    return SimpleNamespace(
        a=FakeQuantity(7000.0, "km"),
        ecc=FakeQuantity(0.1, ""),
        inc=FakeQuantity(28.0, "deg"),
        raan=FakeQuantity(10.0, "deg"),
        argp=FakeQuantity(20.0, "deg"),
        nu=FakeQuantity(30.0, "deg"),
        attractor=attractor,
    )


def test_twobody_orbit_to_elements_and_body():
    orbit = _fake_orbit()
    el = adapters.as_elements(orbit)
    assert np.isclose(el.a, 7000.0) and np.isclose(el.e, 0.1)
    assert np.isclose(el.i, np.radians(28.0)) and np.isclose(el.nu, np.radians(30.0))
    body = adapters.body_of(orbit)
    assert body.apsis_names == ("perigee", "apogee")
    assert np.isclose(body.radius_km, 6378.1366) and np.isclose(body.mu_km3_s2, 398600.4418)


def test_elements_pass_through_and_bad_input():
    el = ink.Elements(a=1.0)
    assert adapters.as_elements(el) is el
    with pytest.raises(TypeError, match="cannot read orbital elements"):
        adapters.as_elements("not an orbit")


def test_positions_from_arrays_quantities_and_skyfield_like():
    pts = np.arange(12.0).reshape(4, 3)
    assert np.array_equal(adapters.as_positions(pts), pts)
    assert np.array_equal(adapters.as_positions(pts.T), pts)  # (3, N) is transposed
    assert np.allclose(adapters.as_positions(FakeQuantity(pts * 1000, "m")), pts)
    skyfield_like = SimpleNamespace(position=SimpleNamespace(km=pts.T))
    assert np.array_equal(adapters.as_positions(skyfield_like), pts)


def test_lat_lon_from_arrays_and_quantities():
    lat, lon = adapters.as_lat_lon([1.0, 2.0], FakeQuantity([np.pi, 0.0], "rad"))
    assert np.allclose(lat, [1.0, 2.0]) and np.allclose(lon, [180.0, 0.0])
    with pytest.raises(TypeError):
        adapters.as_lat_lon([1.0, 2.0])


def test_resolve_body_prefers_explicit_then_orbit_then_earth():
    assert adapters.resolve_body("moon").name == "Moon"
    assert adapters.resolve_body(None, [_fake_orbit("Mars")]).name == "Mars"
    assert adapters.resolve_body(None, [ink.Elements(a=1.0)]).name == "Earth"


def test_diagram_accepts_a_twobody_orbit():
    plate = ink.diagram("orbit", _fake_orbit())
    assert plate.scene.notes[0] == "a = 7,000 km"


@pytest.mark.parametrize("library", ["hapsira", "poliastro"])
def test_real_twobody_orbit(library):
    pytest.importorskip(library)
    u = pytest.importorskip("astropy.units")
    bodies = pytest.importorskip(f"{library}.bodies")
    twobody = pytest.importorskip(f"{library}.twobody")
    orbit = twobody.Orbit.from_classical(
        bodies.Earth, 7000 * u.km, 0.1 * u.one, 28 * u.deg, 10 * u.deg, 20 * u.deg, 30 * u.deg
    )
    el = adapters.as_elements(orbit)
    assert np.isclose(el.a, 7000.0) and np.isclose(el.argp, np.radians(20.0))
    assert adapters.body_of(orbit).apsis_names == ("perigee", "apogee")
    assert ink.diagram("orbit", orbit).scene.items


def test_real_skyfield_ground_track():
    api = pytest.importorskip("skyfield.api")
    ts = api.load.timescale()
    line1 = "1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9000"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391428000"
    sat = api.EarthSatellite(line1, line2, "ISS", ts)
    position = sat.at(ts.utc(2024, 1, 1, 0, range(0, 95)))
    lat, lon = adapters.as_lat_lon(position)
    assert lat.shape == (95,) and np.all(np.abs(lat) <= 52)
    assert adapters.as_positions(position).shape == (95, 3)
    assert ink.diagram("ground_track", position).scene.items


def test_real_astropy_quantities():
    u = pytest.importorskip("astropy.units")
    assert np.isclose(adapters.length_km(42_164 * u.km), 42_164)
    plate = ink.diagram("hohmann", 6678 * u.km, 42_164_000 * u.m)
    assert plate.scene.notes
