"""Classical orbital elements, used only to *draw* an orbit (no propagation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Frame = Literal["perifocal", "plane"]


@dataclass(frozen=True)
class Elements:
    """Keplerian elements in any consistent length unit; angles in radians.

    ``a`` is the semi-major axis (negative or positive for hyperbolas, ``e > 1``;
    its sign is ignored). Parabolic orbits (``e == 1``) are not supported.
    ``nu`` is the true anomaly of the object being shown.
    """

    a: float
    e: float = 0.0
    i: float = 0.0
    raan: float = 0.0
    argp: float = 0.0
    nu: float = 0.0

    def __post_init__(self):
        if self.e < 0:
            raise ValueError("eccentricity must be >= 0")
        if np.isclose(self.e, 1.0):
            raise ValueError("parabolic orbits (e == 1) are not supported; use e slightly below or above 1")
        if self.a == 0:
            raise ValueError("semi-major axis must be non-zero")

    @property
    def is_closed(self) -> bool:
        return self.e < 1

    @property
    def p(self) -> float:
        """Semi-latus rectum."""
        return abs(self.a) * abs(1 - self.e**2)

    @property
    def periapsis(self) -> float:
        return self.p / (1 + self.e)

    @property
    def apoapsis(self) -> float:
        return self.p / (1 - self.e) if self.is_closed else np.inf

    @property
    def nu_limit(self) -> float:
        """Largest |true anomaly| reachable: pi for ellipses, the asymptote for hyperbolas."""
        return np.pi if self.is_closed else float(np.arccos(-1 / self.e))

    def radius(self, nu) -> np.ndarray:
        return self.p / (1 + self.e * np.cos(nu))

    def _rotation(self, frame: Frame) -> float:
        if frame == "perifocal":
            return 0.0
        if frame == "plane":
            # Coplanar view: rotate by the longitude of periapsis.
            return self.raan + self.argp
        raise ValueError(f"unknown frame {frame!r}; use 'perifocal' or 'plane'")

    def position(self, nu=None, frame: Frame = "perifocal") -> np.ndarray:
        """2D position(s) at true anomaly ``nu`` (defaults to ``self.nu``)."""
        nu = self.nu if nu is None else nu
        angle = np.asarray(nu, dtype=float) + self._rotation(frame)
        r = self.radius(nu)
        return np.stack([r * np.cos(angle), r * np.sin(angle)], axis=-1)

    def direction_of_motion(self, nu=None, frame: Frame = "perifocal") -> np.ndarray:
        """Unit vector along the velocity at true anomaly ``nu``."""
        nu = self.nu if nu is None else nu
        w = self._rotation(frame)
        v = np.array([-np.sin(nu), self.e + np.cos(nu)])
        c, s = np.cos(w), np.sin(w)
        v = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        return v / np.linalg.norm(v)

    def angle_of(self, nu, frame: Frame = "perifocal") -> float:
        """Polar angle in the drawing of the point at true anomaly ``nu``."""
        return float(nu + self._rotation(frame))

    def points(
        self,
        n: int = 721,
        nu_range: tuple[float, float] | None = None,
        max_radius: float | None = None,
        frame: Frame = "perifocal",
    ) -> np.ndarray:
        """Sampled 2D points of the orbit, in the direction of motion.

        Ellipses default to a full revolution from periapsis. Hyperbolas default
        to the branch out to ``max_radius`` (four periapsis radii if not given).
        """
        if nu_range is None:
            if self.is_closed:
                nu_range = (0.0, 2 * np.pi)
            else:
                r_max = max_radius or 4 * self.periapsis
                # r = p / (1 + e cos nu)  ->  cos nu = (p / r - 1) / e
                limit = float(np.arccos(np.clip((self.p / r_max - 1) / self.e, -1, 1)))
                nu_range = (-limit, limit)
        nu = np.linspace(nu_range[0], nu_range[1], n)
        return self.position(nu, frame)
