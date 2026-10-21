"""Surface features per body: (lon, lat) polylines in degrees, east longitude in [-180, 180)."""

from __future__ import annotations

import numpy as np

from .bodies import EARTH, Body, get_body
from .maps import coastlines


def features(body: str | Body) -> list[np.ndarray]:
    """Surface polylines of ``body`` as (N, 2) arrays of (longitude, latitude); empty if none are known."""
    return coastlines() if get_body(body) == EARTH else []
