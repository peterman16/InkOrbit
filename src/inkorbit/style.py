"""The look: pens, line roles, lettering and fill patterns.

All physical sizes are in millimetres on the printed page, the way a
draftsman would specify technical pens and lettering templates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType


@dataclass(frozen=True)
class Pen:
    """A technical pen: nib width and an optional dash pattern (on, off, ...)."""

    width_mm: float
    dash_mm: tuple[float, ...] | None = None


# Line roles. Diagrams say *what* a line is; the style decides how it looks.
ORBIT = "orbit"  # the subject of the figure
TRANSFER = "transfer"  # transfer arcs, future or alternative paths
APSE = "apse"  # apse lines, node lines, reference directions
CONSTRUCTION = "construction"  # construction geometry, radius lines
SECONDARY = "secondary"  # other orbits that are not the subject
OUTLINE = "outline"  # body outlines, terminators
HIDDEN = "hidden"  # geometry behind a body
GRID = "grid"  # chart grids; labels may sit on them
GRATICULE = "graticule"  # latitude/longitude lines drawn on a globe
COASTLINE = "coastline"
AXIS = "axis"
BORDER = "border"

_DASHED = (2.4, 1.2)
_DASH_DOT = (5.0, 1.0, 0.01, 1.0)
_DOTTED = (0.01, 0.9)

_DEFAULT_PENS = {
    ORBIT: Pen(0.7),
    TRANSFER: Pen(0.5, _DASHED),
    APSE: Pen(0.25, _DASH_DOT),
    CONSTRUCTION: Pen(0.35, _DOTTED),
    SECONDARY: Pen(0.35),
    OUTLINE: Pen(0.35),
    HIDDEN: Pen(0.25, _DASHED),
    GRID: Pen(0.18),
    GRATICULE: Pen(0.18),
    COASTLINE: Pen(0.25),
    AXIS: Pen(0.25),
    BORDER: Pen(0.5),
}


@dataclass(frozen=True)
class Style:
    """Everything that decides the look of a figure.

    Use :func:`dataclasses.replace` (or :meth:`with_pen`) to derive variations.
    """

    ink: str = "#000000"
    paper: str = "#ffffff"
    pens: Mapping[str, Pen] = field(default_factory=lambda: MappingProxyType(dict(_DEFAULT_PENS)))

    # Lettering (Hershey single-stroke vector fonts).
    font: str = "futural"
    text_height_mm: float = 2.6
    text_pen_mm: float = 0.25
    bold_pen_mm: float = 0.45
    caption_height_mm: float = 2.8
    caption_gap_mm: float = 6.0

    # Symbols.
    arrow_length_mm: float = 3.4
    arrow_width_mm: float = 1.5
    marker_mm: float = 1.3

    # Fill patterns.
    stipple_spacing_mm: float = 0.55
    stipple_dot_mm: float = 0.2
    hatch_spacing_mm: float = 0.9
    hatch_angle_deg: float = 45.0
    hatch_pen_mm: float = 0.18

    def pen(self, role: str) -> Pen:
        try:
            return self.pens[role]
        except KeyError:
            raise KeyError(f"style has no pen for line role {role!r}; known roles: {sorted(self.pens)}") from None

    def with_pen(self, role: str, pen: Pen) -> Style:
        pens = dict(self.pens)
        pens[role] = pen
        return replace(self, pens=MappingProxyType(pens))


DEFAULT_STYLE = Style()
