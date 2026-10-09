"""Single-stroke lettering from the Hershey vector fonts.

The Hershey fonts were digitised by A. V. Hershey at the U.S. Naval Weapons
Laboratory in 1967 for vector plotters, so they are the period's own drafting
lettering. Text is turned into pen strokes, which means it is inked exactly
like every other line and renders identically on every backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from HersheyFonts import HersheyFonts

# Hershey font units: cap line at y=-12, baseline at y=9 (y grows downward).
_CAP_LINE = -12.0
_BASE_LINE = 9.0
_CAP_HEIGHT = _BASE_LINE - _CAP_LINE

# Unicode Greek -> key in the Hershey "greek" font.
_GREEK = dict(
    zip(
        "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
        "abgdezhqiklmnxoprstufcywABGDEZHQIKLMNXOPRSTUFCYW",
        strict=True,
    )
)
_COMBINING_MACRON = "\u0304"  # draws an overbar on the previous letter, e.g. "r\u0304"
_DEGREE = "°"


@cache
def _font(name: str) -> HersheyFonts:
    font = HersheyFonts()
    font.load_default_font(name)
    return font


@dataclass(frozen=True)
class Lettering:
    """Text laid out as strokes, in millimetres, baseline-left at the origin."""

    strokes: tuple[np.ndarray, ...]
    width: float
    height: float  # cap height


def _glyph(font_name: str, char: str):
    if char in _GREEK:
        font_name, char = "greek", _GREEK[char]
    glyphs = list(_font(font_name).glyphs_for_text(char))
    return glyphs[0] if glyphs else None


def letter(text: str, height_mm: float, font: str = "futural") -> Lettering:
    """Lay out one line of text as strokes with the given cap height."""
    scale = height_mm / _CAP_HEIGHT
    strokes: list[np.ndarray] = []
    x = 0.0
    last_extent: tuple[float, float] | None = None  # (left, right) of previous glyph

    for char in text:
        if char == _COMBINING_MACRON:
            if last_extent is not None:
                left, right = last_extent
                y = height_mm * 1.25
                strokes.append(np.array([[left, y], [right, y]]))
            continue
        if char == _DEGREE:
            r = height_mm * 0.18
            t = np.linspace(0, 2 * np.pi, 25)
            cx = x + r * 1.4
            strokes.append(np.column_stack([cx + r * np.cos(t), height_mm - r + r * np.sin(t)]))
            last_extent = (cx - r, cx + r)
            x += r * 3.2
            continue

        glyph = _glyph(font, char)
        if glyph is None:
            raise ValueError(f"no Hershey glyph for {char!r} in font {font!r}")
        for stroke in glyph.strokes:
            pts = np.asarray(stroke, dtype=float)
            xs = x + (pts[:, 0] - glyph.left_offset) * scale
            ys = (_BASE_LINE - pts[:, 1]) * scale
            strokes.append(np.column_stack([xs, ys]))
        width = glyph.char_width * scale
        # Overbars span the drawn glyph, not its advance width.
        (xmin, _), (xmax, _) = glyph.draw_box
        if glyph.strokes:
            last_extent = (x + (xmin - glyph.left_offset) * scale, x + (xmax - glyph.left_offset) * scale)
        else:
            last_extent = (x, x + width)
        x += width

    return Lettering(strokes=tuple(strokes), width=x, height=height_mm)
