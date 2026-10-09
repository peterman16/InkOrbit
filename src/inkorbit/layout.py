"""Put a scene on a page.

Layout resolves every physical size, converts data coordinates to page
millimetres and turns text, arrowheads, markers and fill patterns into plain
ink: strokes, solid polygons and dots. Renderers only ever draw those three
things, which is what keeps them small and interchangeable.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from . import patterns
from .lettering import letter
from .scene import Arrowhead, Fill, Line, Marker, Scene, Text
from .style import DEFAULT_STYLE, Style

# --- Pages -----------------------------------------------------------------------


@dataclass(frozen=True)
class Page:
    width_mm: float
    height_mm: float
    margin_mm: float = 8.0


PAGES: dict[str, Page] = {
    "figure": Page(160, 120),
    "wide": Page(180, 100),
    "column": Page(85, 70, 4.0),
    "square": Page(120, 120),
    "a4": Page(210, 297, 20.0),
    "letter": Page(215.9, 279.4, 20.0),
}


def as_page(page: str | Page | tuple[float, float]) -> Page:
    if isinstance(page, Page):
        return page
    if isinstance(page, str):
        try:
            return PAGES[page]
        except KeyError:
            raise ValueError(
                f"unknown page {page!r}; choose from {sorted(PAGES)} or pass (width_mm, height_mm)"
            ) from None
    width, height = page
    return Page(float(width), float(height))


# --- Ink primitives --------------------------------------------------------------


@dataclass(frozen=True)
class Stroke:
    points: np.ndarray  # page mm
    width_mm: float
    dash_mm: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Solid:
    polygon: np.ndarray  # page mm
    paper: bool = False  # filled with paper colour instead of ink (knock-out)


@dataclass(frozen=True)
class Dots:
    points: np.ndarray  # page mm
    diameter_mm: float


@dataclass
class Drawing:
    """A scene resolved onto a page: everything in millimetres, origin bottom-left."""

    page: Page
    style: Style
    plot_box: tuple[float, float, float, float]  # x0, y0, x1, y1 in page mm
    data_limits: tuple[float, float, float, float]  # data extent mapped onto plot_box
    scale: float  # page mm per data unit
    items: list[Stroke | Solid | Dots] = field(default_factory=list)

    def to_page(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        x0, y0 = self.plot_box[0], self.plot_box[1]
        return np.column_stack(
            [
                x0 + (pts[..., 0] - self.data_limits[0]) * self.scale,
                y0 + (pts[..., 1] - self.data_limits[2]) * self.scale,
            ]
        ).reshape(pts.shape)


# --- Layout ----------------------------------------------------------------------

_PAD = 0.04  # fraction of the data extent kept clear around the geometry


def layout(scene: Scene, page: str | Page | tuple[float, float] = "figure", style: Style | None = None) -> Drawing:
    style = style or DEFAULT_STYLE
    page = as_page(page)

    bottom = page.margin_mm
    if scene.caption:
        bottom += style.caption_height_mm + style.caption_gap_mm
    box_w = page.width_mm - 2 * page.margin_mm
    box_h = page.height_mm - page.margin_mm - bottom
    if box_w <= 0 or box_h <= 0:
        raise ValueError("page is too small for its margins and caption")

    xmin, xmax, ymin, ymax = scene.bounds()
    if scene.limits is None:
        dx, dy = (xmax - xmin) or 1.0, (ymax - ymin) or 1.0
        xmin, xmax, ymin, ymax = xmin - _PAD * dx, xmax + _PAD * dx, ymin - _PAD * dy, ymax + _PAD * dy
    dw, dh = xmax - xmin, ymax - ymin
    scale = min(box_w / dw, box_h / dh)

    # Centre the data in the box and widen the limits to fill it exactly.
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half_w, half_h = box_w / scale / 2, box_h / scale / 2
    limits = (cx - half_w, cx + half_w, cy - half_h, cy + half_h)
    box = (page.margin_mm, bottom, page.margin_mm + box_w, bottom + box_h)

    drawing = Drawing(page, style, box, limits, scale)
    for item in scene.items:
        _resolve(item, drawing)
    if scene.caption:
        _letter_into(
            drawing,
            scene.caption,
            np.array([page.width_mm / 2, page.margin_mm]),
            style.caption_height_mm,
            "center",
            "baseline",
            0.0,
            style.text_pen_mm,
        )
    return drawing


def _resolve(item, d: Drawing) -> None:
    style = d.style
    if isinstance(item, Line):
        pen = style.pen(item.role)
        pts = d.to_page(item.points)
        if item.closed:
            pts = np.vstack([pts, pts[:1]])
        d.items.append(Stroke(pts, pen.width_mm, pen.dash_mm))
    elif isinstance(item, Fill):
        poly = d.to_page(item.polygon)
        if item.pattern == "solid":
            d.items.append(Solid(poly))
        elif item.pattern == "stipple":
            seed = zlib.crc32(np.ascontiguousarray(item.polygon).tobytes())
            dots = patterns.stipple(poly, item.spacing_mm or style.stipple_spacing_mm, seed=seed)
            if len(dots):
                d.items.append(Dots(dots, style.stipple_dot_mm))
        elif item.pattern == "hatch":
            angle = style.hatch_angle_deg if item.angle_deg is None else item.angle_deg
            for seg in patterns.hatch(poly, item.spacing_mm or style.hatch_spacing_mm, angle):
                d.items.append(Stroke(seg, style.hatch_pen_mm))
        else:
            raise ValueError(f"unknown fill pattern {item.pattern!r}")
    elif isinstance(item, Arrowhead):
        length = item.size_mm or style.arrow_length_mm
        width = length * style.arrow_width_mm / style.arrow_length_mm
        tip = d.to_page(item.at)
        u = np.asarray(item.direction, dtype=float)
        u = u / np.linalg.norm(u)
        n = np.array([-u[1], u[0]])
        base = tip - length * u
        d.items.append(Solid(np.array([tip, base + n * width / 2, base - n * width / 2])))
    elif isinstance(item, Marker):
        _marker(item, d)
    elif isinstance(item, Text):
        pen = style.bold_pen_mm if item.bold else style.text_pen_mm
        anchor = d.to_page(item.at) + np.asarray(item.offset_mm, dtype=float)
        _letter_into(
            d, item.text, anchor, item.height_mm or style.text_height_mm, item.ha, item.va, item.rotation_deg, pen
        )
    else:
        raise TypeError(f"cannot lay out {type(item).__name__}")


def _circle(center: np.ndarray, radius: float, n: int = 48) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center + radius * np.column_stack([np.cos(t), np.sin(t)])


def _marker(item: Marker, d: Drawing) -> None:
    size = item.size_mm or d.style.marker_mm
    c = d.to_page(item.at)
    if item.kind == "dot":
        d.items.append(Solid(_circle(c, size / 2)))
    elif item.kind == "open":
        ring = _circle(c, size / 2)
        d.items.append(Solid(ring, paper=True))
        d.items.append(Stroke(np.vstack([ring, ring[:1]]), d.style.pen("outline").width_mm))
    elif item.kind == "star":
        k = np.arange(10)
        r = np.where(k % 2 == 0, size, size * 0.4)
        a = np.pi / 2 + k * np.pi / 5
        d.items.append(Solid(c + np.column_stack([r * np.cos(a), r * np.sin(a)])))
    elif item.kind == "cross":
        h = size / 2
        w = d.style.pen("outline").width_mm
        d.items.append(Stroke(np.array([c + (-h, -h), c + (h, h)]), w))
        d.items.append(Stroke(np.array([c + (-h, h), c + (h, -h)]), w))
    else:
        raise ValueError(f"unknown marker kind {item.kind!r}")


_VA_SHIFT = {"baseline": 0.0, "bottom": 0.3, "center": -0.5, "top": -1.0}
_HA_SHIFT = {"left": 0.0, "center": -0.5, "right": -1.0}


def _letter_into(
    d: Drawing, text: str, anchor: np.ndarray, height: float, ha: str, va: str, rotation_deg: float, pen_mm: float
) -> None:
    lettering = letter(text, height, d.style.font)
    shift = np.array([_HA_SHIFT[ha] * lettering.width, _VA_SHIFT[va] * height])
    a = np.radians(rotation_deg)
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    for stroke in lettering.strokes:
        d.items.append(Stroke((stroke + shift) @ rot + anchor, pen_mm))
