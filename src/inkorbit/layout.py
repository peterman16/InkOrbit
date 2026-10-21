"""Put a scene on a page.

Layout resolves every physical size, converts data coordinates to page
millimetres and turns text, arrowheads, markers and fill patterns into plain
ink: strokes, solid polygons and dots. Renderers only ever draw those three
things, which is what keeps them small and interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import labels, patterns
from .lettering import letter
from .scene import Arrowhead, Fill, Label, Line, Marker, Scene, Text
from .style import AXIS, BORDER, DEFAULT_STYLE, GRID, Style

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
    "map": Page(200, 130),
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
    role: str | None = None  # the line role it came from, if any


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

    left = right = top = page.margin_mm
    bottom = page.margin_mm
    if scene.caption:
        bottom += style.caption_height_mm + style.caption_gap_mm
    notes_layout = None
    if scene.notes:
        notes_layout = _notes_columns(scene.notes, style, page.width_mm - 2 * page.margin_mm)
        bottom += notes_layout[2] + _NOTES_GAP_MM
    if scene.axes:
        # Room for the tick numbers below the border and the units beside them.
        bottom += style.text_height_mm + 3.0
        if scene.units:
            right += letter(scene.units, style.text_height_mm, style.font).width + 2.0
        if scene.numbers_all_sides:
            top += style.text_height_mm + 3.0

    xmin, xmax, ymin, ymax = scene.bounds()
    if scene.limits is None:
        dx, dy = (xmax - xmin) or 1.0, (ymax - ymin) or 1.0
        xmin, xmax, ymin, ymax = xmin - _PAD * dx, xmax + _PAD * dx, ymin - _PAD * dy, ymax + _PAD * dy

    def fit(left: float):
        box_w = page.width_mm - left - right
        box_h = page.height_mm - top - bottom
        if box_w <= 0 or box_h <= 0:
            raise ValueError("page is too small for its margins and caption")
        scale = min(box_w / (xmax - xmin), box_h / (ymax - ymin))
        if scene.limits is not None:
            # Explicit limits are kept exactly: shrink the box to their shape,
            # centred across the page and sitting on the space reserved below.
            w, h = (xmax - xmin) * scale, (ymax - ymin) * scale
            x0 = left + (box_w - w) / 2
            return (x0, bottom, x0 + w, bottom + h), (xmin, xmax, ymin, ymax), scale
        # Centre the data in the box and widen the limits to fill it exactly.
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        half_w, half_h = box_w / scale / 2, box_h / scale / 2
        limits = (cx - half_w, cx + half_w, cy - half_h, cy + half_h)
        return (left, bottom, left + box_w, bottom + box_h), limits, scale

    box, limits, scale = fit(left)
    if scene.axes:
        # The y numbers need the room their widest label takes; that depends on
        # the ticks, which depend on the fit, so fit twice.
        for _ in range(2):
            widest = max(
                letter(_format_tick(v), style.text_height_mm, style.font).width
                for v in nice_ticks(limits[2], limits[3])
            )
            box, limits, scale = fit(page.margin_mm + widest + 2.5)

    drawing = Drawing(page, style, box, limits, scale)
    if scene.axes or scene.grid:
        _frame(drawing, scene)

    fills = 0
    pending_labels: list[Label] = []
    for item in scene.items:
        if isinstance(item, Label):
            if scene.labels:
                pending_labels.append(item)
            continue
        if isinstance(item, Fill):
            # Seed stipple by position, not by coordinates: float noise across
            # platforms must not reshuffle the dots.
            if item.seed is None:
                item = replace(item, seed=fills)
            fills += 1
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
    if notes_layout is not None:
        _draw_notes(drawing, scene, notes_layout)
    if pending_labels:
        _place_labels(drawing, pending_labels)
    return drawing


_NOTES_GAP_MM = 4.0
_NOTES_LINE = 1.7  # line spacing as a multiple of the text height
_NOTES_COLUMN_GAP_MM = 6.0


def _notes_columns(notes: list[str], style: Style, width: float):
    """(columns, rows, height) for setting the notes in equal columns across ``width``."""
    h = style.text_height_mm
    widest = max(letter(n, h, style.font).width for n in notes) + _NOTES_COLUMN_GAP_MM
    columns = max(1, min(len(notes), int(width // widest)))
    rows = int(np.ceil(len(notes) / columns))
    return columns, rows, rows * _NOTES_LINE * h


def _draw_notes(d: Drawing, scene: Scene, notes_layout) -> None:
    """Legend lines in columns, filled top to bottom then left to right, under the figure."""
    columns, rows, height = notes_layout
    style, h = d.style, d.style.text_height_mm
    x0, y0, x1, _ = d.plot_box
    top = y0 - _NOTES_GAP_MM - (style.text_height_mm + 3.0 if scene.axes else 0.0)
    column_w = (x1 - x0) / columns
    for k, note in enumerate(scene.notes):
        col, row = divmod(k, rows)
        baseline = np.array([x0 + col * column_w, top - h - row * _NOTES_LINE * h])
        _letter_into(d, note, baseline, h, "left", "baseline", 0.0, style.text_pen_mm)


def _place_labels(d: Drawing, items: list[Label]) -> None:
    occupancy = labels.Occupancy(d.page.width_mm, d.page.height_mm)
    clearance = labels.CLEARANCE_MM
    for ink in d.items:
        if isinstance(ink, Stroke):
            if ink.role == GRID:
                occupancy.stamp_polyline(ink.points, ink.width_mm / 2, labels.LIGHT)
            else:
                occupancy.stamp_polyline(ink.points, ink.width_mm / 2 + clearance)
        elif isinstance(ink, Dots):
            occupancy.stamp_points(ink.points, ink.diameter_mm / 2 + clearance)
        elif isinstance(ink, Solid):
            occupancy.stamp_polygon(ink.polygon, clearance)

    # Higher priority first; ties keep the order they were added in.
    for item in sorted(items, key=lambda i: -i.priority):
        height = item.height_mm or d.style.text_height_mm
        lettering = letter(item.text, height, d.style.font)
        anchors = [item.at] if item.alternatives is None else [item.at, *item.alternatives]
        anchor, box, origin, needs_leader = labels.place(
            occupancy, [d.to_page(a) for a in anchors], lettering.width, height, item.prefer
        )
        pen = d.style.bold_pen_mm if item.bold else d.style.text_pen_mm
        for stroke in lettering.strokes:
            d.items.append(Stroke(stroke + origin, pen))
        occupancy.stamp_box(np.add(box, (-clearance, -clearance, clearance, clearance)))
        if needs_leader:
            line = labels.leader(anchor, box)
            d.items.append(Stroke(line, d.style.pen(GRID).width_mm))
            occupancy.stamp_polyline(line, clearance)


def nice_ticks(lo: float, hi: float, target: int = 6) -> np.ndarray:
    """Round tick values (1, 2, 2.5 or 5 times a power of ten) inside [lo, hi]."""
    magnitude = 10 ** np.floor(np.log10((hi - lo) / max(target, 1)))

    def ticks_for(step: float) -> np.ndarray:
        first = np.ceil(lo / step - 1e-9) * step
        ticks = np.arange(first, hi + step * 1e-9, step)
        decimals = max(0, int(-np.floor(np.log10(step))) + 2)
        return np.round(np.round(ticks / step) * step, decimals)  # no 0.30000000000000004

    steps = [m * magnitude for m in (1, 2, 2.5, 5, 10)]
    # Closest to the target count; on a tie, the coarser step.
    best = min(steps, key=lambda step: (abs(len(ticks_for(step)) - target), -step))
    return ticks_for(best)


def _ticks(explicit, lo: float, hi: float) -> np.ndarray:
    if explicit is None:
        return nice_ticks(lo, hi)
    ticks = np.asarray(explicit, dtype=float)
    return ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]


def _format_tick(value: float) -> str:
    text = f"{value:g}"
    return "0" if text == "-0" else text


def _frame(d: Drawing, scene: Scene) -> None:
    """Ruled border with inward ticks and numbers (axes), and/or a light grid."""
    style = d.style
    x0, y0, x1, y1 = d.plot_box
    xmin, xmax, ymin, ymax = d.data_limits
    xt, yt = _ticks(scene.xticks, xmin, xmax), _ticks(scene.yticks, ymin, ymax)
    fmt = scene.tick_format or _format_tick
    xs = x0 + (xt - xmin) * d.scale
    ys = y0 + (yt - ymin) * d.scale

    if scene.grid:
        grid = style.pen(GRID)
        for x in xs:
            d.items.append(Stroke(np.array([[x, y0], [x, y1]]), grid.width_mm, grid.dash_mm, GRID))
        for y in ys:
            d.items.append(Stroke(np.array([[x0, y], [x1, y]]), grid.width_mm, grid.dash_mm, GRID))

    if scene.axes:
        border = style.pen(BORDER)
        tick_pen = style.pen(AXIS).width_mm
        rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
        d.items.append(Stroke(rect, border.width_mm, border.dash_mm))
        tick = 1.5
        for x in xs:
            d.items.append(Stroke(np.array([[x, y0], [x, y0 + tick]]), tick_pen))
            d.items.append(Stroke(np.array([[x, y1], [x, y1 - tick]]), tick_pen))
        for y in ys:
            d.items.append(Stroke(np.array([[x0, y], [x0 + tick, y]]), tick_pen))
            d.items.append(Stroke(np.array([[x1, y], [x1 - tick, y]]), tick_pen))
        h, pen = style.text_height_mm, style.text_pen_mm
        for x, v in zip(xs, xt, strict=True):
            _letter_into(d, fmt(v), np.array([x, y0 - 1.5]), h, "center", "top", 0.0, pen)
            if scene.numbers_all_sides:
                _letter_into(d, fmt(v), np.array([x, y1 + 1.5]), h, "center", "bottom", 0.0, pen)
        for y, v in zip(ys, yt, strict=True):
            _letter_into(d, fmt(v), np.array([x0 - 1.5, y]), h, "right", "center", 0.0, pen)
            if scene.numbers_all_sides:
                _letter_into(d, fmt(v), np.array([x1 + 1.5, y]), h, "left", "center", 0.0, pen)
        if scene.units:
            _letter_into(d, scene.units, np.array([x1 + 1.5, y0 - 1.5]), h, "left", "top", 0.0, pen)


def _resolve(item, d: Drawing) -> None:
    style = d.style
    if isinstance(item, Line):
        pen = style.pen(item.role)
        pts = d.to_page(item.points)
        if item.closed:
            pts = np.vstack([pts, pts[:1]])
        d.items.append(Stroke(pts, pen.width_mm, pen.dash_mm, item.role))
    elif isinstance(item, Fill):
        poly = d.to_page(item.polygon)
        if item.pattern in ("solid", "paper"):
            d.items.append(Solid(poly, paper=item.pattern == "paper"))
        elif item.pattern == "stipple":
            dots = patterns.stipple(poly, item.spacing_mm or style.stipple_spacing_mm, seed=item.seed or 0)
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
