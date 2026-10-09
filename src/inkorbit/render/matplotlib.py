"""Matplotlib renderer."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Polygon
from matplotlib.transforms import IdentityTransform

from ..layout import Dots, Drawing, Solid, Stroke

PT_PER_MM = 72 / 25.4


@dataclass
class RenderedFigure:
    """A rendered figure. ``data_ax`` is in the scene's data coordinates, so you
    can keep drawing on it with ordinary matplotlib calls."""

    figure: Figure
    page_ax: Axes
    data_ax: Axes
    drawing: Drawing

    def savefig(self, path, dpi: int = 300, **kwargs) -> None:
        kwargs.setdefault("facecolor", self.drawing.style.paper)
        self.figure.savefig(path, dpi=dpi, **kwargs)

    def to_png(self, dpi: int = 150) -> bytes:
        buf = io.BytesIO()
        self.savefig(buf, dpi=dpi, format="png")
        return buf.getvalue()

    def _repr_png_(self) -> bytes:
        return self.to_png()


class MatplotlibRenderer:
    def draw(self, drawing: Drawing) -> RenderedFigure:
        page, style = drawing.page, drawing.style
        fig = Figure(figsize=(page.width_mm / 25.4, page.height_mm / 25.4), facecolor=style.paper)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, page.width_mm)
        ax.set_ylim(0, page.height_mm)
        ax.set_axis_off()
        ax.set_facecolor(style.paper)

        solid_batch: list[Stroke] = []

        def flush():
            if solid_batch:
                ax.add_collection(_collection(solid_batch, style.ink, None))
                solid_batch.clear()

        # Consecutive solid strokes are batched; anything else keeps its order.
        for item in drawing.items:
            if isinstance(item, Stroke) and item.dash_mm is None:
                solid_batch.append(item)
                continue
            flush()
            if isinstance(item, Stroke):
                ax.add_collection(_collection([item], style.ink, item.dash_mm))
            elif isinstance(item, Solid):
                color = style.paper if item.paper else style.ink
                ax.add_patch(Polygon(item.polygon, closed=True, facecolor=color, edgecolor="none", linewidth=0))
            elif isinstance(item, Dots):
                ax.add_collection(_dots(item, style.ink, ax))
            else:
                raise TypeError(f"cannot draw {type(item).__name__}")
        flush()

        x0, y0, x1, y1 = drawing.plot_box
        data_ax = fig.add_axes(
            (x0 / page.width_mm, y0 / page.height_mm, (x1 - x0) / page.width_mm, (y1 - y0) / page.height_mm)
        )
        xmin, xmax, ymin, ymax = drawing.data_limits
        data_ax.set_xlim(xmin, xmax)
        data_ax.set_ylim(ymin, ymax)
        data_ax.set_axis_off()
        data_ax.patch.set_visible(False)
        return RenderedFigure(fig, ax, data_ax, drawing)


def _collection(strokes: list[Stroke], ink: str, dash_mm) -> LineCollection:
    if dash_mm is None:
        linestyle = "solid"
    else:
        # Pattern given in mm; matplotlib wants points, unscaled by line width.
        width_pt = strokes[0].width_mm * PT_PER_MM
        linestyle = (0, tuple(v * PT_PER_MM / width_pt for v in dash_mm))
    coll = LineCollection(
        [s.points for s in strokes],
        linewidths=[s.width_mm * PT_PER_MM for s in strokes],
        colors=ink,
        linestyles=linestyle,
        capstyle="round",
        joinstyle="round",
    )
    return coll


def _dots(dots: Dots, ink: str, ax: Axes) -> PathCollection:
    marker = MarkerStyle("o")
    path = marker.get_path().transformed(marker.get_transform())
    size = (dots.diameter_mm * PT_PER_MM) ** 2
    coll = PathCollection(
        [path],
        sizes=np.full(len(dots.points), size),
        offsets=dots.points,
        offset_transform=ax.transData,
        facecolors=ink,
        edgecolors="none",
        linewidths=0,
    )
    coll.set_transform(IdentityTransform())
    return coll
