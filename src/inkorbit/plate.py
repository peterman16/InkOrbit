"""A Plate is a finished figure: a scene plus the page and style it is printed with."""

from __future__ import annotations

import functools
import io

import numpy as np
from matplotlib.colors import to_rgb

from . import draw
from .layout import Page
from .printed import DEFAULT_PRINTED, Printed, apply
from .render import render
from .render.matplotlib import bitmap_figure
from .scene import Scene
from .style import Style


class Plate:
    """A figure ready to print.

    Every function in :mod:`inkorbit.draw` is also a method, drawing on this
    plate's scene: ``plate.orbit(elements)`` is ``draw.orbit(plate.scene, elements)``.
    """

    def __init__(
        self,
        caption: str | None = None,
        *,
        page: str | Page | tuple[float, float] = "figure",
        style: Style | None = None,
        labels: bool = True,
        axes: bool = False,
        grid: bool = False,
        units: str | None = None,
        printed: bool | Printed = False,
    ):
        self.scene = Scene(caption=caption, labels=labels, axes=axes, grid=grid, units=units)
        self.page = page
        self.style = style
        self.printed = DEFAULT_PRINTED if printed is True else printed or None

    def __getattr__(self, name: str):
        fn = getattr(draw, name, None)
        if name.startswith("_") or not callable(fn) or name not in draw.__all__:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        return functools.partial(fn, self.scene)

    def render(self, renderer: str = "matplotlib"):
        return render(self.scene, page=self.page, style=self.style, renderer=renderer)

    def _printed_figure(self, dpi: int):
        """The printed bitmap of this plate as a figure holding exactly its pixels."""
        rendered = self.render()
        style = rendered.drawing.style
        ink = apply(rendered.to_array(dpi), dpi, self.printed)
        colours = np.array([to_rgb(style.paper), to_rgb(style.ink)])  # paper first: indexed by the bool mask
        return bitmap_figure(np.rint(colours[ink.astype(int)] * 255).astype(np.uint8), dpi, style.paper)

    def savefig(self, path, dpi: int = 300, **kwargs) -> None:
        if self.printed is None:
            self.render().savefig(path, dpi=dpi, **kwargs)
        else:
            self._printed_figure(dpi).savefig(path, dpi=dpi, **kwargs)

    def _repr_png_(self) -> bytes:
        buf = io.BytesIO()
        # A printed plate needs 300 dpi, or its 0.18 mm pens are only a pixel wide and the noise eats them.
        self.savefig(buf, dpi=300 if self.printed else 150, format="png")
        return buf.getvalue()
