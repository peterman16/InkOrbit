"""A Plate is a finished figure: a scene plus the page and style it is printed with."""

from __future__ import annotations

import functools

from . import draw
from .layout import Page
from .render import render
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
    ):
        self.scene = Scene(caption=caption, labels=labels, axes=axes, grid=grid, units=units)
        self.page = page
        self.style = style

    def __getattr__(self, name: str):
        fn = getattr(draw, name, None)
        if name.startswith("_") or not callable(fn) or name not in draw.__all__:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        return functools.partial(fn, self.scene)

    def render(self, renderer: str = "matplotlib"):
        return render(self.scene, page=self.page, style=self.style, renderer=renderer)

    def savefig(self, path, dpi: int = 300, **kwargs) -> None:
        self.render().savefig(path, dpi=dpi, **kwargs)

    def _repr_png_(self) -> bytes:
        return self.render().to_png()
