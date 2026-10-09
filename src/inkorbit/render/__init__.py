"""Renderers turn a laid-out :class:`~inkorbit.layout.Drawing` into an image.

Each renderer only needs to draw three things: strokes, solid polygons and
dots, all in page millimetres. Matplotlib is the only renderer for now.
"""

from __future__ import annotations

from typing import Protocol

from ..layout import Drawing, Page, layout
from ..scene import Scene
from ..style import Style


class Renderer(Protocol):
    def draw(self, drawing: Drawing): ...


def _renderers() -> dict[str, type]:
    from .matplotlib import MatplotlibRenderer

    return {"matplotlib": MatplotlibRenderer}


def render(
    scene: Scene,
    page: str | Page | tuple[float, float] = "figure",
    style: Style | None = None,
    renderer: str = "matplotlib",
):
    """Lay out ``scene`` on ``page`` and draw it with the named renderer."""
    available = _renderers()
    try:
        backend = available[renderer]()
    except KeyError:
        raise ValueError(f"unknown renderer {renderer!r}; available: {sorted(available)}") from None
    return backend.draw(layout(scene, page, style))
