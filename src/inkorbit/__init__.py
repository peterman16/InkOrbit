"""inkorbit: orbital diagrams in the style of 1960s black-and-white technical print."""

from . import geometry, style
from .layout import PAGES, Page, layout
from .lettering import letter
from .render import render
from .scene import Scene
from .style import DEFAULT_STYLE, Pen, Style

__all__ = [
    "DEFAULT_STYLE",
    "PAGES",
    "Page",
    "Pen",
    "Scene",
    "Style",
    "geometry",
    "layout",
    "letter",
    "render",
    "style",
]

__version__ = "0.0.1"
