"""inkorbit: orbital diagrams in the style of 1960s black-and-white technical print."""

from . import diagrams, draw, geometry, style
from .bodies import Body
from .diagrams import diagram
from .elements import Elements
from .layout import PAGES, Page, layout
from .lettering import letter
from .plate import Plate
from .render import render
from .scene import Scene
from .style import DEFAULT_STYLE, Pen, Style

__all__ = [
    "Body",
    "DEFAULT_STYLE",
    "Elements",
    "PAGES",
    "Page",
    "Pen",
    "Plate",
    "Scene",
    "Style",
    "diagram",
    "diagrams",
    "draw",
    "geometry",
    "layout",
    "letter",
    "render",
    "style",
]

__version__ = "0.0.1"
