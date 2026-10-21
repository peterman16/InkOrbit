from __future__ import annotations

from dataclasses import replace

from inkorbit.style import Style


def test_region_and_crater_stipple_defaults():
    style = Style()
    assert (style.region_stipple_mm, style.region_core_stipple_mm, style.crater_shadow_stipple_mm) == (0.7, 0.4, 0.3)


def test_region_and_crater_stipple_can_be_overridden():
    style = replace(Style(), region_stipple_mm=1.0, region_core_stipple_mm=0.5, crater_shadow_stipple_mm=0.2)
    assert (style.region_stipple_mm, style.region_core_stipple_mm, style.crater_shadow_stipple_mm) == (1.0, 0.5, 0.2)
    assert Style(region_stipple_mm=0.9).region_stipple_mm == 0.9


def test_ring_stipple_default():
    assert Style().ring_stipple_mm == (0.9, 0.6, 0.38)
