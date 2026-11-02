from __future__ import annotations

from inkorbit.style import ORBIT, Pen, Style


def test_with_pen_replaces_one_role_and_leaves_the_original_alone():
    style = Style()
    thick = style.with_pen(ORBIT, Pen(1.2))
    assert thick.pen(ORBIT) == Pen(1.2) and style.pen(ORBIT) == Pen(0.7)
    assert thick.pen("transfer") == style.pen("transfer")
