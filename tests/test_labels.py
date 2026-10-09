import numpy as np

import inkorbit as ink
from inkorbit import labels
from inkorbit.layout import nice_ticks


def test_label_avoids_a_line_through_its_anchor():
    occ = labels.Occupancy(100, 100)
    occ.stamp_polyline(np.array([[0.0, 50.0], [100.0, 50.0]]), 0.6)  # horizontal line
    occ.stamp_polyline(np.array([[50.0, 0.0], [50.0, 100.0]]), 0.6)  # vertical line
    _, box, _, _ = labels.place(occ, [np.array([50.0, 50.0])], 10.0, 2.5)
    assert occ.cost(box) == 0


def test_preferred_direction_wins_when_free():
    occ = labels.Occupancy(100, 100)
    _, box, _, _ = labels.place(occ, [np.array([50.0, 50.0])], 10.0, 2.5, prefer="sw")
    assert box[2] < 50 and box[3] < 50


def test_labels_at_one_point_do_not_overlap_each_other():
    occ = labels.Occupancy(100, 100)
    boxes = []
    for k in range(6):
        lettering = ink.letter(f"label {k}", 2.6)
        _, box, _, _ = labels.place(occ, [np.array([50.0, 50.0])], lettering.width, 2.6)
        occ.stamp_box(box)
        boxes.append(box)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def test_labels_can_be_switched_off():
    scene = ink.Scene()
    scene.line([[0, 0], [1, 1]])
    scene.label("hello", (0.5, 0.5))
    with_labels = len(ink.layout(scene).items)
    scene.labels = False
    assert len(ink.layout(scene).items) == 1 < with_labels


def test_alternative_anchor_used_when_first_is_crowded():
    occ = labels.Occupancy(100, 100)
    occ.stamp_box((30, 30, 70, 70), labels.INK)
    crowded, free = np.array([50.0, 50.0]), np.array([10.0, 10.0])
    anchor, box, _, _ = labels.place(occ, [crowded, free], 8.0, 2.5)
    assert np.array_equal(anchor, free) and occ.cost(box) == 0


def test_nice_ticks():
    assert nice_ticks(0, 10).tolist() == [0, 2, 4, 6, 8, 10]
    assert nice_ticks(-0.33, 0.33).tolist() == [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    assert nice_ticks(-61000, 43000).tolist() == [-60000, -40000, -20000, 0, 20000, 40000]
