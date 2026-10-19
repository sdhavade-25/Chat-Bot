import pymupdf

import rag


def test_overlapping_boxes_are_merged_into_one():
    a = pymupdf.Rect(0, 0, 100, 50)
    b = pymupdf.Rect(98, 10, 150, 60)   # overlaps a
    c = pymupdf.Rect(300, 0, 400, 50)   # far away
    merged = rag._merge_rects([a, b, c])
    assert len(merged) == 2
    big = max(merged, key=lambda r: r.get_area())
    assert (big.x0, big.y0, big.x1, big.y1) == (0, 0, 150, 60)


def test_disjoint_boxes_stay_separate():
    rects = [pymupdf.Rect(0, 0, 10, 10), pymupdf.Rect(100, 100, 110, 110)]
    assert len(rag._merge_rects(rects)) == 2


def test_single_box_passes_through():
    rects = [pymupdf.Rect(5, 5, 50, 50)]
    assert rag._merge_rects(rects) == rects
