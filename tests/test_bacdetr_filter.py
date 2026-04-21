"""Unit tests for the BacDETR adapter's min-area filter.

The filter is defined as a module-level helper `_filter_by_min_area` so we
can test it without instantiating BacDETRSegmenter (which would import
HiTMicTools and load a checkpoint).
"""

import numpy as np
import pytest

from insegment.models.bacdetr import _filter_by_min_area


def _make_result(instances):
    """Build (masks, bboxes, class_ids, scores) from a list of (label, npix)
    tuples. Each instance gets a square block of `npix` pixels laid out on a
    100x100 canvas. Returns arrays in the convention used by
    SegmentationResult: masks[instance_pixels] = label_index+1.
    """
    H = W = 100
    masks = np.zeros((H, W), dtype=np.int32)
    bboxes = []
    class_ids = []
    scores = []
    # Place blocks on a simple grid.
    x = 0
    for i, (cls, npix) in enumerate(instances, start=1):
        side = int(np.ceil(np.sqrt(npix)))
        y0 = 0
        x0 = x
        # Fill only npix pixels exactly (row-major).
        filled = 0
        for dy in range(side):
            for dx in range(side):
                if filled >= npix:
                    break
                masks[y0 + dy, x0 + dx] = i
                filled += 1
            if filled >= npix:
                break
        bboxes.append([x0, y0, x0 + side, y0 + side])
        class_ids.append(cls)
        scores.append(0.9)
        x += side + 2  # small gap
    return (
        masks,
        np.asarray(bboxes, dtype=np.float32),
        np.asarray(class_ids, dtype=np.int64),
        np.asarray(scores, dtype=np.float32),
    )


class TestFilterByMinArea:
    def test_drops_only_small_instances(self):
        # 3 instances: 10 px, 50 px, 5 px. Threshold 20 should keep only #2.
        masks, bboxes, class_ids, scores = _make_result([(0, 10), (1, 50), (2, 5)])
        m2, b2, c2, s2 = _filter_by_min_area(masks, bboxes, class_ids, scores, 20)
        assert len(b2) == 1
        assert c2.tolist() == [1]
        assert s2.tolist() == [pytest.approx(0.9)]

    def test_relabels_mask_to_consecutive(self):
        # 3 instances: 10 px (drop), 50 px (keep), 30 px (keep). After filter,
        # mask should contain labels 1 and 2 only (renumbered).
        masks, bboxes, class_ids, scores = _make_result([(0, 10), (1, 50), (2, 30)])
        m2, _, _, _ = _filter_by_min_area(masks, bboxes, class_ids, scores, 20)
        unique = set(np.unique(m2).tolist())
        assert unique == {0, 1, 2}

    def test_no_op_when_all_pass(self):
        # All above threshold -> returns arrays unchanged.
        masks, bboxes, class_ids, scores = _make_result([(0, 30), (1, 40)])
        m2, b2, c2, s2 = _filter_by_min_area(masks, bboxes, class_ids, scores, 20)
        assert np.array_equal(m2, masks)
        assert np.array_equal(b2, bboxes)
        assert np.array_equal(c2, class_ids)
        assert np.array_equal(s2, scores)

    def test_drops_all_when_threshold_too_high(self):
        masks, bboxes, class_ids, scores = _make_result([(0, 5), (1, 10)])
        m2, b2, c2, s2 = _filter_by_min_area(masks, bboxes, class_ids, scores, 1000)
        assert len(b2) == 0
        assert len(c2) == 0
        assert len(s2) == 0
        assert m2.sum() == 0  # mask is all background

    def test_threshold_is_inclusive(self):
        # instance with exactly `min_area` pixels must be KEPT.
        masks, bboxes, class_ids, scores = _make_result([(0, 20)])
        m2, b2, _, _ = _filter_by_min_area(masks, bboxes, class_ids, scores, 20)
        assert len(b2) == 1

    def test_preserves_dtype(self):
        masks, bboxes, class_ids, scores = _make_result([(0, 10), (1, 50)])
        m2, b2, c2, s2 = _filter_by_min_area(masks, bboxes, class_ids, scores, 20)
        assert m2.dtype == masks.dtype
        assert b2.dtype == bboxes.dtype
        assert c2.dtype == class_ids.dtype
        assert s2.dtype == scores.dtype
