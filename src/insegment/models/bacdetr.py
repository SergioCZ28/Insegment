"""BacDETR adapter for Insegment.

Wraps HiTMicTools ScSegmenter (BacDETR mode) so it can be used via Insegment's
plugin system.

Usage:
    insegment serve --model insegment.models.bacdetr:BacDETRSegmenter \\
        --image-dir <folder>

The `hitmictools` conda env must be active so HiTMicTools is importable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseSegmenter, SegmentationResult

DEFAULT_CHECKPOINT = os.environ.get("BACDETR_CHECKPOINT", "")
_CLASS_NAMES = {0: "single-cell", 1: "clump", 2: "debris"}


class BacDETRSegmenter(BaseSegmenter):
    """Adapter wrapping HiTMicTools ScSegmenter in BacDETR mode.

    Defaults match the v16b production inference config bundled in
    model_collection_scsegm_v16b.zip (patch 432, overlap 0.40, NMS 0.3,
    score 0.3).

    Post-processing:
        min_area: adapter-level instance-area filter applied AFTER the
            pipeline's built-in MIN_MASK_AREA=5 floor in scsegmenter.py.
            Drops any instance whose mask is smaller than this many
            pixels. Default 20. Set to 0 to disable.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        score_threshold: float = 0.30,
        nms_iou: float = 0.30,
        patch_size: int = 432,
        overlap_ratio: float = 0.40,
        batch_size: int = 32,
        min_area: int = 20,
    ) -> None:
        from HiTMicTools.model_components.scsegmenter import ScSegmenter

        ckpt = checkpoint or DEFAULT_CHECKPOINT
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"BacDETR checkpoint not found: {ckpt}")

        if min_area < 0:
            raise ValueError(f"min_area must be >= 0, got {min_area}")

        self._segmenter = ScSegmenter(
            model_path=ckpt,
            patch_size=patch_size,
            overlap_ratio=overlap_ratio,
            score_threshold=score_threshold,
            nms_iou=nms_iou,
            clump_merge_min_overlap=250,
            priority_overlap_fraction=0.5,
            temporal_buffer_size=1,
            batch_size=batch_size,
            mask_threshold=0.5,
            class_dict=_CLASS_NAMES,
            model_type="bacdetr",
        )
        self._batch_size = batch_size
        self._min_area = int(min_area)

    @property
    def class_names(self) -> dict[int, str]:
        return dict(_CLASS_NAMES)

    def predict(self, image: np.ndarray) -> SegmentationResult:
        masks, bboxes_list, class_ids_list, scores_list = self._segmenter.predict(
            image,
            channel_index=0,
            temporal_buffer_size=1,
            batch_size=self._batch_size,
            normalize_to_255=True,
            output_shape="HW",
        )

        if isinstance(bboxes_list, list) and len(bboxes_list) > 0:
            bboxes = bboxes_list[0]
            class_ids = class_ids_list[0]
            scores = scores_list[0]
        else:
            bboxes = bboxes_list
            class_ids = class_ids_list
            scores = scores_list

        if masks.ndim == 3:
            masks = masks[0]

        masks = np.asarray(masks)
        bboxes = np.asarray(bboxes, dtype=np.float32)
        class_ids = np.asarray(class_ids, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float32)

        if self._min_area > 0 and len(bboxes) > 0:
            masks, bboxes, class_ids, scores = _filter_by_min_area(
                masks, bboxes, class_ids, scores, self._min_area
            )

        return SegmentationResult(
            masks=masks,
            bboxes=bboxes,
            class_ids=class_ids,
            scores=scores,
        )


def _filter_by_min_area(
    masks: np.ndarray,
    bboxes: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop instances whose mask area is below min_area.

    Convention (from SegmentationResult): `masks` is an (H, W) int array
    where 0 = background and each positive integer i is a unique instance
    ID corresponding to bboxes[i-1], class_ids[i-1], scores[i-1].

    Returns the filtered arrays with the mask relabeled to consecutive
    1..M so the convention is preserved.
    """
    n = len(bboxes)
    # Count pixels per instance label (0 = background, 1..n = instances).
    counts = np.bincount(masks.ravel(), minlength=n + 1)
    # Keep instance i (0-indexed in bboxes) if its mask label (i+1) has enough pixels.
    keep = counts[1 : n + 1] >= min_area

    if keep.all():
        return masks, bboxes, class_ids, scores

    bboxes = bboxes[keep]
    class_ids = class_ids[keep]
    scores = scores[keep]

    # Relabel mask: old_label -> new_label via LUT (0 for dropped instances).
    lut = np.zeros(n + 1, dtype=masks.dtype)
    new_label = 1
    for old_label in range(1, n + 1):
        if keep[old_label - 1]:
            lut[old_label] = new_label
            new_label += 1
    masks = lut[masks]

    return masks, bboxes, class_ids, scores
