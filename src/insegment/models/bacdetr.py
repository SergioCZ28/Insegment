"""BacDETR adapter for Insegment.

Wraps HiTMicTools ScSegmenter (BacDETR mode) so it can be used via Insegment's
plugin system.

Usage:
    insegment serve --model insegment.models.bacdetr:BacDETRSegmenter \\
        --image-dir <folder>

The `hitmictools` conda env must be active so HiTMicTools is importable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseSegmenter, SegmentationResult

DEFAULT_CHECKPOINT = (
    "C:/Users/sergi/ExperimentsWindows/e009_BacDETR/training/checkpoints/"
    "checkpoint_best_v16b_ema.pth"
)
_CLASS_NAMES = {0: "single-cell", 1: "clump", 2: "debris"}


class BacDETRSegmenter(BaseSegmenter):
    """Adapter wrapping HiTMicTools ScSegmenter in BacDETR mode.

    Defaults match the v16b production inference config bundled in
    model_collection_scsegm_v16b.zip (patch 432, overlap 0.40, NMS 0.3,
    score 0.3).
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        score_threshold: float = 0.30,
        nms_iou: float = 0.30,
        patch_size: int = 432,
        overlap_ratio: float = 0.40,
        batch_size: int = 32,
    ) -> None:
        from HiTMicTools.model_components.scsegmenter import ScSegmenter

        ckpt = checkpoint or DEFAULT_CHECKPOINT
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"BacDETR checkpoint not found: {ckpt}")

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

        return SegmentationResult(
            masks=np.asarray(masks),
            bboxes=np.asarray(bboxes, dtype=np.float32),
            class_ids=np.asarray(class_ids, dtype=np.int64),
            scores=np.asarray(scores, dtype=np.float32),
        )
