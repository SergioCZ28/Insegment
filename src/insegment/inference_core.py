"""Inference pipeline helpers (not routes).

`run_inference` and `_load_saved_annotations` are the two entry points the
Flask routes call to get annotations for an image index. They live here so
the route modules stay thin and test-focused.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

from insegment.state import STATE
from insegment.utils import load_image, mask_to_polygon

logger = logging.getLogger(__name__)


def _load_saved_annotations(index):
    """Load previously saved annotations from `{file_label}_annotations.json`.

    Returns a list of annotations (internal format) or None if no saved file.
    """
    if not STATE.get("output_dir"):
        return None
    if index < 0 or index >= len(STATE.get("images", [])):
        return None
    file_label = STATE["images"][index]["filename"]
    saved_path = Path(STATE["output_dir"]) / f"{file_label}_annotations.json"
    if not saved_path.exists():
        return None
    try:
        with open(saved_path) as f:
            coco = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read saved annotations %s: %s", saved_path, e)
        return None
    annotations = []
    for ann in coco.get("annotations", []):
        annotations.append({
            "id": ann["id"],
            "category_id": ann["category_id"] - 1,  # COCO is 1-indexed
            "bbox": ann["bbox"],
            "area": ann["area"],
            "segmentation": ann["segmentation"],
            "score": 1.0,
            "source": "manual",
        })
    logger.info("Loaded %d saved annotations for index %d", len(annotations), index)
    return annotations


def run_inference(index):
    """Run model inference on image at index and cache results."""
    if index in STATE["annotations"]:
        return STATE["annotations"][index]

    frame = load_image(index)
    if frame is None:
        return None

    filename = STATE["images"][index]["filename"]

    segmenter = STATE.get("segmenter")
    if segmenter is None:
        # No model loaded -- return empty annotations (annotation-only mode)
        h, w = frame.shape[:2]
        saved = _load_saved_annotations(index)
        result = {
            "index": index,
            "filename": filename,
            "width": w,
            "height": h,
            "annotations": saved if saved is not None else [],
            "inference_time": 0,
            "next_id": (max((a["id"] for a in saved), default=-1) + 1) if saved else 0,
        }
        STATE["annotations"][index] = result
        return result

    t0 = time.time()
    result = segmenter.predict(frame)
    elapsed = time.time() - t0

    masks = result.masks
    bboxes = result.bboxes
    class_ids = result.class_ids
    scores = result.scores

    if masks.ndim == 3:
        masks = masks[0]

    # Convert to annotation list
    annotations = []
    n_detections = len(class_ids) if hasattr(class_ids, "__len__") else 0
    for i in range(n_detections):
        inst_id = i + 1
        inst_mask = (masks == inst_id).astype(np.uint8)
        if not inst_mask.any():
            continue

        polygon = mask_to_polygon(inst_mask)
        if polygon is None:
            continue

        x1, y1, x2, y2 = bboxes[i]
        bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        area = float(bbox_xywh[2] * bbox_xywh[3])

        annotations.append({
            "id": i,
            "category_id": int(class_ids[i]),
            "bbox": bbox_xywh,
            "area": area,
            "segmentation": [polygon],
            "score": float(scores[i]),
            "source": "model",
        })

    h, w = frame.shape[:2]
    ann_result = {
        "index": index,
        "filename": filename,
        "width": w,
        "height": h,
        "annotations": annotations,
        "inference_time": round(elapsed, 1),
        "next_id": n_detections,
    }

    saved = _load_saved_annotations(index)
    if saved is not None:
        ann_result["annotations"] = saved
        ann_result["next_id"] = max((a["id"] for a in saved), default=-1) + 1

    STATE["annotations"][index] = ann_result
    return ann_result
