"""Blueprint: model inference routes (sync + SSE streaming)."""

import json
import queue
import threading
import time
import uuid

import numpy as np
from flask import Blueprint, Response, jsonify, request

from insegment.inference_core import run_inference
from insegment.state import STATE, _inference_jobs
from insegment.utils import load_image, mask_to_polygon

bp = Blueprint("inference", __name__)


@bp.route("/api/detections/<int:index>")
def api_detections(index):
    """Run inference and return detections.

    Query params:
        force=1 -- bypass cache and saved-annotations override, run the model
                   fresh. Requires a model to be loaded.
    """
    force = request.args.get("force") == "1"
    if force:
        if STATE.get("segmenter") is None:
            return jsonify({"error": "No model loaded"}), 400
        if index < 0 or index >= len(STATE.get("images", [])):
            return jsonify({"error": "Invalid image index"}), 400
        STATE["annotations"].pop(index, None)
        # Run inference without the saved-annotations override
        frame = load_image(index)
        if frame is None:
            return jsonify({"error": "Could not load image"}), 404
        t0 = time.time()
        seg_result = STATE["segmenter"].predict(frame)
        elapsed = time.time() - t0
        masks = seg_result.masks
        if masks.ndim == 3:
            masks = masks[0]
        bboxes = seg_result.bboxes
        class_ids = seg_result.class_ids
        scores = seg_result.scores
        annotations = []
        n_detections = len(class_ids) if hasattr(class_ids, "__len__") else 0
        for i in range(n_detections):
            inst_mask = (masks == (i + 1)).astype(np.uint8)
            if not inst_mask.any():
                continue
            polygon = mask_to_polygon(inst_mask)
            if polygon is None:
                continue
            x1, y1, x2, y2 = bboxes[i]
            bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            annotations.append({
                "id": i,
                "category_id": int(class_ids[i]),
                "bbox": bbox_xywh,
                "area": float(bbox_xywh[2] * bbox_xywh[3]),
                "segmentation": [polygon],
                "score": float(scores[i]),
                "source": "model",
            })
        h, w = frame.shape[:2]
        ann_result = {
            "index": index,
            "filename": STATE["images"][index]["filename"],
            "width": w,
            "height": h,
            "annotations": annotations,
            "inference_time": round(elapsed, 1),
            "next_id": n_detections,
        }
        STATE["annotations"][index] = ann_result
        return jsonify(ann_result)

    result = run_inference(index)
    if result is None:
        return jsonify({"error": "Could not process image"}), 404
    return jsonify(result)


@bp.route("/api/inference/start/<int:index>", methods=["POST"])
def api_inference_start(index):
    """Start inference in background thread, return job_id for SSE tracking."""
    if index < 0 or index >= len(STATE["images"]):
        return jsonify({"error": "Invalid image index"}), 400

    # If already cached, return immediately
    if index in STATE["annotations"]:
        return jsonify({"status": "cached", "job_id": None})

    job_id = str(uuid.uuid4())[:8]
    q = queue.Queue()

    def _run():
        try:
            q.put({"stage": "loading", "message": "Loading image..."})
            frame = load_image(index)
            if frame is None:
                q.put({"stage": "error", "message": "Failed to load image"})
                return

            filename = STATE["images"][index]["filename"]
            segmenter = STATE.get("segmenter")

            if segmenter is None:
                h, w = frame.shape[:2]
                result = {
                    "index": index,
                    "filename": filename,
                    "width": w,
                    "height": h,
                    "annotations": [],
                    "inference_time": 0,
                    "next_id": 0,
                }
                STATE["annotations"][index] = result
                q.put({"stage": "done", "message": "No model -- annotation-only mode"})
                return

            q.put({"stage": "running", "message": "Running model inference..."})
            t0 = time.time()
            model_result = segmenter.predict(frame)
            elapsed = time.time() - t0

            masks = model_result.masks
            bboxes = model_result.bboxes
            class_ids = model_result.class_ids
            scores = model_result.scores

            if masks.ndim == 3:
                masks = masks[0]

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
            STATE["annotations"][index] = ann_result
            q.put({"stage": "done", "message": f"Done: {len(annotations)} detections in {round(elapsed, 1)}s"})
        except Exception as exc:
            q.put({"stage": "error", "message": str(exc)})

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _inference_jobs[job_id] = {"queue": q, "thread": t}

    return jsonify({"status": "started", "job_id": job_id})


@bp.route("/api/inference/progress/<job_id>")
def api_inference_progress(job_id):
    """SSE stream for inference progress."""
    if job_id not in _inference_jobs:
        return jsonify({"error": "Unknown job"}), 404

    job = _inference_jobs[job_id]
    q = job["queue"]

    def generate():
        while True:
            try:
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event["stage"] in ("done", "error"):
                    # Cleanup
                    _inference_jobs.pop(job_id, None)
                    break
            except queue.Empty:
                # Heartbeat to keep connection alive
                yield f"data: {json.dumps({'stage': 'heartbeat'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")
