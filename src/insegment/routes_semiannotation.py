"""Blueprint: legacy semi-annotation flow.

Kept for backward compat. Frames live in a folder alongside an
`_annotations.coco.json` file. Users can browse them, optionally run a
model on each, and correct the output.
"""

import json
import time
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request, send_file
from PIL import Image

from insegment.state import STATE
from insegment.utils import _safe_coco_path, build_category_map, mask_to_polygon

bp = Blueprint("semiannotation", __name__)


@bp.route("/api/semiannotation/list")
def api_semiannotation_list():
    """List available semi-annotated frames."""
    frames = STATE.get("semiannotation_frames", {})
    return jsonify({"frames": list(frames.keys()), "dir": STATE.get("semiannotation_dir", "")})


@bp.route("/api/browse_folder")
def api_browse_folder():
    """Open native OS folder picker (for semi-annotation scan)."""
    import threading as _threading

    result = {"path": None}

    def pick():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(title="Select annotation folder")
            root.destroy()
            result["path"] = path if path else None
        except Exception:
            result["path"] = None

    t = _threading.Thread(target=pick)
    t.start()
    t.join(timeout=120)

    if result["path"]:
        return jsonify({"status": "ok", "path": result["path"]})
    return jsonify({"status": "cancelled"})


@bp.route("/api/semiannotation/scan", methods=["POST"])
def api_semiannotation_scan():
    """Scan a folder for PNGs + _annotations.coco.json and load them."""
    data = request.json
    folder = data.get("folder", "").strip().strip('"').strip("'")

    folder_path = Path(folder)
    if not folder_path.exists():
        return jsonify({"error": f"Folder not found: {folder}"}), 404

    coco_file = None
    for name in ["_annotations.coco.json", "annotations.coco.json", "annotations.json"]:
        candidate = folder_path / name
        if candidate.exists():
            coco_file = candidate
            break

    if coco_file is None:
        json_files = list(folder_path.glob("*.json"))
        if len(json_files) == 1:
            coco_file = json_files[0]

    if coco_file is None:
        return jsonify({"error": "No COCO JSON found in folder. Expected _annotations.coco.json"}), 404

    with open(coco_file) as f:
        coco = json.load(f)

    frames = {}
    for img in coco.get("images", []):
        png_path = _safe_coco_path(folder_path, img.get("file_name"))
        if png_path is None or not png_path.exists():
            continue
        filename = png_path.name
        name = filename.replace(".png", "")
        frames[name] = {
            "filename": filename,
            "path": str(png_path),
            "width": img["width"],
            "height": img["height"],
        }

    if not frames:
        return jsonify({"error": "No matching PNG files found for images in COCO JSON"}), 404

    STATE["semiannotation_dir"] = str(folder_path)
    STATE["semiannotation_frames"] = frames
    semi_keys = [k for k in STATE["annotations"] if isinstance(k, tuple) and k[0] == "semi"]
    for k in semi_keys:
        del STATE["annotations"][k]

    return jsonify({
        "status": "ok",
        "frames": list(frames.keys()),
        "coco_file": str(coco_file),
        "n_annotations": len(coco.get("annotations", [])),
    })


@bp.route("/api/semiannotation/load/<frame_key>")
def api_semiannotation_load(frame_key):
    """Load a pre-annotated frame from semiannotation directory."""
    frames = STATE.get("semiannotation_frames", {})
    if frame_key not in frames:
        return jsonify({"error": f"Frame '{frame_key}' not found"}), 404

    frame_info = frames[frame_key]
    cache_key = ("semi", frame_key)

    if cache_key in STATE["annotations"]:
        return jsonify(STATE["annotations"][cache_key])

    # Check if a corrected version exists
    corrected_path = Path(STATE["output_dir"]) / f"{frame_key}_annotations.json"
    if corrected_path.exists():
        with open(corrected_path) as f:
            corrected_coco = json.load(f)

        cat_map = build_category_map(corrected_coco.get("categories", []))

        annotations = []
        for ann in corrected_coco.get("annotations", []):
            annotations.append({
                "id": ann["id"],
                "category_id": cat_map.get(ann["category_id"], ann["category_id"]),
                "bbox": ann["bbox"],
                "area": ann["area"],
                "segmentation": ann["segmentation"],
                "score": 1.0,
                "source": "manual",
            })

        img_info = corrected_coco["images"][0]
        result = {
            "well": "semi",
            "timepoint": frame_key,
            "width": img_info["width"],
            "height": img_info["height"],
            "annotations": annotations,
            "inference_time": 0,
            "next_id": max((a["id"] for a in annotations), default=0) + 1,
            "loaded_from": "corrected",
        }
        STATE["annotations"][cache_key] = result
        return jsonify(result)

    # Load from original COCO annotations
    coco_path = Path(STATE["semiannotation_dir"]) / "_annotations.coco.json"
    with open(coco_path) as f:
        coco = json.load(f)

    img_entry = None
    for img in coco["images"]:
        if img["file_name"] == frame_info["filename"]:
            img_entry = img
            break

    if img_entry is None:
        return jsonify({"error": "Image not found in COCO JSON"}), 404

    cat_map = build_category_map(coco["categories"])

    annotations = []
    for ann in coco["annotations"]:
        if ann["image_id"] == img_entry["id"]:
            annotations.append({
                "id": ann["id"],
                "category_id": cat_map.get(ann["category_id"], ann["category_id"]),
                "bbox": ann["bbox"],
                "area": ann["area"],
                "segmentation": ann["segmentation"],
                "score": 1.0,
                "source": "model",
            })

    result = {
        "well": "semi",
        "timepoint": frame_key,
        "width": img_entry["width"],
        "height": img_entry["height"],
        "annotations": annotations,
        "inference_time": 0,
        "next_id": max((a["id"] for a in annotations), default=0) + 1,
    }

    STATE["annotations"][cache_key] = result
    return jsonify(result)


@bp.route("/api/semiannotation/frame/<frame_key>")
def api_semiannotation_frame(frame_key):
    """Serve pre-annotated frame PNG."""
    frames = STATE.get("semiannotation_frames", {})
    if frame_key not in frames:
        return jsonify({"error": "Frame not found"}), 404

    png_path = frames[frame_key]["path"]
    return send_file(png_path, mimetype="image/png")


@bp.route("/api/semiannotation/infer/<frame_key>")
def api_semiannotation_infer(frame_key):
    """Run model inference on a semi-annotation PNG."""
    frames = STATE.get("semiannotation_frames", {})
    if frame_key not in frames:
        return jsonify({"error": f"Frame '{frame_key}' not found"}), 404

    segmenter = STATE.get("segmenter")
    if segmenter is None:
        return jsonify({"error": "No model loaded. Start with --model to enable inference."}), 400

    cache_key = ("semi", frame_key)
    frame_info = frames[frame_key]
    img = Image.open(frame_info["path"]).convert("L")
    frame = np.array(img)

    t0 = time.time()
    result = segmenter.predict(frame)
    elapsed = time.time() - t0

    masks = result.masks
    bboxes = result.bboxes
    class_ids = result.class_ids
    scores = result.scores

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

    ann_result = {
        "well": "semi",
        "timepoint": frame_key,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "annotations": annotations,
        "inference_time": round(elapsed, 1),
        "next_id": n_detections,
    }

    STATE["annotations"][cache_key] = ann_result
    return jsonify(ann_result)
