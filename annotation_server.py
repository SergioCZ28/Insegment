"""
BacDETR Annotation Correction Tool -- Flask Backend

Start: python annotation_server.py --checkpoint path/to/checkpoint.pth
Open:  http://localhost:5000

Workflow:
1. Select a well + timepoint from the dropdown
2. Model runs inference -> detections appear as colored overlays
3. Left-click empty area = ADD cell (circle mask, default class = single-cell)
4. Right-click detection = REMOVE it
5. Click detection + press 1/2/3 = reclassify
6. Export -> saves COCO JSON to output directory
"""

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# Add HiTMicTools to path
sys.path.insert(0, "C:/Users/sergi/HiTMicTools/src")

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

# Global state
STATE = {
    "checkpoint": None,
    "tiff_dir": None,
    "output_dir": None,
    "segmenter": None,
    "cache": {},  # (well, tp) -> detection data
    "edits": {},  # (well, tp) -> list of edits applied
    "annotations": {},  # (well, tp) -> current annotation state
    "cell_radius": 4,  # default radius for added cells (pixels)
}

CLASS_NAMES = {0: "single-cell", 1: "clump", 2: "debris"}


def get_segmenter():
    """Lazy-load ScSegmenter."""
    if STATE["segmenter"] is None:
        from HiTMicTools.model_components.scsegmenter import ScSegmenter

        STATE["segmenter"] = ScSegmenter(
            model_path=STATE["checkpoint"],
            patch_size=256,
            overlap_ratio=0.33,
            score_threshold=0.40,
            nms_iou=0.4,
            clump_merge_min_overlap=250,
            priority_overlap_fraction=0.5,
            temporal_buffer_size=1,
            batch_size=32,
            mask_threshold=0.5,
            class_dict=CLASS_NAMES,
            model_type="bacdetr",
        )
    return STATE["segmenter"]


def list_tiff_files():
    """List available wells from transformed TIFFs."""
    tiff_dir = Path(STATE["tiff_dir"])
    wells = {}
    for f in sorted(tiff_dir.glob("*_transformed.tiff")):
        # Parse well from filename: {date}_{exp}_{well}_p{plate}_transformed.tiff
        parts = f.stem.replace("_transformed", "").split("_")
        # Well is typically the second-to-last part before p01
        for i, part in enumerate(parts):
            if part.startswith("p") and part[1:].isdigit():
                well = parts[i - 1]
                plate = part
                break
        else:
            continue

        # Count timepoints
        with tifffile.TiffFile(str(f)) as tif:
            n_pages = len(tif.pages)
            n_timepoints = n_pages // 2  # BF + FL alternating

        wells[well] = {
            "file": str(f),
            "well": well,
            "n_timepoints": n_timepoints,
            "filename": f.name,
        }
    return wells


def load_frame(well, timepoint):
    """Load BF channel from TIFF."""
    wells = list_tiff_files()
    if well not in wells:
        return None
    tiff_path = wells[well]["file"]
    bf_index = timepoint * 2
    with tifffile.TiffFile(tiff_path) as tif:
        if bf_index >= len(tif.pages):
            return None
        frame = tif.pages[bf_index].asarray()
    return frame


def mask_to_polygon(binary_mask):
    """Convert binary mask to polygon contour (COCO format)."""
    import cv2

    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    # Take largest contour
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    # Flatten to [x1, y1, x2, y2, ...]
    polygon = contour.flatten().tolist()
    return polygon


def circle_polygon(cx, cy, radius, n_points=12):
    """Generate a circle polygon as [x1,y1,x2,y2,...] for COCO."""
    points = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.extend([round(x, 1), round(y, 1)])
    return points


def run_inference(well, timepoint):
    """Run model inference and cache results."""
    cache_key = (well, timepoint)
    if cache_key in STATE["annotations"]:
        return STATE["annotations"][cache_key]

    frame = load_frame(well, timepoint)
    if frame is None:
        return None

    seg = get_segmenter()
    t0 = time.time()
    masks, bboxes_list, class_ids_list, scores_list = seg.predict(
        frame,
        channel_index=0,
        temporal_buffer_size=1,
        batch_size=32,
        normalize_to_255=True,
        output_shape="HW",
    )
    elapsed = time.time() - t0

    # Handle output shapes
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

        # bbox: xyxy -> xywh for COCO
        x1, y1, x2, y2 = bboxes[i]
        bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        area = float(bbox_xywh[2] * bbox_xywh[3])

        annotations.append(
            {
                "id": i,
                "category_id": int(class_ids[i]),
                "bbox": bbox_xywh,
                "area": area,
                "segmentation": [polygon],
                "score": float(scores[i]),
                "source": "model",  # vs "manual"
            }
        )

    result = {
        "well": well,
        "timepoint": timepoint,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "annotations": annotations,
        "inference_time": round(elapsed, 1),
        "next_id": n_detections,
    }

    STATE["annotations"][cache_key] = result
    return result


# ============================================================================
# Routes
# ============================================================================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/frames")
def api_frames():
    """List available wells and timepoints."""
    wells = list_tiff_files()
    result = {
        well: {"n_timepoints": info["n_timepoints"], "filename": info["filename"]}
        for well, info in wells.items()
    }
    # Also list pre-annotated frames from semiannotation dirs
    if STATE.get("semiannotation_dir"):
        result["__semiannotation__"] = {
            "n_timepoints": 0,
            "filename": "Pre-annotated frames",
            "frames": list(STATE.get("semiannotation_frames", {}).keys()),
        }
    return jsonify(result)


@app.route("/api/semiannotation/list")
def api_semiannotation_list():
    """List available semi-annotated frames."""
    frames = STATE.get("semiannotation_frames", {})
    return jsonify({"frames": list(frames.keys()), "dir": STATE.get("semiannotation_dir", "")})


@app.route("/api/semiannotation/scan", methods=["POST"])
def api_semiannotation_scan():
    """Scan a folder for PNGs + _annotations.coco.json and load them."""
    data = request.json
    folder = data.get("folder", "").strip().strip('"').strip("'")

    folder_path = Path(folder)
    if not folder_path.exists():
        return jsonify({"error": f"Folder not found: {folder}"}), 404

    # Look for COCO JSON (try common names)
    coco_file = None
    for name in ["_annotations.coco.json", "annotations.coco.json", "annotations.json"]:
        candidate = folder_path / name
        if candidate.exists():
            coco_file = candidate
            break

    # Also check for any .json file
    if coco_file is None:
        json_files = list(folder_path.glob("*.json"))
        if len(json_files) == 1:
            coco_file = json_files[0]

    if coco_file is None:
        return jsonify({"error": "No COCO JSON found in folder. Expected _annotations.coco.json"}), 404

    # Parse COCO JSON
    with open(coco_file) as f:
        coco = json.load(f)

    # Discover frames
    frames = {}
    for img in coco.get("images", []):
        png_path = folder_path / img["file_name"]
        if png_path.exists():
            name = img["file_name"].replace(".png", "")
            frames[name] = {
                "filename": img["file_name"],
                "path": str(png_path),
                "width": img["width"],
                "height": img["height"],
            }

    if not frames:
        return jsonify({"error": "No matching PNG files found for images in COCO JSON"}), 404

    # Store in state and clear annotation cache (switching folders)
    STATE["semiannotation_dir"] = str(folder_path)
    STATE["semiannotation_frames"] = frames
    # Clear cached annotations so frames reload from new folder
    semi_keys = [k for k in STATE["annotations"] if k[0] == "semi"]
    for k in semi_keys:
        del STATE["annotations"][k]

    return jsonify({
        "status": "ok",
        "frames": list(frames.keys()),
        "coco_file": str(coco_file),
        "n_annotations": len(coco.get("annotations", [])),
    })


@app.route("/api/semiannotation/load/<frame_key>")
def api_semiannotation_load(frame_key):
    """Load a pre-annotated frame from semiannotation directory."""
    frames = STATE.get("semiannotation_frames", {})
    if frame_key not in frames:
        return jsonify({"error": f"Frame '{frame_key}' not found"}), 404

    frame_info = frames[frame_key]
    cache_key = ("semi", frame_key)

    if cache_key in STATE["annotations"]:
        return jsonify(STATE["annotations"][cache_key])

    # Check if a corrected version exists (from previous session)
    corrected_path = Path(STATE["output_dir"]) / f"{frame_key}_annotations.json"
    if corrected_path.exists():
        with open(corrected_path) as f:
            corrected_coco = json.load(f)

        # Build category map (COCO 1-indexed -> 0-indexed)
        cat_map = {}
        for cat in corrected_coco.get("categories", []):
            if cat["name"] == "single-cell":
                cat_map[cat["id"]] = 0
            elif cat["name"] == "clump":
                cat_map[cat["id"]] = 1
            elif cat["name"] == "debris":
                cat_map[cat["id"]] = 2

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

    # Load COCO annotations for this frame (original semi-annotation)
    coco_path = Path(STATE["semiannotation_dir"]) / "_annotations.coco.json"
    with open(coco_path) as f:
        coco = json.load(f)

    # Find matching image
    img_entry = None
    for img in coco["images"]:
        if img["file_name"] == frame_info["filename"]:
            img_entry = img
            break

    if img_entry is None:
        return jsonify({"error": "Image not found in COCO JSON"}), 404

    # Build category map (COCO 1-indexed -> 0-indexed)
    cat_map = {}
    for cat in coco["categories"]:
        if cat["name"] == "single-cell":
            cat_map[cat["id"]] = 0
        elif cat["name"] == "clump":
            cat_map[cat["id"]] = 1
        elif cat["name"] == "debris":
            cat_map[cat["id"]] = 2

    # Get annotations for this image
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


@app.route("/api/semiannotation/frame/<frame_key>")
def api_semiannotation_frame(frame_key):
    """Serve pre-annotated frame PNG."""
    frames = STATE.get("semiannotation_frames", {})
    if frame_key not in frames:
        return jsonify({"error": "Frame not found"}), 404

    png_path = frames[frame_key]["path"]
    return send_file(png_path, mimetype="image/png")


@app.route("/api/semiannotation/infer/<frame_key>")
def api_semiannotation_infer(frame_key):
    """Run model inference on a semi-annotation PNG (instead of TIFF)."""
    frames = STATE.get("semiannotation_frames", {})
    if frame_key not in frames:
        return jsonify({"error": f"Frame '{frame_key}' not found"}), 404

    cache_key = ("semi", frame_key)

    # Load the PNG as a numpy array
    frame_info = frames[frame_key]
    img = Image.open(frame_info["path"]).convert("L")  # grayscale
    frame = np.array(img)

    # Run inference
    seg = get_segmenter()
    t0 = time.time()
    masks, bboxes_list, class_ids_list, scores_list = seg.predict(
        frame,
        channel_index=0,
        temporal_buffer_size=1,
        batch_size=32,
        normalize_to_255=True,
        output_shape="HW",
    )
    elapsed = time.time() - t0

    # Handle output shapes
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

        annotations.append(
            {
                "id": i,
                "category_id": int(class_ids[i]),
                "bbox": bbox_xywh,
                "area": area,
                "segmentation": [polygon],
                "score": float(scores[i]),
                "source": "model",
            }
        )

    result = {
        "well": "semi",
        "timepoint": frame_key,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "annotations": annotations,
        "inference_time": round(elapsed, 1),
        "next_id": n_detections,
    }

    STATE["annotations"][cache_key] = result
    return jsonify(result)


@app.route("/api/frame/<well>/<int:timepoint>")
def api_frame(well, timepoint):
    """Serve BF frame as PNG."""
    frame = load_frame(well, timepoint)
    if frame is None:
        return jsonify({"error": "Frame not found"}), 404

    # Normalize to uint8
    frame_f = frame.astype(np.float32)
    frame_f = (frame_f - frame_f.min()) / (frame_f.max() - frame_f.min() + 1e-6) * 255
    img = Image.fromarray(frame_f.astype(np.uint8))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/detections/<well>/<int:timepoint>")
def api_detections(well, timepoint):
    """Run inference and return detections."""
    result = run_inference(well, timepoint)
    if result is None:
        return jsonify({"error": "Could not process frame"}), 404
    return jsonify(result)


@app.route("/api/add", methods=["POST"])
def api_add():
    """Add a new annotation at (x, y)."""
    data = request.json
    well = data["well"]
    timepoint = data["timepoint"]
    x = data["x"]
    y = data["y"]
    class_id = data.get("class_id", 0)  # default: single-cell

    cache_key = (well, timepoint)
    if cache_key not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded for this frame"}), 400

    ann_data = STATE["annotations"][cache_key]
    next_id = ann_data["next_id"]

    radius = STATE["cell_radius"]
    polygon = circle_polygon(x, y, radius)
    bbox = [x - radius, y - radius, 2 * radius, 2 * radius]
    area = math.pi * radius * radius

    new_ann = {
        "id": next_id,
        "category_id": class_id,
        "bbox": bbox,
        "area": round(area, 1),
        "segmentation": [polygon],
        "score": 1.0,
        "source": "manual",
    }

    ann_data["annotations"].append(new_ann)
    ann_data["next_id"] = next_id + 1

    return jsonify({"status": "ok", "annotation": new_ann})


@app.route("/api/add_polygon", methods=["POST"])
def api_add_polygon():
    """Add a new annotation with a custom polygon shape."""
    data = request.json
    well = data["well"]
    timepoint = data["timepoint"]
    polygon = data["polygon"]  # flat list [x1, y1, x2, y2, ...]
    class_id = data.get("class_id", 1)  # default: clump

    cache_key = (well, timepoint)
    if cache_key not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded for this frame"}), 400

    ann_data = STATE["annotations"][cache_key]
    next_id = ann_data["next_id"]

    # Compute bbox from polygon
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

    # Compute area using shoelace formula
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    area = abs(area) / 2.0

    new_ann = {
        "id": next_id,
        "category_id": class_id,
        "bbox": bbox,
        "area": round(area, 1),
        "segmentation": [polygon],
        "score": 1.0,
        "source": "manual",
    }

    ann_data["annotations"].append(new_ann)
    ann_data["next_id"] = next_id + 1

    return jsonify({"status": "ok", "annotation": new_ann})


@app.route("/api/remove/<well>/<timepoint>/<int:ann_id>", methods=["POST"])
def api_remove(well, timepoint, ann_id):
    """Remove an annotation by ID."""
    try:
        timepoint = int(timepoint)
    except ValueError:
        pass  # keep as string for semi-annotations
    cache_key = (well, timepoint)
    if cache_key not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded"}), 400

    ann_data = STATE["annotations"][cache_key]
    ann_data["annotations"] = [
        a for a in ann_data["annotations"] if a["id"] != ann_id
    ]
    return jsonify({"status": "ok"})


@app.route("/api/reclassify/<well>/<timepoint>/<int:ann_id>", methods=["POST"])
def api_reclassify(well, timepoint, ann_id):
    """Change class of an annotation."""
    data = request.json
    new_class = data["class_id"]

    try:
        timepoint = int(timepoint)
    except ValueError:
        pass
    cache_key = (well, timepoint)
    if cache_key not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded"}), 400

    ann_data = STATE["annotations"][cache_key]
    for ann in ann_data["annotations"]:
        if ann["id"] == ann_id:
            ann["category_id"] = new_class
            return jsonify({"status": "ok"})

    return jsonify({"error": "Annotation not found"}), 404


@app.route("/api/export/<well>/<timepoint>")
def api_export(well, timepoint):
    """Export current annotations as COCO JSON."""
    # Handle both integer timepoints and string keys (for semiannotation)
    try:
        tp = int(timepoint)
        cache_key = (well, tp)
        file_label = f"{well}_t{tp:02d}"
    except ValueError:
        cache_key = (well, timepoint)
        file_label = timepoint

    if cache_key not in STATE["annotations"]:
        return jsonify({"error": "No annotations to export"}), 400

    ann_data = STATE["annotations"][cache_key]

    # Use original filename if from semiannotation
    if well == "semi" and timepoint in STATE.get("semiannotation_frames", {}):
        file_name = STATE["semiannotation_frames"][timepoint]["filename"]
    else:
        file_name = f"{file_label}.png"

    coco = {
        "images": [
            {
                "id": 0,
                "file_name": file_name,
                "width": ann_data["width"],
                "height": ann_data["height"],
            }
        ],
        "categories": [
            {"id": 1, "name": "single-cell"},
            {"id": 2, "name": "clump"},
            {"id": 3, "name": "debris"},
        ],
        "annotations": [
            {
                "id": i,
                "image_id": 0,
                "category_id": a["category_id"] + 1,  # 0-indexed -> 1-indexed for COCO
                "bbox": a["bbox"],
                "area": a["area"],
                "segmentation": a["segmentation"],
                "iscrowd": 0,
            }
            for i, a in enumerate(ann_data["annotations"])
        ],
    }

    # Save to output directory
    out_dir = Path(STATE["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_label}_annotations.json"
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    return jsonify({"status": "ok", "path": str(out_path), "n_annotations": len(coco["annotations"])})


@app.route("/api/tile_info/<well>/<timepoint>")
def api_tile_info(well, timepoint):
    """Return tile grid info for a loaded frame."""
    try:
        tp = int(timepoint)
        cache_key = (well, tp)
    except ValueError:
        cache_key = (well, timepoint)

    if cache_key not in STATE["annotations"]:
        return jsonify({"error": "Frame not loaded"}), 400

    ann_data = STATE["annotations"][cache_key]
    tile_size = 432
    n_rows = math.ceil(ann_data["height"] / tile_size)
    n_cols = math.ceil(ann_data["width"] / tile_size)

    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = c * tile_size
            y0 = r * tile_size
            x1 = min(x0 + tile_size, ann_data["width"])
            y1 = min(y0 + tile_size, ann_data["height"])

            # Count annotations in this tile
            n_anns = 0
            for ann in ann_data["annotations"]:
                bbox = ann["bbox"]
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2
                if x0 <= cx < x1 and y0 <= cy < y1:
                    n_anns += 1

            tiles.append({
                "row": r, "col": c,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "n_annotations": n_anns,
            })

    return jsonify({
        "tile_size": tile_size,
        "n_rows": n_rows, "n_cols": n_cols,
        "image_width": ann_data["width"],
        "image_height": ann_data["height"],
        "tiles": tiles,
    })


@app.route("/api/tile/<well>/<timepoint>/<int:row>/<int:col>")
def api_tile(well, timepoint, row, col):
    """Serve a single tile with coordinate grid overlay and existing annotations marked."""
    import cv2

    tile_size = 432

    # Load the frame image
    try:
        tp = int(timepoint)
        cache_key = (well, tp)
    except ValueError:
        cache_key = (well, timepoint)
        tp = timepoint

    # Get the raw image
    if well == "semi":
        frames = STATE.get("semiannotation_frames", {})
        if tp not in frames:
            return jsonify({"error": "Frame not found"}), 404
        img_pil = Image.open(frames[tp]["path"]).convert("L")
        frame = np.array(img_pil)
    else:
        frame = load_frame(well, int(tp))
        if frame is None:
            return jsonify({"error": "Frame not found"}), 404

    h, w = frame.shape[:2]
    x0 = col * tile_size
    y0 = row * tile_size
    x1 = min(x0 + tile_size, w)
    y1 = min(y0 + tile_size, h)

    # Crop tile
    tile = frame[y0:y1, x0:x1]

    # Normalize to uint8
    tile_f = tile.astype(np.float32)
    tile_f = (tile_f - tile_f.min()) / (tile_f.max() - tile_f.min() + 1e-6) * 255
    tile_u8 = tile_f.astype(np.uint8)

    # Upscale 2x for better visibility
    scale = 2
    tile_rgb = cv2.cvtColor(tile_u8, cv2.COLOR_GRAY2RGB)
    tile_rgb = cv2.resize(tile_rgb, (tile_rgb.shape[1] * scale, tile_rgb.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    # Draw coordinate grid (every 50px in image space = every 100px in display)
    for gx in range(0, tile_size + 1, 50):
        dx = gx * scale
        if dx <= tile_rgb.shape[1]:
            cv2.line(tile_rgb, (dx, 0), (dx, tile_rgb.shape[0]), (80, 80, 80), 1)
    for gy in range(0, tile_size + 1, 50):
        dy = gy * scale
        if dy <= tile_rgb.shape[0]:
            cv2.line(tile_rgb, (0, dy), (tile_rgb.shape[1], dy), (80, 80, 80), 1)

    # Add axis labels (image coordinates)
    for gx in range(0, tile_size + 1, 50):
        dx = gx * scale
        if dx <= tile_rgb.shape[1]:
            label = str(x0 + gx)
            cv2.putText(tile_rgb, label, (dx + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
    for gy in range(50, tile_size + 1, 50):
        dy = gy * scale
        if dy <= tile_rgb.shape[0]:
            label = str(y0 + gy)
            cv2.putText(tile_rgb, label, (3, dy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

    # Draw existing annotations in this tile (thick, solid outlines)
    if cache_key in STATE["annotations"]:
        ann_data = STATE["annotations"][cache_key]
        colors = {0: (76, 175, 80), 1: (244, 67, 54), 2: (255, 235, 59)}
        for ann in ann_data["annotations"]:
            seg = ann.get("segmentation", [[]])
            if not seg or not seg[0] or len(seg[0]) < 6:
                continue
            poly = seg[0]
            # Convert to tile-local coordinates, then scale 2x
            pts = []
            in_tile = False
            for i in range(0, len(poly), 2):
                px, py = poly[i] - x0, poly[i + 1] - y0
                pts.append([int(px * scale), int(py * scale)])
                if 0 <= px < tile_size and 0 <= py < tile_size:
                    in_tile = True
            if not in_tile:
                continue
            pts_np = np.array(pts, dtype=np.int32)
            color = colors.get(ann["category_id"], (128, 128, 128))
            # Thick outline + semi-transparent fill
            cv2.polylines(tile_rgb, [pts_np], True, color, 2)
            # Centroid dot
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            cv2.circle(tile_rgb, (cx, cy), 4, color, -1)

    # Tile label (top-right corner)
    label = f"R{row}C{col} ({x0},{y0})"
    cv2.putText(tile_rgb, label, (tile_rgb.shape[1] - 240, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

    # Convert to PNG
    img_pil = Image.fromarray(tile_rgb)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/stats/<well>/<timepoint>")
def api_stats(well, timepoint):
    """Get annotation statistics."""
    try:
        timepoint = int(timepoint)
    except ValueError:
        pass
    cache_key = (well, timepoint)
    if cache_key not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded"}), 400

    ann_data = STATE["annotations"][cache_key]
    anns = ann_data["annotations"]

    stats = {
        "total": len(anns),
        "single_cell": sum(1 for a in anns if a["category_id"] == 0),
        "clump": sum(1 for a in anns if a["category_id"] == 1),
        "debris": sum(1 for a in anns if a["category_id"] == 2),
        "model": sum(1 for a in anns if a.get("source") == "model"),
        "manual": sum(1 for a in anns if a.get("source") == "manual"),
    }
    return jsonify(stats)


def main():
    parser = argparse.ArgumentParser(description="BacDETR Annotation Correction Tool")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="C:/Users/sergi/ExperimentsWindows/e009_BacDETR/training/checkpoints/checkpoint_best_v13_total.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tiff-dir",
        type=str,
        default="C:/Users/sergi/ExperimentsWindows/e009_BacDETR/transformed_p01",
        help="Path to transformed TIFF directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="C:/Users/sergi/ExperimentsWindows/e009_BacDETR/annotations_corrected",
        help="Output directory for exported COCO JSON",
    )
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--cell-radius", type=int, default=4, help="Radius for manually added cells")
    parser.add_argument(
        "--semiannotation-dir",
        type=str,
        default=None,
        help="Path to semiannotation directory (PNGs + _annotations.coco.json)",
    )

    args = parser.parse_args()
    STATE["checkpoint"] = args.checkpoint
    STATE["tiff_dir"] = args.tiff_dir
    STATE["output_dir"] = args.output_dir
    STATE["cell_radius"] = args.cell_radius

    # Load semi-annotation directory if provided
    if args.semiannotation_dir:
        semi_dir = Path(args.semiannotation_dir)
        coco_file = semi_dir / "_annotations.coco.json"
        if coco_file.exists():
            STATE["semiannotation_dir"] = str(semi_dir)
            # Discover frames
            frames = {}
            with open(coco_file) as f:
                coco = json.load(f)
            for img in coco["images"]:
                png_path = semi_dir / img["file_name"]
                if png_path.exists():
                    # Key: e.g. "K11_t24"
                    name = img["file_name"].replace(".png", "")
                    # Extract well and timepoint for display
                    frames[name] = {
                        "filename": img["file_name"],
                        "path": str(png_path),
                        "width": img["width"],
                        "height": img["height"],
                    }
            STATE["semiannotation_frames"] = frames
            print(f"Semi-annotations: {len(frames)} frames from {semi_dir}")
            for k in frames:
                print(f"  - {k}")
        else:
            print(f"WARNING: No _annotations.coco.json in {semi_dir}")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"TIFF dir:   {args.tiff_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Cell radius: {args.cell_radius}px")
    print(f"Starting server on http://localhost:{args.port}")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
