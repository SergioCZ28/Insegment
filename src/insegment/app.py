"""Insegment -- Flask backend for interactive annotation.

This module defines the Flask web application. It is started by the CLI
(insegment.cli) or can be imported and configured programmatically.
"""

import io
import json
import logging
import math
import queue
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from PIL import Image

from insegment.models.base import SegmentationResult

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
logger = logging.getLogger(__name__)

# A palette of 10 visually distinct colors. When a user adds a new class,
# it automatically gets the next color from this list (cycling back to the
# start after 10). Users can override individual colors via the UI.
DEFAULT_COLORS = [
    "#4caf50",  # green
    "#f44336",  # red
    "#ffeb3b",  # yellow
    "#2196f3",  # blue
    "#9c27b0",  # purple
    "#ff9800",  # orange
    "#00bcd4",  # cyan
    "#e91e63",  # pink
    "#8bc34a",  # lime
    "#795548",  # brown
]

# Supported image extensions (case-insensitive).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Global state -- shared across requests (single-process server).
# In production you'd use a database, but for a local annotation tool this is fine.
STATE = {
    "segmenter": None,           # Model instance (BaseSegmenter subclass) or None
    "image_dir": None,           # Path to directory with images
    "images": [],                # [{path: str, filename: str}, ...] sorted
    "output_dir": None,          # Path for exported annotation files
    "annotations": {},           # int index -> annotation data for that image
    "cell_radius": 4,            # default radius for manually added cells (pixels)
    "class_names": {0: "single-cell", 1: "clump", 2: "debris"},  # default labels
    "class_colors": {0: "#4caf50", 1: "#f44336", 2: "#ffeb3b"},  # default colors
}

# Inference job tracking for SSE progress
_inference_jobs = {}


def configure_app(
    segmenter=None,
    image_dir=None,
    output_dir=None,
    cell_radius=4,
    semiannotation_dir=None,
):
    """Configure the app with a model and paths.

    Args:
        segmenter: An instance of BaseSegmenter (or None for annotation-only mode).
        image_dir: Path to folder containing image files (PNG, JPEG, TIFF, etc.).
        output_dir: Path where exported annotations are saved.
        cell_radius: Radius in pixels for manually added circle annotations.
        semiannotation_dir: Path to folder with PNGs + _annotations.coco.json.
    """
    STATE["segmenter"] = segmenter
    STATE["image_dir"] = image_dir
    STATE["output_dir"] = output_dir or "./annotations_output"
    STATE["cell_radius"] = cell_radius

    # Update class names from model if available
    if segmenter is not None:
        STATE["class_names"] = segmenter.class_names

    # Assign colors to each class (cycling through the palette)
    STATE["class_colors"] = {
        idx: DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        for idx in STATE["class_names"]
    }

    # Scan image directory if provided
    if image_dir:
        _scan_image_dir(image_dir)

    # Load semi-annotation directory if provided
    if semiannotation_dir:
        _load_semiannotation_dir(semiannotation_dir)


def _scan_image_dir(directory):
    """Scan a directory for image files and populate STATE['images']."""
    directory = Path(directory)
    if not directory.exists():
        logger.warning("Image directory not found: %s", directory)
        return

    images = []
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append({
                "path": str(f.resolve()),
                "filename": f.stem,
            })

    STATE["images"] = images
    STATE["image_dir"] = str(directory)
    # Clear annotations cache when switching directories
    STATE["annotations"] = {
        k: v for k, v in STATE["annotations"].items()
        if not isinstance(k, int)
    }
    logger.info("Loaded %d images from %s", len(images), directory)


def _load_semiannotation_dir(semi_dir):
    """Load a semi-annotation directory (PNGs + COCO JSON)."""
    semi_dir = Path(semi_dir)
    coco_file = semi_dir / "_annotations.coco.json"
    if not coco_file.exists():
        logger.warning("No _annotations.coco.json in %s", semi_dir)
        return

    STATE["semiannotation_dir"] = str(semi_dir)
    frames = {}
    with open(coco_file) as f:
        coco = json.load(f)
    for img in coco["images"]:
        png_path = semi_dir / img["file_name"]
        if png_path.exists():
            name = img["file_name"].replace(".png", "")
            frames[name] = {
                "filename": img["file_name"],
                "path": str(png_path),
                "width": img["width"],
                "height": img["height"],
            }
    STATE["semiannotation_frames"] = frames
    logger.info("Semi-annotations: %d frames from %s", len(frames), semi_dir)
    for k in frames:
        logger.debug("  - %s", k)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_image(index):
    """Load an image by index from STATE['images'].

    Opens the image with PIL (handles PNG, JPEG, TIFF, BMP, WebP, etc.)
    and returns a numpy array (uint8, grayscale or RGB).
    """
    if index < 0 or index >= len(STATE["images"]):
        return None
    path = STATE["images"][index]["path"]
    img = Image.open(path)
    frame = np.array(img)
    # Normalize to uint8 if needed (e.g. 16-bit TIFF)
    if frame.dtype != np.uint8:
        frame_f = frame.astype(np.float32)
        frame_f = (frame_f - frame_f.min()) / (frame_f.max() - frame_f.min() + 1e-6) * 255
        frame = frame_f.astype(np.uint8)
    return frame


def build_category_map(coco_categories):
    """Map COCO category IDs to our internal class IDs.

    If a COCO category name already exists in our class_names, map to that ID.
    If it's a new name we haven't seen, auto-create a new class for it
    (with the next available ID and an auto-assigned color).

    Args:
        coco_categories: list of dicts from COCO JSON, e.g.
            [{"id": 1, "name": "leaf"}, {"id": 2, "name": "stem"}]

    Returns:
        dict mapping COCO category ID -> internal class ID
    """
    cat_map = {}
    # Build reverse lookup: name -> internal ID
    name_to_id = {name: idx for idx, name in STATE["class_names"].items()}

    for cat in coco_categories:
        coco_id = cat["id"]
        name = cat["name"]

        if name in name_to_id:
            # Already exists -- map to existing internal ID
            cat_map[coco_id] = name_to_id[name]
        else:
            # New category -- auto-create it
            next_id = max(STATE["class_names"].keys()) + 1 if STATE["class_names"] else 0
            STATE["class_names"][next_id] = name
            STATE["class_colors"][next_id] = DEFAULT_COLORS[next_id % len(DEFAULT_COLORS)]
            name_to_id[name] = next_id
            cat_map[coco_id] = next_id
            logger.info("Auto-created class: %s (id=%d)", name, next_id)

    return cat_map


def mask_to_polygon(binary_mask):
    """Convert binary mask to polygon contour (COCO format).

    Uses OpenCV to find the contour of a binary mask and returns it as a
    flat list of coordinates [x1, y1, x2, y2, ...] -- the format COCO uses.
    """
    import cv2

    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    polygon = contour.flatten().tolist()
    return polygon


def circle_polygon(cx, cy, radius, n_points=12):
    """Generate a circle as a polygon [x1, y1, x2, y2, ...] for COCO format."""
    points = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.extend([round(x, 1), round(y, 1)])
    return points


def rectangle_polygon(cx, cy, width, height):
    """Generate a rectangle as a polygon [x1, y1, x2, y2, ...] for COCO format."""
    hw, hh = width / 2, height / 2
    return [
        round(cx - hw, 1), round(cy - hh, 1),
        round(cx + hw, 1), round(cy - hh, 1),
        round(cx + hw, 1), round(cy + hh, 1),
        round(cx - hw, 1), round(cy + hh, 1),
    ]


def ellipse_polygon(cx, cy, rx, ry, n_points=24):
    """Generate an ellipse as a polygon [x1, y1, x2, y2, ...] for COCO format."""
    points = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        x = cx + rx * math.cos(angle)
        y = cy + ry * math.sin(angle)
        points.extend([round(x, 1), round(y, 1)])
    return points


def polygon_bbox_area(flat_polygon):
    """Compute bbox [x, y, w, h] and area from a flat polygon list.

    Uses the shoelace formula for area.

    Returns:
        (bbox, area) tuple.
    """
    xs = flat_polygon[0::2]
    ys = flat_polygon[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox = [round(min_x, 1), round(min_y, 1),
            round(max_x - min_x, 1), round(max_y - min_y, 1)]
    # Shoelace formula
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    area = abs(area) / 2.0
    return bbox, round(area, 1)


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

    STATE["annotations"][index] = ann_result
    return ann_result


# ---------------------------------------------------------------------------
# Helper: convert hex color to RGB tuple (for tile rendering)
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_color):
    """Convert '#4caf50' -> (76, 175, 80). Used for OpenCV tile rendering."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Label management API
# ---------------------------------------------------------------------------

@app.route("/api/labels")
def api_labels():
    """Return current label configuration (names, colors, IDs).

    The frontend calls this on page load to know what classes exist
    and how to render them. Nothing is hardcoded in the HTML anymore.
    """
    labels = []
    for idx in sorted(STATE["class_names"].keys()):
        labels.append({
            "id": idx,
            "name": STATE["class_names"][idx],
            "color": STATE["class_colors"].get(idx, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]),
        })
    return jsonify({"labels": labels})


@app.route("/api/labels/rename", methods=["POST"])
def api_labels_rename():
    """Rename a label class.

    Body: {"id": 0, "new_name": "single-cell"}
    """
    data = request.json
    class_id = data["id"]
    new_name = data["new_name"].strip()

    if class_id not in STATE["class_names"]:
        return jsonify({"error": f"Class ID {class_id} not found"}), 404
    if not new_name:
        return jsonify({"error": "Name cannot be empty"}), 400

    old_name = STATE["class_names"][class_id]
    STATE["class_names"][class_id] = new_name

    return jsonify({
        "status": "ok",
        "old_name": old_name,
        "new_name": new_name,
        "id": class_id,
    })


@app.route("/api/labels/add", methods=["POST"])
def api_labels_add():
    """Add a new label class.

    Body: {"name": "mitotic", "color": "#ff00ff"}  (color is optional)
    """
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    next_id = max(STATE["class_names"].keys()) + 1 if STATE["class_names"] else 0
    color = data.get("color") or DEFAULT_COLORS[next_id % len(DEFAULT_COLORS)]

    STATE["class_names"][next_id] = name
    STATE["class_colors"][next_id] = color

    return jsonify({
        "status": "ok",
        "id": next_id,
        "name": name,
        "color": color,
    })


@app.route("/api/labels/remove", methods=["POST"])
def api_labels_remove():
    """Remove a label class.

    Body: {"id": 3}
    """
    data = request.json
    class_id = data["id"]

    if class_id not in STATE["class_names"]:
        return jsonify({"error": f"Class ID {class_id} not found"}), 404

    if len(STATE["class_names"]) <= 1:
        return jsonify({"error": "Cannot remove the last class"}), 400

    n_affected = 0
    for ann_data in STATE["annotations"].values():
        if isinstance(ann_data, dict) and "annotations" in ann_data:
            n_affected += sum(
                1 for a in ann_data["annotations"] if a["category_id"] == class_id
            )

    removed_name = STATE["class_names"].pop(class_id)
    STATE["class_colors"].pop(class_id, None)

    return jsonify({
        "status": "ok",
        "removed_name": removed_name,
        "removed_id": class_id,
        "affected_annotations": n_affected,
    })


@app.route("/api/labels/color", methods=["POST"])
def api_labels_color():
    """Change the color of a label class.

    Body: {"id": 0, "color": "#ff5722"}
    """
    data = request.json
    class_id = data["id"]
    color = data["color"]

    if class_id not in STATE["class_names"]:
        return jsonify({"error": f"Class ID {class_id} not found"}), 404

    STATE["class_colors"][class_id] = color
    return jsonify({"status": "ok", "id": class_id, "color": color})


# ---------------------------------------------------------------------------
# Image listing and navigation API
# ---------------------------------------------------------------------------

@app.route("/api/images")
def api_images():
    """List available images."""
    images = [
        {"index": i, "filename": img["filename"]}
        for i, img in enumerate(STATE["images"])
    ]
    result = {
        "images": images,
        "count": len(images),
        "has_model": STATE["segmenter"] is not None,
    }
    if STATE.get("semiannotation_dir"):
        result["semiannotation"] = {
            "dir": STATE.get("semiannotation_dir", ""),
            "frames": list(STATE.get("semiannotation_frames", {}).keys()),
        }
    return jsonify(result)


@app.route("/api/image/<int:index>")
def api_image(index):
    """Serve an image as PNG by index."""
    frame = load_image(index)
    if frame is None:
        return jsonify({"error": "Image not found"}), 404

    # Convert to 8-bit for display (already uint8 from load_image)
    if frame.ndim == 2:
        img = Image.fromarray(frame, mode="L")
    else:
        img = Image.fromarray(frame)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/detections/<int:index>")
def api_detections(index):
    """Run inference and return detections."""
    result = run_inference(index)
    if result is None:
        return jsonify({"error": "Could not process image"}), 404
    return jsonify(result)


@app.route("/api/browse")
def api_browse():
    """Open native OS folder picker, scan for images, return list."""
    import threading as _threading

    result = {"path": None}

    def pick():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(title="Select image folder")
            root.destroy()
            result["path"] = path if path else None
        except Exception:
            result["path"] = None

    t = _threading.Thread(target=pick)
    t.start()
    t.join(timeout=120)

    if result["path"]:
        _scan_image_dir(result["path"])
        images = [
            {"index": i, "filename": img["filename"]}
            for i, img in enumerate(STATE["images"])
        ]
        return jsonify({
            "status": "ok",
            "path": result["path"],
            "images": images,
            "count": len(images),
        })
    return jsonify({"status": "cancelled"})


# ---------------------------------------------------------------------------
# Semi-annotation routes (kept for backward compat)
# ---------------------------------------------------------------------------

@app.route("/api/semiannotation/list")
def api_semiannotation_list():
    """List available semi-annotated frames."""
    frames = STATE.get("semiannotation_frames", {})
    return jsonify({"frames": list(frames.keys()), "dir": STATE.get("semiannotation_dir", "")})


@app.route("/api/browse_folder")
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


@app.route("/api/semiannotation/scan", methods=["POST"])
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


@app.route("/api/semiannotation/load/<frame_key>")
def api_semiannotation_load(frame_key):
    """Load a pre-annotated frame from semiannotation directory."""
    frames = STATE.get("semiannotation_frames", {})
    if frame_key not in frames:
        return jsonify({"error": f"Frame '{frame_key}' not found"}), 404

    frame_info = frames[frame_key]
    cache_key = ("semi", frame_key)
    class_names = STATE["class_names"]

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


# ---------------------------------------------------------------------------
# Annotation editing API (index-based)
# ---------------------------------------------------------------------------

@app.route("/api/add", methods=["POST"])
def api_add():
    """Add a new annotation at (x, y).

    Accepts optional 'shape' and 'params' fields for different shape types.
    Defaults to circle with STATE["cell_radius"] when not specified.
    """
    data = request.json
    index = data["index"]
    x = data["x"]
    y = data["y"]
    class_id = data.get("class_id", 0)
    shape = data.get("shape", "circle")
    params = data.get("params", {})

    if index not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded for this image"}), 400

    ann_data = STATE["annotations"][index]
    next_id = ann_data["next_id"]

    if shape == "rectangle":
        w = params.get("width", STATE["cell_radius"] * 2)
        h = params.get("height", STATE["cell_radius"] * 2)
        polygon = rectangle_polygon(x, y, w, h)
    elif shape == "ellipse":
        rx = params.get("rx", STATE["cell_radius"])
        ry = params.get("ry", STATE["cell_radius"] * 0.7)
        polygon = ellipse_polygon(x, y, rx, ry)
    else:
        radius = params.get("radius", STATE["cell_radius"])
        polygon = circle_polygon(x, y, radius)

    bbox, area = polygon_bbox_area(polygon)

    new_ann = {
        "id": next_id,
        "category_id": class_id,
        "bbox": bbox,
        "area": area,
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
    index = data["index"]
    polygon = data["polygon"]
    class_id = data.get("class_id", 1)

    if index not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded for this image"}), 400

    ann_data = STATE["annotations"][index]
    next_id = ann_data["next_id"]

    bbox, area = polygon_bbox_area(polygon)

    new_ann = {
        "id": next_id,
        "category_id": class_id,
        "bbox": bbox,
        "area": area,
        "segmentation": [polygon],
        "score": 1.0,
        "source": "manual",
    }

    ann_data["annotations"].append(new_ann)
    ann_data["next_id"] = next_id + 1

    return jsonify({"status": "ok", "annotation": new_ann})


@app.route("/api/remove/<int:index>/<int:ann_id>", methods=["POST"])
def api_remove(index, ann_id):
    """Remove an annotation by ID."""
    if index not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded"}), 400

    ann_data = STATE["annotations"][index]
    ann_data["annotations"] = [
        a for a in ann_data["annotations"] if a["id"] != ann_id
    ]
    return jsonify({"status": "ok"})


@app.route("/api/reclassify/<int:index>/<int:ann_id>", methods=["POST"])
def api_reclassify(index, ann_id):
    """Change class of an annotation."""
    data = request.json
    new_class = data["class_id"]

    if index not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded"}), 400

    ann_data = STATE["annotations"][index]
    for ann in ann_data["annotations"]:
        if ann["id"] == ann_id:
            ann["category_id"] = new_class
            return jsonify({"status": "ok"})

    return jsonify({"error": "Annotation not found"}), 404


# ---------------------------------------------------------------------------
# Export and autosave (index-based)
# ---------------------------------------------------------------------------

def _build_coco_dict(ann_data, class_names, file_name):
    """Build a COCO-format dict from annotation data."""
    from insegment.exporters import export_coco
    return export_coco(ann_data, class_names, file_name)


def _get_file_label(index):
    """Get file label for export/autosave filenames."""
    if index < 0 or index >= len(STATE["images"]):
        return f"image_{index}"
    return STATE["images"][index]["filename"]


def _get_file_name(index):
    """Get the original image filename for export metadata."""
    if index < 0 or index >= len(STATE["images"]):
        return f"image_{index}.png"
    path = Path(STATE["images"][index]["path"])
    return path.name


@app.route("/api/export/<int:index>")
def api_export(index):
    """Export annotations in the requested format (default: coco).

    Query params:
        format: coco | yolo | csv | voc | labelme (default: coco)
    """
    from insegment.exporters import (
        export_coco,
        export_csv,
        export_labelme,
        export_voc,
        export_yolo,
    )

    if index not in STATE["annotations"]:
        return jsonify({"error": "No annotations to export"}), 400

    ann_data = STATE["annotations"][index]
    class_names = STATE["class_names"]
    file_label = _get_file_label(index)
    file_name = _get_file_name(index)
    fmt = request.args.get("format", "coco")

    out_dir = Path(STATE["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "yolo":
        content = export_yolo(ann_data, class_names, ann_data["width"], ann_data["height"])
        out_path = out_dir / f"{file_label}_annotations.txt"
        with open(out_path, "w") as f:
            f.write(content)
    elif fmt == "csv":
        content = export_csv(ann_data, class_names, file_name)
        out_path = out_dir / f"{file_label}_annotations.csv"
        with open(out_path, "w") as f:
            f.write(content)
    elif fmt == "voc":
        content = export_voc(ann_data, class_names, file_name, ann_data["width"], ann_data["height"])
        out_path = out_dir / f"{file_label}_annotations.xml"
        with open(out_path, "w") as f:
            f.write(content)
    elif fmt == "labelme":
        labelme = export_labelme(
            ann_data, class_names, file_name,
            ann_data["width"], ann_data["height"],
        )
        out_path = out_dir / f"{file_label}_labelme.json"
        with open(out_path, "w") as f:
            json.dump(labelme, f, indent=2)
    else:
        # Default: COCO JSON
        coco = export_coco(ann_data, class_names, file_name)
        out_path = out_dir / f"{file_label}_annotations.json"
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)

    # Clear the autosave file since user explicitly saved
    autosave_path = out_dir / f"{file_label}_autosave.json"
    if autosave_path.exists():
        autosave_path.unlink()

    n_ann = len(ann_data["annotations"])
    return jsonify({"status": "ok", "path": str(out_path), "format": fmt, "n_annotations": n_ann})


@app.route("/api/autosave", methods=["POST"])
def api_autosave():
    """Auto-save annotations to a recovery file."""
    data = request.get_json()
    index = data.get("index")

    if index not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded"}), 400

    ann_data = STATE["annotations"][index]
    file_label = _get_file_label(index)
    file_name = _get_file_name(index)
    coco = _build_coco_dict(ann_data, STATE["class_names"], file_name)

    out_dir = Path(STATE["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_label}_autosave.json"
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    return jsonify({"status": "ok", "path": str(out_path)})


@app.route("/api/autosave/<int:index>")
def api_get_autosave(index):
    """Check if an autosave exists and return its annotations."""
    file_label = _get_file_label(index)
    out_dir = Path(STATE["output_dir"])
    autosave_path = out_dir / f"{file_label}_autosave.json"

    if not autosave_path.exists():
        return jsonify({"has_autosave": False})

    with open(autosave_path) as f:
        coco = json.load(f)

    # Convert COCO back to internal format (1-indexed -> 0-indexed)
    annotations = []
    for ann in coco.get("annotations", []):
        annotations.append({
            "id": ann["id"],
            "category_id": ann["category_id"] - 1,
            "bbox": ann["bbox"],
            "area": ann["area"],
            "segmentation": ann["segmentation"],
            "score": 1.0,
            "source": "manual",
        })

    return jsonify({
        "has_autosave": True,
        "annotations": annotations,
        "n_annotations": len(annotations),
        "path": str(autosave_path),
    })


@app.route("/api/restore", methods=["POST"])
def api_restore():
    """Bulk-load annotations (for autosave recovery)."""
    data = request.get_json()
    index = data.get("index")
    annotations = data.get("annotations", [])

    if index not in STATE["annotations"]:
        return jsonify({"error": "Image not loaded"}), 400

    ann_data = STATE["annotations"][index]
    ann_data["annotations"] = annotations
    max_id = max((a["id"] for a in annotations), default=-1)
    ann_data["next_id"] = max_id + 1

    return jsonify({"status": "ok", "n_annotations": len(annotations)})


# ---------------------------------------------------------------------------
# Stats API
# ---------------------------------------------------------------------------

@app.route("/api/stats/<int:index>")
def api_stats(index):
    """Get annotation statistics."""
    if index not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded"}), 400

    ann_data = STATE["annotations"][index]
    anns = ann_data["annotations"]
    class_names = STATE["class_names"]

    stats = {
        "total": len(anns),
        "model": sum(1 for a in anns if a.get("source") == "model"),
        "manual": sum(1 for a in anns if a.get("source") == "manual"),
    }
    for idx, name in class_names.items():
        stats[name] = sum(1 for a in anns if a["category_id"] == idx)

    return jsonify(stats)


# ---------------------------------------------------------------------------
# Tile API (index-based)
# ---------------------------------------------------------------------------

@app.route("/api/tile_info/<int:index>")
def api_tile_info(index):
    """Return tile grid info for a loaded image."""
    if index not in STATE["annotations"]:
        return jsonify({"error": "Image not loaded"}), 400

    ann_data = STATE["annotations"][index]
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


@app.route("/api/tile/<int:index>/<int:row>/<int:col>")
def api_tile(index, row, col):
    """Serve a single tile with coordinate grid overlay and annotations."""
    import cv2

    tile_size = 432

    frame = load_image(index)
    if frame is None:
        return jsonify({"error": "Image not found"}), 404

    # Ensure grayscale for tile rendering
    if frame.ndim == 3:
        frame = np.mean(frame, axis=2).astype(np.uint8)

    h, w = frame.shape[:2]
    x0 = col * tile_size
    y0 = row * tile_size
    x1 = min(x0 + tile_size, w)
    y1 = min(y0 + tile_size, h)

    tile = frame[y0:y1, x0:x1]

    tile_f = tile.astype(np.float32)
    tile_f = (tile_f - tile_f.min()) / (tile_f.max() - tile_f.min() + 1e-6) * 255
    tile_u8 = tile_f.astype(np.uint8)

    scale = 2
    tile_rgb = cv2.cvtColor(tile_u8, cv2.COLOR_GRAY2RGB)
    tile_rgb = cv2.resize(
        tile_rgb,
        (tile_rgb.shape[1] * scale, tile_rgb.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    for gx in range(0, tile_size + 1, 50):
        dx = gx * scale
        if dx <= tile_rgb.shape[1]:
            cv2.line(tile_rgb, (dx, 0), (dx, tile_rgb.shape[0]), (80, 80, 80), 1)
    for gy in range(0, tile_size + 1, 50):
        dy = gy * scale
        if dy <= tile_rgb.shape[0]:
            cv2.line(tile_rgb, (0, dy), (tile_rgb.shape[1], dy), (80, 80, 80), 1)

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

    if index in STATE["annotations"]:
        ann_data = STATE["annotations"][index]
        colors = {
            idx: hex_to_rgb(c) for idx, c in STATE["class_colors"].items()
        }
        for ann in ann_data["annotations"]:
            seg = ann.get("segmentation", [[]])
            if not seg or not seg[0] or len(seg[0]) < 6:
                continue
            poly = seg[0]
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
            cv2.polylines(tile_rgb, [pts_np], True, color, 2)
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            cv2.circle(tile_rgb, (cx, cy), 4, color, -1)

    label = f"R{row}C{col} ({x0},{y0})"
    cv2.putText(tile_rgb, label, (tile_rgb.shape[1] - 240, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

    img_pil = Image.fromarray(tile_rgb)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ---------------------------------------------------------------------------
# SSE inference progress
# ---------------------------------------------------------------------------

@app.route("/api/inference/start/<int:index>", methods=["POST"])
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


@app.route("/api/inference/progress/<job_id>")
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
