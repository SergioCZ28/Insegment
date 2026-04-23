"""Pure-ish helpers used by the Flask routes.

Everything here is either stateless (polygon math, color conversion) or
reads/writes `STATE` through its module-level reference. Nothing in this
module touches the Flask app object, and nothing here is a route.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
from flask import jsonify
from PIL import Image

from insegment.state import DEFAULT_COLORS, IMAGE_EXTENSIONS, STATE

logger = logging.getLogger(__name__)


def require_fields(data, *fields):
    """Validate that required fields exist in request JSON data.

    Returns (data_dict, None) on success, or (None, error_response) on failure.
    Usage::

        data, err = require_fields(request.json, "id", "name")
        if err:
            return err
    """
    if data is None:
        return None, (jsonify({"error": "Request body must be JSON"}), 400)
    missing = [f for f in fields if f not in data]
    if missing:
        return None, (
            jsonify({"error": f"Missing required field(s): {', '.join(missing)}"}),
            400,
        )
    return data, None


def _safe_coco_path(folder_path, file_name):
    """Join a COCO `file_name` onto `folder_path`, rejecting anything that would
    resolve outside `folder_path`.

    Returns the resolved Path on success, or None if `file_name` is missing,
    not a string, or escapes the folder.
    """
    if not isinstance(file_name, str) or not file_name:
        return None
    folder_path = Path(folder_path).resolve()
    try:
        candidate = (folder_path / file_name).resolve()
        candidate.relative_to(folder_path)
    except (ValueError, OSError):
        return None
    return candidate


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
        png_path = _safe_coco_path(semi_dir, img.get("file_name"))
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
    STATE["semiannotation_frames"] = frames
    logger.info("Semi-annotations: %d frames from %s", len(frames), semi_dir)
    for k in frames:
        logger.debug("  - %s", k)


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


def hex_to_rgb(hex_color):
    """Convert '#4caf50' -> (76, 175, 80). Used for OpenCV tile rendering."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


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
