"""Blueprint: tile grid overlay + per-tile PNG rendering."""

import io
import math

import numpy as np
from flask import Blueprint, jsonify, send_file
from PIL import Image

from insegment.state import STATE
from insegment.utils import hex_to_rgb, load_image

bp = Blueprint("tiles", __name__)


@bp.route("/api/tile_info/<int:index>")
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


@bp.route("/api/tile/<int:index>/<int:row>/<int:col>")
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
