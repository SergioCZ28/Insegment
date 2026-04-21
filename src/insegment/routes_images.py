"""Blueprint: image listing, serving, and navigation (no inference)."""

import io

from flask import Blueprint, jsonify, send_file
from PIL import Image

from insegment.inference_core import _load_saved_annotations
from insegment.state import STATE
from insegment.utils import _scan_image_dir, load_image

bp = Blueprint("images", __name__)


@bp.route("/api/images")
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


@bp.route("/api/image/<int:index>")
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


@bp.route("/api/load/<int:index>")
def api_load(index):
    """Load image metadata + saved annotations WITHOUT running inference.

    Used on image navigation so the model never fires unless the user
    explicitly clicks "Run Model".
    """
    if index < 0 or index >= len(STATE.get("images", [])):
        return jsonify({"error": "Invalid image index"}), 400
    # Use cached STATE if available, otherwise build an empty-but-valid result.
    if index in STATE["annotations"]:
        return jsonify(STATE["annotations"][index])
    frame = load_image(index)
    if frame is None:
        return jsonify({"error": "Could not load image"}), 404
    h, w = frame.shape[:2]
    saved = _load_saved_annotations(index)
    result = {
        "index": index,
        "filename": STATE["images"][index]["filename"],
        "width": w,
        "height": h,
        "annotations": saved if saved is not None else [],
        "inference_time": 0,
        "next_id": (max((a["id"] for a in saved), default=-1) + 1) if saved else 0,
    }
    STATE["annotations"][index] = result
    return jsonify(result)


@bp.route("/api/browse")
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
