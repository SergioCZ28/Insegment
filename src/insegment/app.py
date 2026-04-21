"""Insegment -- Flask backend for interactive annotation.

This module is the Flask factory. It creates the app object, registers
the route blueprints, and exposes `configure_app` for bootstrap. All
route handlers live in `routes_*.py` modules; helpers live in
`utils.py` / `inference_core.py`; shared state lives in `state.py`.

The four polygon helpers (circle/rectangle/ellipse_polygon,
polygon_bbox_area) are re-exported here so `tests/test_shapes.py` can
keep importing them from `insegment.app`.
"""

import logging

from flask import Flask, render_template

from insegment.routes_annotations import bp as annotations_bp
from insegment.routes_export import bp as export_bp
from insegment.routes_images import bp as images_bp
from insegment.routes_inference import bp as inference_bp
from insegment.routes_labels import bp as labels_bp
from insegment.routes_semiannotation import bp as semiannotation_bp
from insegment.routes_tiles import bp as tiles_bp
from insegment.state import DEFAULT_COLORS, STATE
from insegment.utils import (
    _load_semiannotation_dir,
    _scan_image_dir,
    circle_polygon,
    ellipse_polygon,
    polygon_bbox_area,
    rectangle_polygon,
)

__all__ = [
    "app",
    "configure_app",
    "STATE",
    "circle_polygon",
    "ellipse_polygon",
    "polygon_bbox_area",
    "rectangle_polygon",
]

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(labels_bp)
app.register_blueprint(images_bp)
app.register_blueprint(annotations_bp)
app.register_blueprint(inference_bp)
app.register_blueprint(export_bp)
app.register_blueprint(tiles_bp)
app.register_blueprint(semiannotation_bp)


@app.route("/")
def index():
    return render_template("index.html")


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

    if segmenter is not None:
        STATE["class_names"] = segmenter.class_names

    STATE["class_colors"] = {
        idx: DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        for idx in STATE["class_names"]
    }

    if image_dir:
        _scan_image_dir(image_dir)

    if semiannotation_dir:
        _load_semiannotation_dir(semiannotation_dir)
