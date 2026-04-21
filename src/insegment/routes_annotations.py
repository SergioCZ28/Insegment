"""Blueprint: annotation editing (add / add_polygon / remove / reclassify)."""

from flask import Blueprint, jsonify, request

from insegment.state import STATE
from insegment.utils import (
    circle_polygon,
    ellipse_polygon,
    polygon_bbox_area,
    rectangle_polygon,
    require_fields,
)

bp = Blueprint("annotations", __name__)


@bp.route("/api/add", methods=["POST"])
def api_add():
    """Add a new annotation at (x, y).

    Accepts optional 'shape' and 'params' fields for different shape types.
    Defaults to circle with STATE["cell_radius"] when not specified.
    """
    data, err = require_fields(request.json, "index", "x", "y")
    if err:
        return err
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


@bp.route("/api/add_polygon", methods=["POST"])
def api_add_polygon():
    """Add a new annotation with a custom polygon shape.

    Expects polygon as a FLAT list of alternating x/y coordinates:
        [x1, y1, x2, y2, x3, y3, ...]
    NOT a list of [x, y] pairs.
    """
    data, err = require_fields(request.json, "index", "polygon")
    if err:
        return err
    index = data["index"]
    polygon = data["polygon"]
    if not isinstance(polygon, list):
        return jsonify({"error": "Field 'polygon' must be a list of coordinates"}), 400

    # Validate flat polygon format: list of numbers, even count, >= 3 points
    # (6 numbers). Previously this would 500 deep inside polygon_bbox_area
    # if the caller passed list-of-pairs like [[x, y], [x, y], ...].
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in polygon):
        return jsonify({
            "error": (
                "Field 'polygon' must be a FLAT list of numbers "
                "[x1, y1, x2, y2, ...], not a list of pairs"
            ),
        }), 400
    if len(polygon) < 6:
        return jsonify({
            "error": "Polygon needs at least 3 points (6 coordinates)",
        }), 400
    if len(polygon) % 2 != 0:
        return jsonify({
            "error": "Polygon coordinate list length must be even (x, y pairs)",
        }), 400

    class_id = data.get("class_id", 0)

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


@bp.route("/api/remove/<int:index>/<int:ann_id>", methods=["POST"])
def api_remove(index, ann_id):
    """Remove an annotation by ID."""
    if index not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded"}), 400

    ann_data = STATE["annotations"][index]
    ann_data["annotations"] = [
        a for a in ann_data["annotations"] if a["id"] != ann_id
    ]
    return jsonify({"status": "ok"})


@bp.route("/api/reclassify/<int:index>/<int:ann_id>", methods=["POST"])
def api_reclassify(index, ann_id):
    """Change class of an annotation."""
    data, err = require_fields(request.json, "class_id")
    if err:
        return err
    new_class = data["class_id"]

    if index not in STATE["annotations"]:
        return jsonify({"error": "No annotations loaded"}), 400

    if new_class not in STATE["class_names"]:
        return jsonify({"error": f"Class ID {new_class} not found"}), 404

    ann_data = STATE["annotations"][index]
    for ann in ann_data["annotations"]:
        if ann["id"] == ann_id:
            ann["category_id"] = new_class
            return jsonify({"status": "ok"})

    return jsonify({"error": "Annotation not found"}), 404
