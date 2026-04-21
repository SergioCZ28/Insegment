"""Blueprint: label/class management.

Routes for listing, renaming, adding, removing, and recoloring classes.
"""

from flask import Blueprint, jsonify, request

from insegment.state import DEFAULT_COLORS, STATE
from insegment.utils import require_fields

bp = Blueprint("labels", __name__)


@bp.route("/api/labels")
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


@bp.route("/api/labels/rename", methods=["POST"])
def api_labels_rename():
    """Rename a label class.

    Body: {"id": 0, "new_name": "single-cell"}
    """
    data, err = require_fields(request.json, "id", "new_name")
    if err:
        return err
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


@bp.route("/api/labels/add", methods=["POST"])
def api_labels_add():
    """Add a new label class.

    Body: {"name": "mitotic", "color": "#ff00ff"}  (color is optional)
    """
    data, err = require_fields(request.json, "name")
    if err:
        return err
    name = data["name"].strip()
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


@bp.route("/api/labels/remove", methods=["POST"])
def api_labels_remove():
    """Remove a label class.

    Body: {"id": 3}
    """
    data, err = require_fields(request.json, "id")
    if err:
        return err
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


@bp.route("/api/labels/color", methods=["POST"])
def api_labels_color():
    """Change the color of a label class.

    Body: {"id": 0, "color": "#ff5722"}
    """
    data, err = require_fields(request.json, "id", "color")
    if err:
        return err
    class_id = data["id"]
    color = data["color"]

    if class_id not in STATE["class_names"]:
        return jsonify({"error": f"Class ID {class_id} not found"}), 404

    STATE["class_colors"][class_id] = color
    return jsonify({"status": "ok", "id": class_id, "color": color})
