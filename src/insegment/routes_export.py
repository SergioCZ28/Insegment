"""Blueprint: export, autosave, restore, stats."""

import json
import logging
from pathlib import Path

from flask import Blueprint, jsonify, request

from insegment.state import STATE
from insegment.utils import (
    _build_coco_dict,
    _get_file_label,
    _get_file_name,
    build_category_map,
    require_fields,
)

bp = Blueprint("export", __name__)
logger = logging.getLogger(__name__)


def _drop_unknown_classes(annotations, source_label):
    """Return only annotations whose `category_id` is a current internal class.

    Defense in depth against stale data: a previous Insegment session
    could have written an autosave (server-side OR localStorage) whose
    `category_id` is no longer valid -- typically the off-by-one bug
    that produced `-1`, but also any class the user has since deleted.
    Loading those silently makes them render as gray polygons (the
    `#888888` fallback in index.html) and look like "ghost"
    annotations the user didn't make.

    Drop them at the boundary, log how many got dropped, and the UI
    only sees clean data. Caller passes `source_label` so the log
    line is actionable ("autosave file" vs "/api/restore body").
    """
    valid_ids = set(STATE.get("class_names", {}).keys())
    if not valid_ids:
        return list(annotations)
    cleaned = []
    dropped = []
    for ann in annotations:
        cid = ann.get("category_id")
        if cid in valid_ids:
            cleaned.append(ann)
        else:
            dropped.append(cid)
    if dropped:
        from collections import Counter
        logger.warning(
            "Dropped %d %s annotations with invalid category_ids (%s); "
            "valid ids are %s.",
            len(dropped), source_label,
            dict(Counter(dropped)), sorted(valid_ids),
        )
    return cleaned


@bp.route("/api/export/<int:index>")
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


@bp.route("/api/autosave", methods=["POST"])
def api_autosave():
    """Auto-save annotations to a recovery file."""
    data, err = require_fields(request.get_json(), "index")
    if err:
        return err
    index = data["index"]

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


@bp.route("/api/autosave/<int:index>")
def api_get_autosave(index):
    """Check if an autosave exists and return its annotations."""
    file_label = _get_file_label(index)
    out_dir = Path(STATE["output_dir"])
    autosave_path = out_dir / f"{file_label}_autosave.json"

    if not autosave_path.exists():
        return jsonify({"has_autosave": False})

    with open(autosave_path) as f:
        coco = json.load(f)

    # Map COCO category IDs to internal class IDs by name. Handles both the
    # 0-indexed export Insegment writes today and any 1-indexed legacy file.
    cat_map = build_category_map(coco.get("categories", []))
    annotations = []
    for ann in coco.get("annotations", []):
        coco_cat_id = ann["category_id"]
        internal_id = cat_map.get(coco_cat_id, coco_cat_id)
        annotations.append({
            "id": ann["id"],
            "category_id": internal_id,
            "bbox": ann["bbox"],
            "area": ann["area"],
            "segmentation": ann["segmentation"],
            "score": 1.0,
            "source": "manual",
        })

    annotations = _drop_unknown_classes(
        annotations, f"autosave file {autosave_path.name}"
    )
    return jsonify({
        "has_autosave": True,
        "annotations": annotations,
        "n_annotations": len(annotations),
        "path": str(autosave_path),
    })


@bp.route("/api/restore", methods=["POST"])
def api_restore():
    """Bulk-load annotations (for autosave recovery)."""
    data, err = require_fields(request.get_json(), "index")
    if err:
        return err
    index = data["index"]
    annotations = data.get("annotations", [])

    if index not in STATE["annotations"]:
        return jsonify({"error": "Image not loaded"}), 400

    # Strip annotations whose class no longer exists. Protects against
    # browser-localStorage autosaves saved before a class rename / delete
    # / off-by-one fix landed.
    annotations = _drop_unknown_classes(annotations, "/api/restore payload")

    ann_data = STATE["annotations"][index]
    ann_data["annotations"] = annotations
    max_id = max((a["id"] for a in annotations), default=-1)
    ann_data["next_id"] = max_id + 1

    return jsonify({"status": "ok", "n_annotations": len(annotations)})


@bp.route("/api/stats/<int:index>")
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
