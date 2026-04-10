"""Export adapters for Insegment.

Each function takes annotation data and returns serializable output.
All share a common pattern: (ann_data, class_names, ...) -> content.
"""

import csv
import io
import xml.etree.ElementTree as ET


def export_coco(ann_data, class_names, file_name):
    """Build a COCO-format dict.

    Args:
        ann_data: Internal annotation dict with 'annotations', 'width', 'height'.
        class_names: {int: str} mapping of class IDs to names.
        file_name: Image file name for the COCO images entry.

    Returns:
        dict ready for json.dump().
    """
    categories = [
        {"id": idx + 1, "name": name}
        for idx, name in sorted(class_names.items())
    ]
    return {
        "images": [{
            "id": 0,
            "file_name": file_name,
            "width": ann_data["width"],
            "height": ann_data["height"],
        }],
        "categories": categories,
        "annotations": [
            {
                "id": i,
                "image_id": 0,
                "category_id": a["category_id"] + 1,
                "bbox": a["bbox"],
                "area": a["area"],
                "segmentation": a["segmentation"],
                "iscrowd": 0,
            }
            for i, a in enumerate(ann_data["annotations"])
        ],
    }


def export_yolo(ann_data, class_names, img_width, img_height):
    """Export annotations in YOLO format.

    Each line: class_id center_x center_y width height (all normalized 0-1).
    Class IDs are 0-indexed (matching internal format).

    Returns:
        str with one line per annotation.
    """
    lines = []
    for ann in ann_data["annotations"]:
        bx, by, bw, bh = ann["bbox"]
        cx = (bx + bw / 2) / img_width
        cy = (by + bh / 2) / img_height
        nw = bw / img_width
        nh = bh / img_height
        lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def export_csv(ann_data, class_names, file_name):
    """Export annotations as a flat CSV table.

    Columns: image, ann_id, class_id, class_name, bbox_x, bbox_y, bbox_w, bbox_h, area

    Returns:
        str with CSV content.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "image", "ann_id", "class_id", "class_name",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "area",
    ])
    for ann in ann_data["annotations"]:
        bx, by, bw, bh = ann["bbox"]
        name = class_names.get(ann["category_id"], f"class-{ann['category_id']}")
        writer.writerow([
            file_name, ann["id"], ann["category_id"], name,
            f"{bx:.2f}", f"{by:.2f}", f"{bw:.2f}", f"{bh:.2f}",
            f"{ann['area']:.2f}",
        ])
    return buf.getvalue()


def export_labelme(ann_data, class_names, file_name, img_width, img_height):
    """Export annotations as LabelMe JSON.

    Compatible with the labelme tool (github.com/wkentaro/labelme). Each
    annotation becomes a 'polygon' shape built from its segmentation data,
    or a 'rectangle' shape built from the bbox if no segmentation is
    available.

    Args:
        ann_data: Internal annotation dict with 'annotations'.
        class_names: {int: str} mapping of class IDs to names.
        file_name: Image file name (used as imagePath).
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        dict ready for json.dump().
    """
    shapes = []
    for ann in ann_data["annotations"]:
        label = class_names.get(
            ann["category_id"], f"class-{ann['category_id']}"
        )
        segmentation = ann.get("segmentation") or []
        if segmentation and segmentation[0]:
            # Use the first polygon; convert flat [x1,y1,x2,y2,...]
            # into LabelMe's [[x1,y1],[x2,y2],...] shape.
            flat = segmentation[0]
            points = [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]
            shape_type = "polygon"
        else:
            # No polygon -- fall back to a rectangle from the bbox.
            # LabelMe rectangles are two corner points.
            bx, by, bw, bh = ann["bbox"]
            points = [[bx, by], [bx + bw, by + bh]]
            shape_type = "rectangle"
        shapes.append({
            "label": label,
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": shape_type,
            "flags": {},
        })
    return {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": file_name,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width,
    }


def export_voc(ann_data, class_names, file_name, img_width, img_height):
    """Export annotations as Pascal VOC XML.

    Returns:
        str with XML content.
    """
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = file_name

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = "3"

    for ann in ann_data["annotations"]:
        obj = ET.SubElement(root, "object")
        name = class_names.get(ann["category_id"], f"class-{ann['category_id']}")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bx, by, bw, bh = ann["bbox"]
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(round(bx)))
        ET.SubElement(bndbox, "ymin").text = str(int(round(by)))
        ET.SubElement(bndbox, "xmax").text = str(int(round(bx + bw)))
        ET.SubElement(bndbox, "ymax").text = str(int(round(by + bh)))

    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)
