"""Unit tests for insegment.exporters.

The exporters take an internal annotation dict and a class_names mapping
and produce serialized output in COCO / YOLO / CSV / Pascal-VOC formats.

Internal annotation shape (built in app.py around line 360):
    {
        "index": int,
        "filename": str,
        "width": int,
        "height": int,
        "annotations": [
            {
                "id": int,
                "category_id": int,         # 0-based, internal
                "bbox": [x, y, w, h],       # floats
                "area": float,
                "segmentation": [polygon],  # list of [x1,y1,x2,y2,...]
                ...
            },
            ...
        ],
        "next_id": int,
    }

Note: COCO output is 0-indexed for category_id (matching BacDETR /
unified_annotations convention). All exporters pass the raw 0-indexed id
through.
"""

import csv
import io
import xml.etree.ElementTree as ET
import zlib

import pytest

from insegment.exporters import (
    export_coco,
    export_csv,
    export_labelme,
    export_voc,
    export_yolo,
)


def _expected_image_id(file_name):
    """Mirror of exporters._stable_image_id for test assertions."""
    return zlib.crc32(file_name.encode("utf-8")) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def class_names():
    return {0: "cell", 1: "debris"}


@pytest.fixture
def ann_data():
    """A small but realistic annotation payload with two objects."""
    return {
        "index": 0,
        "filename": "img_001.png",
        "width": 200,
        "height": 100,
        "annotations": [
            {
                "id": 0,
                "category_id": 0,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "area": 1200.0,
                "segmentation": [[10, 20, 40, 20, 40, 60, 10, 60]],
            },
            {
                "id": 1,
                "category_id": 1,
                "bbox": [50.5, 60.25, 20.0, 10.0],
                "area": 200.0,
                "segmentation": [[50, 60, 70, 60, 70, 70, 50, 70]],
            },
        ],
        "next_id": 2,
    }


@pytest.fixture
def empty_ann_data():
    return {
        "index": 0,
        "filename": "blank.png",
        "width": 64,
        "height": 32,
        "annotations": [],
        "next_id": 0,
    }


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------

class TestExportCoco:
    def test_top_level_keys(self, ann_data, class_names):
        out = export_coco(ann_data, class_names, "img_001.png")
        assert set(out.keys()) == {"images", "categories", "annotations"}

    def test_image_entry(self, ann_data, class_names):
        out = export_coco(ann_data, class_names, "img_001.png")
        assert out["images"] == [{
            "id": _expected_image_id("img_001.png"),
            "file_name": "img_001.png",
            "width": 200,
            "height": 100,
        }]

    def test_categories_are_zero_indexed_sorted_with_supercategory(self, ann_data):
        # Pass class names out of order to make sure the exporter sorts them.
        cn = {2: "third", 0: "first", 1: "second"}
        out = export_coco(ann_data, cn, "img.png")
        assert out["categories"] == [
            {"id": 0, "name": "first", "supercategory": "none"},
            {"id": 1, "name": "second", "supercategory": "none"},
            {"id": 2, "name": "third", "supercategory": "none"},
        ]

    def test_annotation_count(self, ann_data, class_names):
        out = export_coco(ann_data, class_names, "img_001.png")
        assert len(out["annotations"]) == 2

    def test_annotation_fields(self, ann_data, class_names):
        out = export_coco(ann_data, class_names, "img_001.png")
        first = out["annotations"][0]
        assert first == {
            "id": 0,
            "image_id": _expected_image_id("img_001.png"),
            "category_id": 0,  # 0-indexed, passthrough from internal
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "area": 1200.0,
            "segmentation": [[10, 20, 40, 20, 40, 60, 10, 60]],
            "iscrowd": 0,
        }

    def test_category_id_is_zero_indexed_passthrough(self, ann_data, class_names):
        out = export_coco(ann_data, class_names, "img_001.png")
        # Internal ids 0 and 1 -> COCO ids 0 and 1 (no offset).
        assert [a["category_id"] for a in out["annotations"]] == [0, 1]

    def test_annotation_ids_are_sequential(self, ann_data, class_names):
        # The exporter renumbers annotation ids by enumerate() position.
        ann_data["annotations"][0]["id"] = 99
        ann_data["annotations"][1]["id"] = 42
        out = export_coco(ann_data, class_names, "img_001.png")
        assert [a["id"] for a in out["annotations"]] == [0, 1]

    def test_image_id_is_stable_crc_and_consistent(self, ann_data, class_names):
        out = export_coco(ann_data, class_names, "img_001.png")
        expected = _expected_image_id("img_001.png")
        # Every annotation shares the same image_id, matching images[0].id.
        assert out["images"][0]["id"] == expected
        assert all(a["image_id"] == expected for a in out["annotations"])

    def test_image_id_differs_across_files(self, ann_data, class_names):
        # Two exports with different filenames must produce different image_ids
        # (the whole point of the change -- avoid collisions when merging).
        a = export_coco(ann_data, class_names, "img_001.png")
        b = export_coco(ann_data, class_names, "img_002.png")
        assert a["images"][0]["id"] != b["images"][0]["id"]

    def test_iscrowd_always_zero(self, ann_data, class_names):
        out = export_coco(ann_data, class_names, "img_001.png")
        assert all(a["iscrowd"] == 0 for a in out["annotations"])

    def test_empty_annotations(self, empty_ann_data, class_names):
        out = export_coco(empty_ann_data, class_names, "blank.png")
        assert out["annotations"] == []
        assert out["images"][0]["file_name"] == "blank.png"
        assert out["categories"] == [
            {"id": 0, "name": "cell", "supercategory": "none"},
            {"id": 1, "name": "debris", "supercategory": "none"},
        ]

    def test_is_json_serializable(self, ann_data, class_names):
        import json
        out = export_coco(ann_data, class_names, "img_001.png")
        # Should not raise.
        json.dumps(out)


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------

class TestExportYolo:
    def test_one_line_per_annotation(self, ann_data, class_names):
        out = export_yolo(ann_data, class_names, 200, 100)
        lines = out.strip().split("\n")
        assert len(lines) == 2

    def test_uses_zero_based_class_id(self, ann_data, class_names):
        out = export_yolo(ann_data, class_names, 200, 100)
        lines = out.strip().split("\n")
        # YOLO does not add the +1 offset that COCO does.
        assert lines[0].split()[0] == "0"
        assert lines[1].split()[0] == "1"

    def test_normalized_coordinates(self, ann_data, class_names):
        out = export_yolo(ann_data, class_names, 200, 100)
        first = out.strip().split("\n")[0].split()
        # bbox = [10, 20, 30, 40], img = 200x100
        # cx = (10 + 30/2) / 200 = 25/200 = 0.125
        # cy = (20 + 40/2) / 100 = 40/100 = 0.4
        # nw = 30/200 = 0.15
        # nh = 40/100 = 0.4
        assert float(first[1]) == pytest.approx(0.125)
        assert float(first[2]) == pytest.approx(0.4)
        assert float(first[3]) == pytest.approx(0.15)
        assert float(first[4]) == pytest.approx(0.4)

    def test_six_decimal_places(self, ann_data, class_names):
        out = export_yolo(ann_data, class_names, 200, 100)
        first = out.strip().split("\n")[0].split()
        # The 4 numeric fields should each have 6 digits after the decimal point.
        for value in first[1:]:
            assert "." in value
            assert len(value.split(".")[1]) == 6

    def test_trailing_newline_when_nonempty(self, ann_data, class_names):
        out = export_yolo(ann_data, class_names, 200, 100)
        assert out.endswith("\n")

    def test_empty_annotations_returns_empty_string(self, empty_ann_data, class_names):
        out = export_yolo(empty_ann_data, class_names, 64, 32)
        assert out == ""

    def test_normalization_with_full_image_box(self, class_names):
        data = {
            "annotations": [{
                "id": 0,
                "category_id": 0,
                "bbox": [0.0, 0.0, 200.0, 100.0],
                "area": 20000.0,
                "segmentation": [],
            }],
        }
        out = export_yolo(data, class_names, 200, 100)
        parts = out.strip().split()
        assert float(parts[1]) == pytest.approx(0.5)
        assert float(parts[2]) == pytest.approx(0.5)
        assert float(parts[3]) == pytest.approx(1.0)
        assert float(parts[4]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class TestExportCsv:
    def _parse(self, content):
        return list(csv.reader(io.StringIO(content)))

    def test_header_row(self, ann_data, class_names):
        rows = self._parse(export_csv(ann_data, class_names, "img_001.png"))
        assert rows[0] == [
            "image", "ann_id", "class_id", "class_name",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h", "area",
        ]

    def test_one_row_per_annotation(self, ann_data, class_names):
        rows = self._parse(export_csv(ann_data, class_names, "img_001.png"))
        # 1 header + 2 data rows
        assert len(rows) == 3

    def test_row_values(self, ann_data, class_names):
        rows = self._parse(export_csv(ann_data, class_names, "img_001.png"))
        assert rows[1] == [
            "img_001.png", "0", "0", "cell",
            "10.00", "20.00", "30.00", "40.00", "1200.00",
        ]
        assert rows[2] == [
            "img_001.png", "1", "1", "debris",
            "50.50", "60.25", "20.00", "10.00", "200.00",
        ]

    def test_class_name_fallback_for_unknown_id(self, ann_data):
        # class_names doesn't include id 1
        rows = self._parse(export_csv(ann_data, {0: "cell"}, "img.png"))
        assert rows[2][3] == "class-1"

    def test_empty_annotations_just_header(self, empty_ann_data, class_names):
        rows = self._parse(export_csv(empty_ann_data, class_names, "blank.png"))
        assert len(rows) == 1
        assert rows[0][0] == "image"

    def test_uses_zero_based_class_id(self, ann_data, class_names):
        rows = self._parse(export_csv(ann_data, class_names, "img_001.png"))
        # No +1 offset like COCO does.
        assert rows[1][2] == "0"
        assert rows[2][2] == "1"


# ---------------------------------------------------------------------------
# Pascal VOC
# ---------------------------------------------------------------------------

class TestExportVoc:
    def _parse(self, content):
        return ET.fromstring(content)

    def test_root_element(self, ann_data, class_names):
        out = export_voc(ann_data, class_names, "img_001.png", 200, 100)
        root = self._parse(out)
        assert root.tag == "annotation"

    def test_filename(self, ann_data, class_names):
        out = export_voc(ann_data, class_names, "img_001.png", 200, 100)
        root = self._parse(out)
        assert root.findtext("filename") == "img_001.png"

    def test_size(self, ann_data, class_names):
        out = export_voc(ann_data, class_names, "img_001.png", 200, 100)
        root = self._parse(out)
        size = root.find("size")
        assert size is not None
        assert size.findtext("width") == "200"
        assert size.findtext("height") == "100"
        assert size.findtext("depth") == "3"

    def test_object_count(self, ann_data, class_names):
        out = export_voc(ann_data, class_names, "img_001.png", 200, 100)
        root = self._parse(out)
        assert len(root.findall("object")) == 2

    def test_object_fields(self, ann_data, class_names):
        out = export_voc(ann_data, class_names, "img_001.png", 200, 100)
        root = self._parse(out)
        obj = root.findall("object")[0]
        assert obj.findtext("name") == "cell"
        assert obj.findtext("pose") == "Unspecified"
        assert obj.findtext("truncated") == "0"
        assert obj.findtext("difficult") == "0"

    def test_bbox_is_xyxy_integer(self, ann_data, class_names):
        out = export_voc(ann_data, class_names, "img_001.png", 200, 100)
        root = self._parse(out)
        bbox = root.findall("object")[0].find("bndbox")
        assert bbox is not None
        # bbox = [10.0, 20.0, 30.0, 40.0] -> xmin=10, ymin=20, xmax=40, ymax=60
        assert bbox.findtext("xmin") == "10"
        assert bbox.findtext("ymin") == "20"
        assert bbox.findtext("xmax") == "40"
        assert bbox.findtext("ymax") == "60"

    def test_bbox_rounding_uses_round_half_to_even(self, class_names):
        # 50.5 + 20.0 = 70.5 -> rounds to 70 (banker's rounding via Python's round())
        # 60.25 + 10.0 = 70.25 -> rounds to 70
        data = {
            "annotations": [{
                "id": 0,
                "category_id": 0,
                "bbox": [50.5, 60.25, 20.0, 10.0],
                "area": 200.0,
                "segmentation": [],
            }],
        }
        out = export_voc(data, class_names, "img.png", 200, 100)
        bbox = self._parse(out).find("object").find("bndbox")
        assert bbox.findtext("xmin") == "50"  # round(50.5) -> 50 (banker's)
        assert bbox.findtext("ymin") == "60"  # round(60.25) -> 60
        assert bbox.findtext("xmax") == "70"  # round(70.5) -> 70 (banker's)
        assert bbox.findtext("ymax") == "70"  # round(70.25) -> 70

    def test_class_name_fallback_for_unknown_id(self, ann_data):
        out = export_voc(ann_data, {0: "cell"}, "img.png", 200, 100)
        root = self._parse(out)
        names = [o.findtext("name") for o in root.findall("object")]
        assert names == ["cell", "class-1"]

    def test_empty_annotations(self, empty_ann_data, class_names):
        out = export_voc(empty_ann_data, class_names, "blank.png", 64, 32)
        root = self._parse(out)
        assert root.findtext("filename") == "blank.png"
        assert len(root.findall("object")) == 0

    def test_xml_declaration(self, ann_data, class_names):
        out = export_voc(ann_data, class_names, "img_001.png", 200, 100)
        assert out.startswith("<?xml")

    def test_xml_is_parseable(self, ann_data, class_names):
        out = export_voc(ann_data, class_names, "img_001.png", 200, 100)
        # Should not raise.
        ET.fromstring(out)


# ---------------------------------------------------------------------------
# LabelMe
# ---------------------------------------------------------------------------

class TestExportLabelme:
    def test_top_level_keys(self, ann_data, class_names):
        out = export_labelme(ann_data, class_names, "img_001.png", 200, 100)
        assert set(out.keys()) == {
            "version", "flags", "shapes",
            "imagePath", "imageData", "imageHeight", "imageWidth",
        }

    def test_image_metadata(self, ann_data, class_names):
        out = export_labelme(ann_data, class_names, "img_001.png", 200, 100)
        assert out["imagePath"] == "img_001.png"
        assert out["imageWidth"] == 200
        assert out["imageHeight"] == 100
        assert out["imageData"] is None

    def test_version_is_string(self, ann_data, class_names):
        out = export_labelme(ann_data, class_names, "img_001.png", 200, 100)
        assert isinstance(out["version"], str)
        # Should follow labelme's semver-ish format (e.g. "5.4.1").
        assert out["version"].count(".") >= 1

    def test_top_level_flags_is_dict(self, ann_data, class_names):
        out = export_labelme(ann_data, class_names, "img_001.png", 200, 100)
        assert out["flags"] == {}

    def test_shapes_count(self, ann_data, class_names):
        out = export_labelme(ann_data, class_names, "img_001.png", 200, 100)
        assert len(out["shapes"]) == 2

    def test_polygon_shape_fields(self, ann_data, class_names):
        out = export_labelme(ann_data, class_names, "img_001.png", 200, 100)
        shape = out["shapes"][0]
        assert shape["label"] == "cell"
        assert shape["shape_type"] == "polygon"
        assert shape["group_id"] is None
        assert shape["flags"] == {}
        assert shape["description"] == ""

    def test_polygon_points_are_xy_pairs(self, ann_data, class_names):
        out = export_labelme(ann_data, class_names, "img_001.png", 200, 100)
        # segmentation = [[10, 20, 40, 20, 40, 60, 10, 60]]
        # -> points = [[10,20],[40,20],[40,60],[10,60]]
        assert out["shapes"][0]["points"] == [
            [10, 20], [40, 20], [40, 60], [10, 60],
        ]

    def test_labels_come_from_class_names(self, ann_data, class_names):
        out = export_labelme(ann_data, class_names, "img_001.png", 200, 100)
        labels = [s["label"] for s in out["shapes"]]
        assert labels == ["cell", "debris"]

    def test_class_name_fallback_for_unknown_id(self, ann_data):
        out = export_labelme(ann_data, {0: "cell"}, "img.png", 200, 100)
        labels = [s["label"] for s in out["shapes"]]
        assert labels == ["cell", "class-1"]

    def test_rectangle_fallback_when_no_segmentation(self, class_names):
        data = {
            "annotations": [{
                "id": 0,
                "category_id": 0,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "area": 1200.0,
                "segmentation": [],
            }],
        }
        out = export_labelme(data, class_names, "img.png", 200, 100)
        shape = out["shapes"][0]
        assert shape["shape_type"] == "rectangle"
        # Two corner points: top-left and bottom-right.
        assert shape["points"] == [[10.0, 20.0], [40.0, 60.0]]

    def test_empty_annotations(self, empty_ann_data, class_names):
        out = export_labelme(empty_ann_data, class_names, "blank.png", 64, 32)
        assert out["shapes"] == []
        assert out["imagePath"] == "blank.png"
        assert out["imageHeight"] == 32
        assert out["imageWidth"] == 64

    def test_is_json_serializable(self, ann_data, class_names):
        import json
        out = export_labelme(ann_data, class_names, "img_001.png", 200, 100)
        # Should not raise.
        json.dumps(out)
