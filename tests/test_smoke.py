"""End-to-end smoke tests for the Insegment Flask app.

Spins up the app in-process via Flask's test client (no real HTTP, no real
model) and walks a typical annotation workflow: load a folder of images,
manage classes, add / reclassify / remove annotations, autosave, export.

Goal is NOT exhaustive coverage of every edge case (that's unit tests'
job). Goal is: if you rearrange routes or break STATE plumbing, at least
one of these fails loudly.

All tests share a single image folder fixture. STATE is reset to a known
clean default before every test so tests can't leak into each other.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from insegment import app as app_module
from insegment.app import app, configure_app


# A known-good default. Tests mutate STATE freely, then get reset by the
# autouse fixture before the next test.
_DEFAULT_STATE = {
    "segmenter": None,
    "image_dir": None,
    "images": [],
    "output_dir": None,
    "annotations": {},
    "cell_radius": 4,
    "class_names": {0: "single-cell", 1: "clump", 2: "debris"},
    "class_colors": {0: "#4caf50", 1: "#f44336", 2: "#ffeb3b"},
}


@pytest.fixture
def image_dir(tmp_path):
    """Create a temp folder with two small PNG images. Returns its path."""
    out_dir = tmp_path / "imgs"
    out_dir.mkdir()
    for i in range(2):
        arr = np.full((32, 40), 128 + i * 20, dtype=np.uint8)
        Image.fromarray(arr).save(out_dir / f"img_{i:03d}.png")
    return out_dir


@pytest.fixture
def output_dir(tmp_path):
    """Temp output directory for exports / autosaves."""
    d = tmp_path / "out"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def reset_state():
    """Restore STATE to a clean default before each test.

    STATE is a module-level global, so without this fixture the tests
    would interfere with each other.
    """
    app_module.STATE.clear()
    app_module.STATE.update(copy.deepcopy(_DEFAULT_STATE))
    yield


@pytest.fixture
def client(image_dir, output_dir):
    """Flask test client with the app configured against the temp folders."""
    configure_app(
        segmenter=None,
        image_dir=str(image_dir),
        output_dir=str(output_dir),
        cell_radius=4,
    )
    with app.test_client() as c:
        yield c


def _bootstrap(client, index=0):
    """Hit /api/detections/<index> to populate STATE['annotations'][index].

    Without this, /api/add and friends return 400 ("No annotations loaded
    for this image") because the per-image slot doesn't exist yet. With
    no model loaded, this just returns an empty annotation container.
    """
    resp = client.get(f"/api/detections/{index}")
    assert resp.status_code == 200
    return resp.get_json()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_index_page_renders(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"canvas" in resp.data.lower()

    def test_images_listed_after_configure(self, client):
        resp = client.get("/api/images")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 2
        filenames = [img["filename"] for img in data["images"]]
        assert filenames == ["img_000", "img_001"]

    def test_image_bytes_served_as_png(self, client):
        resp = client.get("/api/image/0")
        assert resp.status_code == 200
        assert resp.mimetype == "image/png"
        assert resp.data.startswith(b"\x89PNG\r\n\x1a\n")

    def test_image_out_of_range_returns_error(self, client):
        resp = client.get("/api/image/99")
        assert resp.status_code in (400, 404)

    def test_image_missing_on_disk_returns_404_not_500(self, client, image_dir):
        """Regression: deleting the PNG between scan and reload must not
        crash the route. Previously `Image.open(missing_path)` raised
        FileNotFoundError, which propagated as a 500. Now load_image()
        checks Path.exists() first and the route returns a 404 with a
        descriptive error message naming the missing path.
        """
        # Delete the PNG that was scanned at fixture time.
        missing = image_dir / "img_000.png"
        missing.unlink()
        resp = client.get("/api/image/0")
        assert resp.status_code == 404
        body = resp.get_json()
        assert "no longer available" in body["error"]
        # /api/load should also degrade gracefully, not 500.
        resp2 = client.get("/api/load/0")
        assert resp2.status_code == 404

    def test_detections_without_model_returns_empty(self, client):
        # With no model, /api/detections should still succeed and return
        # an empty annotation container (annotation-only mode).
        resp = client.get("/api/detections/0")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["annotations"] == []
        assert data["next_id"] == 0


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class TestLabels:
    def test_default_labels_exposed(self, client):
        resp = client.get("/api/labels")
        assert resp.status_code == 200
        data = resp.get_json()
        names = {c["id"]: c["name"] for c in data["labels"]}
        assert names == {0: "single-cell", 1: "clump", 2: "debris"}

    def test_rename_class(self, client):
        resp = client.post(
            "/api/labels/rename",
            json={"id": 0, "new_name": "bacterium"},
        )
        assert resp.status_code == 200
        labels = client.get("/api/labels").get_json()
        assert any(c["name"] == "bacterium" for c in labels["labels"])

    def test_add_class(self, client):
        resp = client.post("/api/labels/add", json={"name": "virus"})
        assert resp.status_code == 200
        labels = client.get("/api/labels").get_json()
        assert any(c["name"] == "virus" for c in labels["labels"])

    def test_change_class_color(self, client):
        resp = client.post(
            "/api/labels/color",
            json={"id": 0, "color": "#123456"},
        )
        assert resp.status_code == 200
        labels = client.get("/api/labels").get_json()
        single_cell = next(c for c in labels["labels"] if c["id"] == 0)
        assert single_cell["color"] == "#123456"

    def test_missing_field_returns_400(self, client):
        # require_fields() should catch missing 'new_name'.
        resp = client.post("/api/labels/rename", json={"id": 0})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Annotation CRUD
# ---------------------------------------------------------------------------

class TestAnnotationCrud:
    def test_add_circle_then_polygon(self, client):
        _bootstrap(client)
        r1 = client.post(
            "/api/add",
            json={"index": 0, "x": 10.0, "y": 15.0, "class_id": 0},
        )
        assert r1.status_code == 200
        ann1 = r1.get_json()["annotation"]
        assert "id" in ann1
        assert ann1["category_id"] == 0

        # /api/add_polygon expects a FLAT polygon [x1, y1, x2, y2, ...].
        # (Note: it 500s on pairs -- mild validation gap worth fixing later.)
        r2 = client.post(
            "/api/add_polygon",
            json={
                "index": 0,
                "polygon": [5.0, 5.0, 15.0, 5.0, 15.0, 15.0, 5.0, 15.0],
                "class_id": 1,
            },
        )
        assert r2.status_code == 200
        ann2 = r2.get_json()["annotation"]
        assert ann2["category_id"] == 1
        assert ann1["id"] != ann2["id"]

    def test_reclassify(self, client):
        _bootstrap(client)
        r1 = client.post(
            "/api/add",
            json={"index": 0, "x": 10.0, "y": 15.0, "class_id": 0},
        )
        ann_id = r1.get_json()["annotation"]["id"]
        resp = client.post(
            f"/api/reclassify/0/{ann_id}",
            json={"class_id": 2},
        )
        assert resp.status_code == 200
        stats = client.get("/api/stats/0").get_json()
        # Stats are keyed by class NAME, not ID. Class 2 = "debris".
        assert stats["debris"] == 1
        assert stats["single-cell"] == 0

    def test_remove(self, client):
        _bootstrap(client)
        r1 = client.post(
            "/api/add",
            json={"index": 0, "x": 10.0, "y": 15.0, "class_id": 0},
        )
        ann_id = r1.get_json()["annotation"]["id"]
        resp = client.post(f"/api/remove/0/{ann_id}")
        assert resp.status_code == 200
        stats = client.get("/api/stats/0").get_json()
        assert stats["total"] == 0

    def test_reclassify_invalid_class_id_rejected(self, client):
        _bootstrap(client)
        r1 = client.post(
            "/api/add",
            json={"index": 0, "x": 10.0, "y": 15.0, "class_id": 0},
        )
        ann_id = r1.get_json()["annotation"]["id"]
        resp = client.post(
            f"/api/reclassify/0/{ann_id}",
            json={"class_id": 99},  # doesn't exist
        )
        assert resp.status_code == 404  # endpoint returns 404 for unknown class


# ---------------------------------------------------------------------------
# Export / autosave round-trip
# ---------------------------------------------------------------------------

class TestExportAndAutosave:
    def _add_one(self, client):
        _bootstrap(client)
        return client.post(
            "/api/add",
            json={"index": 0, "x": 10.0, "y": 15.0, "class_id": 0},
        ).get_json()["annotation"]

    def test_export_coco_produces_valid_json(self, client, output_dir):
        self._add_one(client)
        resp = client.get("/api/export/0?format=coco")
        assert resp.status_code == 200
        files = list(Path(output_dir).glob("*_annotations.json"))
        assert len(files) == 1
        with open(files[0]) as f:
            doc = json.load(f)
        assert set(doc.keys()) == {"images", "categories", "annotations"}
        assert len(doc["annotations"]) == 1

    def test_export_yolo_plain_text(self, client, output_dir):
        self._add_one(client)
        resp = client.get("/api/export/0?format=yolo")
        assert resp.status_code == 200
        files = list(Path(output_dir).glob("*.txt"))
        assert len(files) == 1
        content = files[0].read_text()
        parts = content.strip().split()
        assert len(parts) == 5
        assert parts[0] == "0"

    def test_autosave_then_check(self, client):
        self._add_one(client)
        resp = client.post("/api/autosave", json={"index": 0})
        assert resp.status_code == 200
        check = client.get("/api/autosave/0").get_json()
        assert check["has_autosave"] is True
        assert len(check["annotations"]) == 1

    def test_restore_bulk_load(self, client):
        _bootstrap(client)
        payload = {
            "index": 0,
            "annotations": [{
                "id": 0,
                "category_id": 1,
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "area": 12.0,
                "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]],
            }],
        }
        resp = client.post("/api/restore", json=payload)
        assert resp.status_code == 200
        stats = client.get("/api/stats/0").get_json()
        assert stats["total"] == 1

    # -- Regression tests for the off-by-one loader bug ---------------------
    # Symptom: saved annotations rendered gray on reload because the loader
    # subtracted 1 from `category_id`, while `export_coco` writes 0-indexed
    # category_ids by design (matching BacDETR / unified_annotations).
    # Single-cell (id 0) became id -1 -> no class match -> gray.
    # Both round-trips below should preserve the id exactly.

    def test_save_reload_preserves_category_id(self, client, output_dir):
        """Save -> drop cached annotations -> reload via _load_saved_annotations.

        Adds one of each class, exports to COCO, clears the in-memory
        annotations cache (forcing the next /api/detections to read from
        disk), and asserts the reloaded category_ids match what was saved.
        """
        from insegment import app as app_module

        _bootstrap(client)
        for class_id in (0, 1, 2):
            r = client.post(
                "/api/add",
                json={"index": 0, "x": 10.0, "y": 15.0, "class_id": class_id},
            )
            assert r.status_code == 200

        # Save -> writes <stem>_annotations.json into output_dir.
        resp = client.get("/api/export/0?format=coco")
        assert resp.status_code == 200

        # Drop the in-memory cache so the next /api/detections reloads from disk.
        app_module.STATE["annotations"].pop(0, None)

        reloaded = client.get("/api/detections/0").get_json()
        ids = sorted(a["category_id"] for a in reloaded["annotations"])
        assert ids == [0, 1, 2], (
            f"Expected category_ids preserved as [0, 1, 2], got {ids}. "
            "If you see [-1, 0, 1] the off-by-one loader bug has come back."
        )

    def test_autosave_roundtrip_preserves_category_id(self, client):
        """Autosave -> /api/autosave/<idx> reload preserves category_id.

        Same regression bug also lived in routes_export.api_get_autosave.
        """
        _bootstrap(client)
        for class_id in (0, 1, 2):
            client.post(
                "/api/add",
                json={"index": 0, "x": 10.0, "y": 15.0, "class_id": class_id},
            )

        resp = client.post("/api/autosave", json={"index": 0})
        assert resp.status_code == 200

        check = client.get("/api/autosave/0").get_json()
        assert check["has_autosave"] is True
        ids = sorted(a["category_id"] for a in check["annotations"])
        assert ids == [0, 1, 2], (
            f"Expected category_ids preserved as [0, 1, 2], got {ids}."
        )

    # -- Defense-in-depth: drop annotations with stale category_ids --------
    # Stale autosaves (server-side or browser localStorage) written by an
    # older Insegment may contain category_ids that no longer exist (e.g.
    # the off-by-one bug produced -1, or a class was deleted). Loading
    # them silently makes the polygons render gray. Both the autosave-load
    # path and /api/restore must filter them out so the UI only ever
    # receives annotations whose class is currently defined.

    def test_restore_drops_unknown_category_ids(self, client):
        """POST /api/restore with -1 / 99 entries: dropped, only valid kept."""
        _bootstrap(client)
        payload = {
            "index": 0,
            "annotations": [
                {"id": 0, "category_id": 0, "bbox": [0, 0, 1, 1],
                 "area": 1.0, "segmentation": [[0, 0, 1, 0, 1, 1]]},
                {"id": 1, "category_id": -1, "bbox": [0, 0, 1, 1],
                 "area": 1.0, "segmentation": [[0, 0, 1, 0, 1, 1]]},
                {"id": 2, "category_id": 99, "bbox": [0, 0, 1, 1],
                 "area": 1.0, "segmentation": [[0, 0, 1, 0, 1, 1]]},
                {"id": 3, "category_id": 1, "bbox": [0, 0, 1, 1],
                 "area": 1.0, "segmentation": [[0, 0, 1, 0, 1, 1]]},
            ],
        }
        resp = client.post("/api/restore", json=payload)
        assert resp.status_code == 200
        # /api/restore reports cleaned count, not posted count.
        assert resp.get_json()["n_annotations"] == 2
        stats = client.get("/api/stats/0").get_json()
        assert stats["total"] == 2
        assert stats["single-cell"] == 1
        assert stats["clump"] == 1

    def test_get_autosave_drops_unknown_category_ids(self, client, output_dir):
        """A stale autosave file with -1 entries: served WITHOUT the -1s.

        Reproduces the exact corruption Sergio's
        Cip_..._p02_t00_autosave.json had: 558 entries with id -1
        plus 1 with id 0. After the fix, GET /api/autosave/<idx>
        returns only the valid entry.
        """
        _bootstrap(client)
        # Hand-craft a corrupt autosave file mimicking pre-fix data.
        file_label = "img_000"
        corrupt = {
            "images": [{"id": 1, "file_name": "img_000.png",
                        "width": 40, "height": 32}],
            "categories": [
                {"id": 0, "name": "single-cell"},
                {"id": 1, "name": "clump"},
                {"id": 2, "name": "debris"},
            ],
            "annotations": [
                {"id": 0, "image_id": 1, "category_id": 0,
                 "bbox": [0, 0, 1, 1], "area": 1.0,
                 "segmentation": [[0, 0, 1, 0, 1, 1]], "iscrowd": 0},
                # Two -1 entries that would render as gray. The bug
                # produced these on every reload pre-fix.
                {"id": 1, "image_id": 1, "category_id": -1,
                 "bbox": [0, 0, 1, 1], "area": 1.0,
                 "segmentation": [[0, 0, 1, 0, 1, 1]], "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": -1,
                 "bbox": [0, 0, 1, 1], "area": 1.0,
                 "segmentation": [[0, 0, 1, 0, 1, 1]], "iscrowd": 0},
            ],
        }
        autosave_path = Path(output_dir) / f"{file_label}_autosave.json"
        with open(autosave_path, "w") as f:
            json.dump(corrupt, f)

        resp = client.get("/api/autosave/0")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["has_autosave"] is True
        # Only the id=0 entry survives the sanitizer; the two -1s are gone.
        assert data["n_annotations"] == 1
        assert data["annotations"][0]["category_id"] == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_empty(self, client):
        _bootstrap(client)
        resp = client.get("/api/stats/0")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 0

    def test_stats_after_adds(self, client):
        _bootstrap(client)
        for _ in range(3):
            client.post(
                "/api/add",
                json={"index": 0, "x": 10.0, "y": 15.0, "class_id": 0},
            )
        data = client.get("/api/stats/0").get_json()
        assert data["total"] == 3
        assert data["single-cell"] == 3
        assert data["manual"] == 3
