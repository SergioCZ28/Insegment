"""Targeted input-validation tests for API endpoints.

These are smaller than the smoke tests -- the goal is just to assert that
malformed inputs return a clean 400 instead of a 500. Add a new test here
any time you spot a crash-on-bad-input case.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from PIL import Image

from insegment import app as app_module
from insegment.app import app, configure_app


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


@pytest.fixture(autouse=True)
def reset_state():
    app_module.STATE.clear()
    app_module.STATE.update(copy.deepcopy(_DEFAULT_STATE))
    yield


@pytest.fixture
def client(tmp_path):
    image_dir = tmp_path / "imgs"
    image_dir.mkdir()
    arr = np.full((32, 40), 128, dtype=np.uint8)
    Image.fromarray(arr).save(image_dir / "img_000.png")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    configure_app(
        segmenter=None,
        image_dir=str(image_dir),
        output_dir=str(output_dir),
        cell_radius=4,
    )
    with app.test_client() as c:
        # Bootstrap the annotations slot so /api/add_polygon doesn't bail
        # out on "No annotations loaded" before reaching the format check.
        c.get("/api/detections/0")
        yield c


class TestAddPolygonValidation:
    """Regression coverage for /api/add_polygon input handling.

    Previously the endpoint would 500 on several malformed inputs because
    polygon_bbox_area() assumes a flat numeric list. Now each of these
    should return a 400 with a specific error message.
    """

    def test_list_of_pairs_rejected_with_400(self, client):
        # Most likely caller mistake: passing [[x, y], [x, y], ...] instead
        # of flat. Previously 500'd deep in polygon_bbox_area.
        resp = client.post(
            "/api/add_polygon",
            json={
                "index": 0,
                "polygon": [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0]],
                "class_id": 0,
            },
        )
        assert resp.status_code == 400
        assert "flat" in resp.get_json()["error"].lower()

    def test_too_few_points_rejected(self, client):
        # 2 points = 4 numbers, below the 3-point minimum.
        resp = client.post(
            "/api/add_polygon",
            json={"index": 0, "polygon": [1.0, 2.0, 3.0, 4.0], "class_id": 0},
        )
        assert resp.status_code == 400
        assert "3 points" in resp.get_json()["error"]

    def test_odd_coordinate_count_rejected(self, client):
        # Missing one coordinate -- would drop a y and silently shift pairs.
        resp = client.post(
            "/api/add_polygon",
            json={
                "index": 0,
                "polygon": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "class_id": 0,
            },
        )
        assert resp.status_code == 400
        assert "even" in resp.get_json()["error"].lower()

    def test_non_numeric_coordinate_rejected(self, client):
        resp = client.post(
            "/api/add_polygon",
            json={
                "index": 0,
                "polygon": [1.0, 2.0, "bad", 4.0, 5.0, 6.0],
                "class_id": 0,
            },
        )
        assert resp.status_code == 400

    def test_booleans_not_accepted_as_numbers(self, client):
        # Python quirk: bool is a subclass of int. Explicitly reject.
        resp = client.post(
            "/api/add_polygon",
            json={
                "index": 0,
                "polygon": [True, False, 3.0, 4.0, 5.0, 6.0],
                "class_id": 0,
            },
        )
        assert resp.status_code == 400

    def test_valid_flat_polygon_still_works(self, client):
        resp = client.post(
            "/api/add_polygon",
            json={
                "index": 0,
                "polygon": [5.0, 5.0, 15.0, 5.0, 15.0, 15.0, 5.0, 15.0],
                "class_id": 0,
            },
        )
        assert resp.status_code == 200
        assert resp.get_json()["annotation"]["category_id"] == 0

    def test_non_list_polygon_still_rejected(self, client):
        # Existing behaviour from before this PR -- kept as regression guard.
        resp = client.post(
            "/api/add_polygon",
            json={"index": 0, "polygon": "oops", "class_id": 0},
        )
        assert resp.status_code == 400
