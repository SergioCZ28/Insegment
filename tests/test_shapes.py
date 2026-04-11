"""Unit tests for polygon / shape generator helpers in insegment.app.

These are pure-math functions used to create annotation shapes (circles,
rectangles, ellipses) and compute bounding boxes and areas from flat
polygon lists.

Note: mask_to_polygon() is NOT tested here because it requires OpenCV,
which is not available on a headless cloud VM. The other four functions
are pure Python + math and can be tested anywhere.
"""

import math

import pytest

from insegment.app import (
    circle_polygon,
    ellipse_polygon,
    polygon_bbox_area,
    rectangle_polygon,
)


# ---------------------------------------------------------------------------
# circle_polygon
# ---------------------------------------------------------------------------

class TestCirclePolygon:
    def test_returns_flat_list(self):
        poly = circle_polygon(100, 100, 10)
        assert isinstance(poly, list)
        # Flat list: x1, y1, x2, y2, ... so length = 2 * n_points.
        assert len(poly) == 24  # default n_points=12

    def test_custom_n_points(self):
        poly = circle_polygon(0, 0, 5, n_points=6)
        assert len(poly) == 12  # 6 points * 2 coords

    def test_first_point_is_east(self):
        # At angle=0, point should be at (cx + radius, cy).
        poly = circle_polygon(50, 50, 20, n_points=4)
        assert poly[0] == 70.0   # cx + radius
        assert poly[1] == 50.0   # cy

    def test_symmetry(self):
        # With n_points=4, the four points should be at compass directions:
        # E (cx+r, cy), N (cx, cy+r), W (cx-r, cy), S (cx, cy-r)
        poly = circle_polygon(0, 0, 10, n_points=4)
        xs = poly[0::2]
        ys = poly[1::2]
        assert xs[0] == 10.0    # east
        assert ys[0] == 0.0
        assert xs[2] == -10.0   # west
        assert ys[2] == pytest.approx(0.0, abs=0.15)  # rounding

    def test_all_points_equidistant_from_center(self):
        cx, cy, r = 30, 40, 15
        poly = circle_polygon(cx, cy, r, n_points=8)
        for i in range(0, len(poly), 2):
            px, py = poly[i], poly[i + 1]
            dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            assert dist == pytest.approx(r, abs=0.2)  # within rounding

    def test_values_are_rounded_to_one_decimal(self):
        poly = circle_polygon(0, 0, 7, n_points=8)
        for v in poly:
            # Each value should have at most 1 decimal place.
            assert v == round(v, 1)

    def test_zero_radius_collapses_to_center(self):
        poly = circle_polygon(50, 60, 0, n_points=4)
        for i in range(0, len(poly), 2):
            assert poly[i] == 50.0
            assert poly[i + 1] == 60.0


# ---------------------------------------------------------------------------
# rectangle_polygon
# ---------------------------------------------------------------------------

class TestRectanglePolygon:
    def test_returns_eight_values(self):
        poly = rectangle_polygon(100, 100, 20, 10)
        assert isinstance(poly, list)
        # Rectangle = 4 corners * 2 coords = 8 values.
        assert len(poly) == 8

    def test_corner_coordinates(self):
        # cx=50, cy=50, w=20, h=10 -> half-widths: hw=10, hh=5
        poly = rectangle_polygon(50, 50, 20, 10)
        assert poly == [40.0, 45.0, 60.0, 45.0, 60.0, 55.0, 40.0, 55.0]

    def test_winding_order_is_clockwise(self):
        poly = rectangle_polygon(0, 0, 4, 2)
        # Corners: top-left, top-right, bottom-right, bottom-left
        xs = poly[0::2]
        ys = poly[1::2]
        assert xs == [-2.0, 2.0, 2.0, -2.0]
        assert ys == [-1.0, -1.0, 1.0, 1.0]

    def test_values_are_rounded_to_one_decimal(self):
        poly = rectangle_polygon(0, 0, 3, 7)
        for v in poly:
            assert v == round(v, 1)

    def test_square(self):
        poly = rectangle_polygon(10, 10, 6, 6)
        xs = poly[0::2]
        ys = poly[1::2]
        # All sides should span 6 units.
        assert max(xs) - min(xs) == 6.0
        assert max(ys) - min(ys) == 6.0

    def test_zero_dimensions(self):
        poly = rectangle_polygon(5, 5, 0, 0)
        # All four corners collapse to center.
        for i in range(0, len(poly), 2):
            assert poly[i] == 5.0
            assert poly[i + 1] == 5.0


# ---------------------------------------------------------------------------
# ellipse_polygon
# ---------------------------------------------------------------------------

class TestEllipsePolygon:
    def test_returns_flat_list(self):
        poly = ellipse_polygon(0, 0, 10, 5)
        assert isinstance(poly, list)
        # Default n_points=24.
        assert len(poly) == 48

    def test_custom_n_points(self):
        poly = ellipse_polygon(0, 0, 10, 5, n_points=8)
        assert len(poly) == 16

    def test_first_point_is_on_x_axis(self):
        poly = ellipse_polygon(50, 50, 20, 10, n_points=4)
        # angle=0 -> (cx + rx, cy)
        assert poly[0] == 70.0
        assert poly[1] == 50.0

    def test_semi_axes(self):
        # With n_points=4, we hit the four extremes of the ellipse.
        rx, ry = 30, 10
        poly = ellipse_polygon(0, 0, rx, ry, n_points=4)
        xs = poly[0::2]
        ys = poly[1::2]
        # x-extent should be [-rx, rx], y-extent should be [-ry, ry]
        assert max(xs) == pytest.approx(rx, abs=0.15)
        assert min(xs) == pytest.approx(-rx, abs=0.15)
        assert max(ys) == pytest.approx(ry, abs=0.15)
        assert min(ys) == pytest.approx(-ry, abs=0.15)

    def test_degenerates_to_circle_when_rx_equals_ry(self):
        r = 15
        circle = circle_polygon(0, 0, r, n_points=8)
        ellipse = ellipse_polygon(0, 0, r, r, n_points=8)
        assert circle == ellipse

    def test_values_are_rounded_to_one_decimal(self):
        poly = ellipse_polygon(0, 0, 7, 3, n_points=12)
        for v in poly:
            assert v == round(v, 1)


# ---------------------------------------------------------------------------
# polygon_bbox_area
# ---------------------------------------------------------------------------

class TestPolygonBboxArea:
    def test_simple_square(self):
        # Square: (0,0), (10,0), (10,10), (0,10)
        flat = [0, 0, 10, 0, 10, 10, 0, 10]
        bbox, area = polygon_bbox_area(flat)
        assert bbox == [0, 0, 10, 10]
        assert area == 100.0

    def test_rectangle(self):
        flat = [5, 5, 15, 5, 15, 25, 5, 25]
        bbox, area = polygon_bbox_area(flat)
        assert bbox == [5.0, 5.0, 10.0, 20.0]
        assert area == 200.0

    def test_bbox_is_xywh(self):
        # x, y, width, height format.
        flat = [3, 7, 13, 7, 13, 17, 3, 17]
        bbox, _ = polygon_bbox_area(flat)
        assert bbox == [3.0, 7.0, 10.0, 10.0]

    def test_triangle(self):
        # Right triangle with legs 6 and 4. Area = 0.5 * 6 * 4 = 12.
        flat = [0, 0, 6, 0, 0, 4]
        bbox, area = polygon_bbox_area(flat)
        assert bbox == [0, 0, 6, 4]
        assert area == 12.0

    def test_works_with_circle_polygon_output(self):
        poly = circle_polygon(50, 50, 10, n_points=32)
        bbox, area = polygon_bbox_area(poly)
        # Bbox should tightly bound the circle (approximately).
        assert bbox[0] == pytest.approx(40.0, abs=0.5)
        assert bbox[1] == pytest.approx(40.0, abs=0.5)
        assert bbox[2] == pytest.approx(20.0, abs=1.0)
        assert bbox[3] == pytest.approx(20.0, abs=1.0)
        # Area of a 32-gon inscribed in r=10: ~pi*r^2 = ~314.
        assert area == pytest.approx(math.pi * 100, abs=5)

    def test_works_with_rectangle_polygon_output(self):
        poly = rectangle_polygon(50, 50, 20, 10)
        bbox, area = polygon_bbox_area(poly)
        assert bbox == [40.0, 45.0, 20.0, 10.0]
        assert area == 200.0

    def test_values_are_rounded_to_one_decimal(self):
        flat = [0.123, 0.456, 10.789, 0.456, 10.789, 10.123, 0.123, 10.123]
        bbox, area = polygon_bbox_area(flat)
        for v in bbox:
            assert v == round(v, 1)
        assert area == round(area, 1)

    def test_non_axis_aligned_polygon(self):
        # Diamond shape rotated 45 degrees. Vertices at compass points of a
        # square with diagonal 10: (5,0), (10,5), (5,10), (0,5).
        flat = [5, 0, 10, 5, 5, 10, 0, 5]
        bbox, area = polygon_bbox_area(flat)
        assert bbox == [0, 0, 10.0, 10.0]
        # Shoelace area of this diamond = 50.
        assert area == 50.0
