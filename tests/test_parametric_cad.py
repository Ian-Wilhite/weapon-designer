"""Tests for parametric_cad.py — superellipse cutouts and polar pockets."""

from __future__ import annotations

import math
import numpy as np
import pytest
from shapely.geometry import Polygon, Point

from weapon_designer.parametric_cad import (
    make_superellipse_cutout,
    CUTOUT_STRIDE_CAD,
    CUTOUT_STRIDE_POLAR,
)


# ---------------------------------------------------------------------------
# make_superellipse_cutout
# ---------------------------------------------------------------------------

class TestMakeSuperellipseCutout:
    def test_basic_ellipse(self):
        """n=2 gives an ellipse."""
        cut = make_superellipse_cutout(50, 0, 15, 10, 2.0, 0.0)
        assert cut is not None
        assert isinstance(cut, Polygon)
        assert cut.area > 0

    def test_squircle(self):
        """n=4 gives a squircle-ish shape."""
        cut = make_superellipse_cutout(50, 0, 15, 15, 4.0, 0.0)
        assert cut is not None
        assert cut.area > 0

    def test_small_axes_returns_none(self):
        """a or b < 1mm → None."""
        cut = make_superellipse_cutout(50, 0, 0.5, 10, 2.0, 0.0)
        assert cut is None
        cut2 = make_superellipse_cutout(50, 0, 10, 0.5, 2.0, 0.0)
        assert cut2 is None

    def test_centred_at_cx_cy(self):
        """Cutout centroid should be close to (cx, cy)."""
        cx, cy = 60.0, 20.0
        cut = make_superellipse_cutout(cx, cy, 10, 8, 2.0, 0.0)
        assert cut is not None
        assert abs(cut.centroid.x - cx) < 2.0
        assert abs(cut.centroid.y - cy) < 2.0

    def test_rotation_changes_orientation(self):
        """Rotating 90° should swap the bounding-box aspect ratio."""
        cut0 = make_superellipse_cutout(0, 0, 20, 10, 2.0, 0.0)
        cut90 = make_superellipse_cutout(0, 0, 20, 10, 2.0, 90.0)
        assert cut0 is not None and cut90 is not None
        b0 = cut0.bounds   # (minx, miny, maxx, maxy)
        b90 = cut90.bounds
        w0 = b0[2] - b0[0]
        h0 = b0[3] - b0[1]
        w90 = b90[2] - b90[0]
        h90 = b90[3] - b90[1]
        # After 90° rotation the width/height should swap (with tolerance)
        assert abs(w0 - h90) < 3.0
        assert abs(h0 - w90) < 3.0

    def test_degenerate_exponent_guarded(self):
        """n very small (near 0) should still return a polygon or None — no crash."""
        try:
            cut = make_superellipse_cutout(50, 0, 15, 10, 0.1, 0.0)
            # May return None or a valid polygon
            if cut is not None:
                assert cut.area >= 0
        except Exception:
            pytest.fail("make_superellipse_cutout raised on degenerate n")

    def test_polygon_is_valid(self):
        cut = make_superellipse_cutout(50, 0, 15, 12, 3.0, 45.0)
        assert cut is not None
        assert cut.is_valid or cut.buffer(0).area > 0


# ---------------------------------------------------------------------------
# Stride constants sanity
# ---------------------------------------------------------------------------

class TestStrideConstants:
    def test_cad_stride(self):
        # (cx, cy, a, b, n, angle)
        assert CUTOUT_STRIDE_CAD == 6

    def test_polar_stride(self):
        # (r, phi_deg, a, b, n)
        assert CUTOUT_STRIDE_POLAR == 5
