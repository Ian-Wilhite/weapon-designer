"""Tests for bspline_profile.py — periodic cubic B-spline outer profile."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from weapon_designer.bspline_profile import (
    build_bspline_profile,
    get_bspline_bounds,
    N_BSPLINE_DEFAULT,
)


# ---------------------------------------------------------------------------
# build_bspline_profile
# ---------------------------------------------------------------------------

class TestBuildBsplineProfile:
    def test_uniform_radii_produces_disk(self):
        """Uniform radii -> near-circular profile."""
        radii = np.full(12, 80.0)
        poly = build_bspline_profile(radii, max_radius_mm=150.0)
        assert poly is not None
        assert isinstance(poly, Polygon)
        assert poly.area > 0

    def test_returns_none_for_too_few_points(self):
        """N < 4 should return None."""
        radii = np.array([80.0, 80.0, 80.0])  # N=3
        result = build_bspline_profile(radii, max_radius_mm=150.0)
        assert result is None

    def test_polygon_is_valid(self):
        radii = np.array([80, 100, 90, 70, 85, 95, 80, 75, 90, 100, 85, 80], dtype=float)
        poly = build_bspline_profile(radii, max_radius_mm=150.0)
        assert poly is not None
        assert poly.is_valid or poly.buffer(0).area > 0

    def test_clamp_to_max_radius(self):
        """Control radii exceeding max_radius should be clamped."""
        radii = np.full(8, 200.0)  # all exceed max=100
        poly = build_bspline_profile(radii, max_radius_mm=100.0)
        assert poly is not None
        bounds = poly.bounds
        extent = max(abs(b) for b in bounds)
        assert extent <= 110.0  # allow small polygon overrun

    def test_clamp_to_min_radius(self):
        """Control radii below min_radius should be clamped upward."""
        radii = np.full(8, 1.0)  # below min=5.0
        poly = build_bspline_profile(radii, max_radius_mm=150.0, min_radius_mm=5.0)
        assert poly is not None
        assert poly.area > 0

    def test_area_grows_with_radii(self):
        small_radii = np.full(12, 40.0)
        large_radii = np.full(12, 80.0)
        poly_small = build_bspline_profile(small_radii, 150.0)
        poly_large = build_bspline_profile(large_radii, 150.0)
        assert poly_small is not None and poly_large is not None
        assert poly_large.area > poly_small.area

    def test_polygon_closes(self):
        """First and last vertex of exterior should be identical (closed ring)."""
        radii = np.full(12, 80.0)
        poly = build_bspline_profile(radii, 150.0)
        assert poly is not None
        coords = list(poly.exterior.coords)
        assert coords[0] == coords[-1]

    def test_varying_radii_non_circular(self):
        """Non-uniform radii should produce a non-circular polygon."""
        uniform = np.full(12, 80.0)
        varying = np.array([80, 100, 80, 60, 80, 100, 80, 60, 80, 100, 80, 60], dtype=float)
        p_uniform = build_bspline_profile(uniform, 150.0)
        p_varying = build_bspline_profile(varying, 150.0)
        assert p_uniform is not None and p_varying is not None
        # Varying should differ significantly from uniform
        assert abs(p_uniform.area - p_varying.area) > 100.0


# ---------------------------------------------------------------------------
# get_bspline_bounds
# ---------------------------------------------------------------------------

class TestGetBsplineBounds:
    def test_returns_n_bounds(self, disk_cfg):
        bounds = get_bspline_bounds(disk_cfg)
        N = getattr(disk_cfg.optimization, "n_bspline_points", N_BSPLINE_DEFAULT)
        assert len(bounds) == N

    def test_bounds_are_positive(self, disk_cfg):
        bounds = get_bspline_bounds(disk_cfg)
        for lo, hi in bounds:
            assert lo > 0
            assert hi > lo

    def test_bar_bounds_use_diagonal(self, bar_cfg):
        import math
        bounds = get_bspline_bounds(bar_cfg)
        max_l = bar_cfg.envelope.max_length_mm
        max_w = bar_cfg.envelope.max_width_mm
        expected_max = math.hypot(max_l / 2, max_w / 2)
        _, hi = bounds[0]
        assert hi == pytest.approx(expected_max, rel=0.01)

    def test_disk_bounds_use_max_radius(self, disk_cfg):
        bounds = get_bspline_bounds(disk_cfg)
        _, hi = bounds[0]
        assert hi == pytest.approx(disk_cfg.envelope.max_radius_mm, rel=0.01)
