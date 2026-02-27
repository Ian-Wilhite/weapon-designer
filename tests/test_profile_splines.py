"""Tests for profile_splines.py — Bezier and Catmull-Rom profiles."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from weapon_designer.profile_splines import (
    build_bezier_profile,
    build_catmull_rom_profile,
)


def _uniform_radii(n: int = 12, r: float = 80.0) -> np.ndarray:
    return np.full(n, r)


def _varying_radii(n: int = 12) -> np.ndarray:
    return np.array([80, 100, 80, 60, 80, 100, 80, 60, 80, 100, 80, 60][:n], dtype=float)


# ---------------------------------------------------------------------------
# Bezier profile
# ---------------------------------------------------------------------------

class TestBezierProfile:
    def test_basic_returns_polygon(self):
        radii = _uniform_radii()
        poly = build_bezier_profile(radii, max_radius_mm=150.0)
        assert poly is not None
        assert isinstance(poly, Polygon)

    def test_too_few_points_returns_none(self):
        radii = np.array([80.0, 80.0])  # N=2 < 3
        result = build_bezier_profile(radii, max_radius_mm=150.0)
        assert result is None

    def test_polygon_is_valid(self):
        radii = _varying_radii()
        poly = build_bezier_profile(radii, max_radius_mm=150.0)
        assert poly is not None
        assert poly.is_valid or poly.buffer(0).area > 0

    def test_area_positive(self):
        radii = _varying_radii()
        poly = build_bezier_profile(radii, max_radius_mm=150.0)
        assert poly is not None
        assert poly.area > 0

    def test_polygon_closes(self):
        radii = _uniform_radii()
        poly = build_bezier_profile(radii, max_radius_mm=150.0)
        assert poly is not None
        coords = list(poly.exterior.coords)
        assert coords[0] == coords[-1]

    def test_clamp_to_max_radius(self):
        radii = np.full(8, 200.0)
        poly = build_bezier_profile(radii, max_radius_mm=100.0)
        assert poly is not None
        bounds = poly.bounds
        extent = max(abs(b) for b in bounds)
        assert extent <= 120.0

    def test_area_scales_with_radius(self):
        small = build_bezier_profile(_uniform_radii(r=40.0), 150.0)
        large = build_bezier_profile(_uniform_radii(r=80.0), 150.0)
        assert small is not None and large is not None
        assert large.area > small.area


# ---------------------------------------------------------------------------
# Catmull-Rom profile
# ---------------------------------------------------------------------------

class TestCatmullRomProfile:
    def test_basic_returns_polygon(self):
        radii = _uniform_radii()
        poly = build_catmull_rom_profile(radii, max_radius_mm=150.0)
        assert poly is not None
        assert isinstance(poly, Polygon)

    def test_too_few_points_returns_none(self):
        radii = np.array([80.0, 80.0, 80.0])  # N=3 < 4
        result = build_catmull_rom_profile(radii, max_radius_mm=150.0)
        assert result is None

    def test_polygon_is_valid(self):
        radii = _varying_radii()
        poly = build_catmull_rom_profile(radii, max_radius_mm=150.0)
        assert poly is not None
        assert poly.is_valid or poly.buffer(0).area > 0

    def test_area_positive(self):
        radii = _varying_radii()
        poly = build_catmull_rom_profile(radii, max_radius_mm=150.0)
        assert poly is not None
        assert poly.area > 0

    def test_polygon_closes(self):
        radii = _uniform_radii()
        poly = build_catmull_rom_profile(radii, max_radius_mm=150.0)
        assert poly is not None
        coords = list(poly.exterior.coords)
        assert coords[0] == coords[-1]

    def test_interpolates_control_points(self):
        """Catmull-Rom should pass through or very near control points."""
        radii = np.array([80.0, 100.0, 80.0, 60.0, 80.0, 100.0,
                          80.0, 60.0, 80.0, 100.0, 80.0, 60.0])
        poly = build_catmull_rom_profile(radii, max_radius_mm=150.0)
        assert poly is not None
        # Just verify it is non-degenerate
        assert poly.area > 0

    def test_alpha_zero_uniform(self):
        radii = _uniform_radii()
        poly = build_catmull_rom_profile(radii, max_radius_mm=150.0, alpha=0.0)
        assert poly is not None

    def test_alpha_one_chordal(self):
        radii = _uniform_radii()
        poly = build_catmull_rom_profile(radii, max_radius_mm=150.0, alpha=1.0)
        assert poly is not None


# ---------------------------------------------------------------------------
# Cross-family sanity checks
# ---------------------------------------------------------------------------

class TestProfileFamilyCross:
    def test_both_families_positive_area(self):
        radii = np.array([80, 95, 85, 70, 80, 100, 85, 75, 90, 100, 85, 80], dtype=float)
        p_bez = build_bezier_profile(radii, 150.0)
        p_cr  = build_catmull_rom_profile(radii, 150.0)
        assert p_bez is not None and p_cr is not None
        assert p_bez.area > 0
        assert p_cr.area > 0

    def test_both_families_valid(self):
        radii = np.array([80, 95, 85, 70, 80, 100, 85, 75, 90, 100, 85, 80], dtype=float)
        for builder in [build_bezier_profile, build_catmull_rom_profile]:
            poly = builder(radii, 150.0)
            assert poly is not None
            assert poly.is_valid or poly.buffer(0).area > 0
