"""Tests for profile_builder.py — dispatcher for all profile families."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from weapon_designer.profile_builder import build_profile, get_profile_bounds


def _mid_radii(cfg, n: int = 12) -> np.ndarray:
    """Return radii at the midpoint of the bspline bounds for cfg."""
    bounds = get_profile_bounds("bspline", cfg)
    return np.array([(lo + hi) / 2 for lo, hi in bounds])


# ---------------------------------------------------------------------------
# build_profile dispatcher
# ---------------------------------------------------------------------------

class TestBuildProfileDispatcher:
    @pytest.mark.parametrize("ptype", ["bspline", "bezier", "catmull_rom"])
    def test_all_spline_types_return_polygon(self, disk_cfg, ptype):
        radii = _mid_radii(disk_cfg)
        poly = build_profile(ptype, radii, disk_cfg)
        assert poly is not None
        assert isinstance(poly, Polygon)
        assert poly.area > 0

    def test_fourier_falls_back_to_bspline(self, disk_cfg):
        radii = _mid_radii(disk_cfg)
        poly = build_profile("fourier", radii, disk_cfg)
        assert poly is not None
        assert poly.area > 0

    def test_unknown_type_raises(self, disk_cfg):
        radii = _mid_radii(disk_cfg)
        with pytest.raises(ValueError):
            build_profile("unknown_profile", radii, disk_cfg)

    @pytest.mark.parametrize("ptype", ["bspline", "bezier", "catmull_rom"])
    def test_bar_cfg_all_types(self, bar_cfg, ptype):
        radii = _mid_radii(bar_cfg)
        poly = build_profile(ptype, radii, bar_cfg)
        assert poly is not None
        assert poly.area > 0

    def test_case_insensitive(self, disk_cfg):
        radii = _mid_radii(disk_cfg)
        poly = build_profile("BSPLINE", radii, disk_cfg)
        assert poly is not None


# ---------------------------------------------------------------------------
# get_profile_bounds dispatcher
# ---------------------------------------------------------------------------

class TestGetProfileBounds:
    @pytest.mark.parametrize("ptype", ["bspline", "bezier", "catmull_rom"])
    def test_bounds_count_matches_n_bspline_points(self, disk_cfg, ptype):
        bounds = get_profile_bounds(ptype, disk_cfg)
        N = getattr(disk_cfg.optimization, "n_bspline_points", 12)
        assert len(bounds) == N

    @pytest.mark.parametrize("ptype", ["bspline", "bezier", "catmull_rom"])
    def test_bounds_are_valid_ranges(self, disk_cfg, ptype):
        bounds = get_profile_bounds(ptype, disk_cfg)
        for lo, hi in bounds:
            assert lo >= 0
            assert hi > lo

    @pytest.mark.parametrize("ptype", ["bspline", "bezier", "catmull_rom"])
    def test_all_spline_bounds_identical(self, disk_cfg, ptype):
        """All spline families share the same bounds."""
        bspline_bounds = get_profile_bounds("bspline", disk_cfg)
        ptype_bounds   = get_profile_bounds(ptype, disk_cfg)
        assert bspline_bounds == ptype_bounds

    def test_fourier_returns_different_structure(self, disk_cfg):
        """Fourier bounds may differ in length (different param count)."""
        fourier_bounds = get_profile_bounds("fourier", disk_cfg)
        assert len(fourier_bounds) > 0
        for lo, hi in fourier_bounds:
            assert hi > lo
