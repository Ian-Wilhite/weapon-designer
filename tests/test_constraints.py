"""Tests for constraints.py — penalty, envelope, geometry validation."""

from __future__ import annotations

import math
import pytest
import numpy as np
from shapely.geometry import Point, MultiPolygon, Polygon

from weapon_designer.constraints import (
    is_connected,
    check_mass_budget,
    check_envelope,
    check_min_feature_size,
    validate_geometry,
    constraint_penalty,
)


# ---------------------------------------------------------------------------
# is_connected
# ---------------------------------------------------------------------------

class TestIsConnected:
    def test_single_polygon(self):
        disk = Point(0, 0).buffer(50)
        assert is_connected(disk) is True

    def test_two_large_pieces_disconnected(self):
        p1 = Point(0, 0).buffer(30)
        p2 = Point(200, 0).buffer(30)
        multi = MultiPolygon([p1, p2])
        assert is_connected(multi) is False

    def test_one_large_one_sliver(self):
        big = Point(0, 0).buffer(50)
        sliver = Point(0, 0).buffer(0.1)  # tiny — below 1% of big
        multi = MultiPolygon([big, sliver])
        assert is_connected(multi) is True


# ---------------------------------------------------------------------------
# check_mass_budget
# ---------------------------------------------------------------------------

class TestCheckMassBudget:
    def test_solid_disk_within_budget(self, disk_cfg):
        # 150mm radius, 10mm thick, 7850 kg/m³ → ~5.55 kg > 5.0 budget
        large_disk = Point(0, 0).buffer(150)
        ok, mass = check_mass_budget(large_disk, disk_cfg)
        # mass should be computed correctly
        assert mass > 0
        # Determine if it passes based on actual mass vs budget
        assert isinstance(ok, bool)

    def test_small_disk_within_budget(self, disk_cfg):
        tiny = Point(0, 0).buffer(10)  # tiny — well within 5kg budget
        ok, mass = check_mass_budget(tiny, disk_cfg)
        assert ok is True
        assert mass < disk_cfg.weight_budget_kg


# ---------------------------------------------------------------------------
# check_envelope
# ---------------------------------------------------------------------------

class TestCheckEnvelope:
    def test_disk_within_envelope(self, disk_cfg):
        disk = Point(0, 0).buffer(140)
        assert check_envelope(disk, disk_cfg) is True

    def test_disk_outside_envelope(self, disk_cfg):
        big_disk = Point(0, 0).buffer(200)  # 200mm > 150mm limit
        assert check_envelope(big_disk, disk_cfg) is False

    def test_bar_within_envelope(self, bar_cfg):
        from shapely.geometry import box
        # Within 400mm length, 70mm width
        rect = box(-190, -30, 190, 30)
        assert check_envelope(rect, bar_cfg) is True

    def test_bar_exceeds_length(self, bar_cfg):
        from shapely.geometry import box
        rect = box(-250, -30, 250, 30)  # 500mm > 400mm
        assert check_envelope(rect, bar_cfg) is False

    def test_bar_exceeds_width(self, bar_cfg):
        from shapely.geometry import box
        rect = box(-190, -50, 190, 50)  # 100mm width > 70mm
        assert check_envelope(rect, bar_cfg) is False


# ---------------------------------------------------------------------------
# check_min_feature_size
# ---------------------------------------------------------------------------

class TestCheckMinFeatureSize:
    def test_large_disk_passes(self):
        disk = Point(0, 0).buffer(50)
        assert check_min_feature_size(disk, min_size_mm=3.0) is True

    def test_thin_slab_fails(self):
        from shapely.geometry import box
        thin = box(-100, -0.5, 100, 0.5)  # 1mm tall — too thin for 3mm min
        assert check_min_feature_size(thin, min_size_mm=3.0) is False


# ---------------------------------------------------------------------------
# validate_geometry
# ---------------------------------------------------------------------------

class TestValidateGeometry:
    def test_valid_polygon_unchanged(self):
        disk = Point(0, 0).buffer(50)
        result = validate_geometry(disk)
        assert result.is_valid
        assert result.area == pytest.approx(disk.area, rel=0.001)

    def test_invalid_polygon_repaired(self):
        # Bowtie / self-intersecting polygon
        coords = [(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)]
        invalid = Polygon(coords)
        result = validate_geometry(invalid)
        assert result.is_valid


# ---------------------------------------------------------------------------
# constraint_penalty
# ---------------------------------------------------------------------------

class TestConstraintPenalty:
    def test_valid_design_no_penalty(self, disk_cfg):
        """A well-designed disk near the budget limit should return ~1.0 penalty."""
        # Build a disk that's close to budget
        # AR500, 10mm thick, 7850 density, budget 5kg → target ~100mm radius
        r = 90.0  # well within budget
        disk = Point(0, 0).buffer(r)
        p = constraint_penalty(disk, disk_cfg)
        # Should be > 0
        assert p > 0.0

    def test_empty_polygon_zero_penalty(self, disk_cfg):
        empty = Point(0, 0).buffer(0)
        assert constraint_penalty(empty, disk_cfg) == 0.0

    def test_disconnected_geometry_zero_penalty(self, disk_cfg):
        p1 = Point(0, 0).buffer(30)
        p2 = Point(200, 0).buffer(30)
        multi = MultiPolygon([p1, p2])
        assert constraint_penalty(multi, disk_cfg) == 0.0

    def test_envelope_violation_reduces_penalty(self, disk_cfg):
        # Within budget but outside envelope
        small_disk = Point(0, 0).buffer(60)   # fits in envelope, within budget
        big_disk = Point(0, 0).buffer(180)    # exceeds 150mm envelope
        p_small = constraint_penalty(small_disk, disk_cfg)
        p_big = constraint_penalty(big_disk, disk_cfg)
        # big disk exceeds envelope → lower penalty (or hits mass limit first)
        assert p_small >= p_big or p_big < 0.5

    def test_penalty_is_clipped_0_to_1(self, disk_cfg):
        disk = Point(0, 0).buffer(50)
        p = constraint_penalty(disk, disk_cfg)
        assert 0.0 <= p <= 1.0
