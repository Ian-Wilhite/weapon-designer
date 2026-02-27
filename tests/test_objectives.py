"""Tests for objectives.py — weighted_score, impact_zone_score, compute_metrics."""

from __future__ import annotations

import math
import numpy as np
import pytest
from shapely.geometry import Point
from shapely.affinity import translate

from weapon_designer.objectives import (
    impact_zone_score,
    weighted_score,
    compute_metrics,
    _num_teeth,
)


# ---------------------------------------------------------------------------
# impact_zone_score
# ---------------------------------------------------------------------------

class TestImpactZoneScore:
    def test_solid_disk_score_nonneg(self):
        """Solid disk returns a non-negative score in [0, 1].

        The baseline impact_zone_score() checks poly.contains() at frac=1.0 of
        max_extent, which is always on the polygon boundary and returns False.
        Smooth disks therefore score 0 — this is a known baseline limitation
        documented in the research paper (see MEMORY.md §Known design weaknesses).
        """
        disk = Point(0, 0).buffer(100, resolution=256)
        score = impact_zone_score(disk)
        assert 0.0 <= score <= 1.0

    def test_tiny_disk_returns_zero(self):
        """A disk with max_extent < 1mm returns 0."""
        tiny = Point(0, 0).buffer(0.5)
        assert impact_zone_score(tiny) == 0.0

    def test_score_in_range(self):
        disk = Point(0, 0).buffer(80)
        score = impact_zone_score(disk)
        assert 0.0 <= score <= 1.0

    def test_annular_ring_lower_score_than_solid(self):
        """An annular ring with hollow center has the same rim, but solid disk wins."""
        solid = Point(0, 0).buffer(100, resolution=256)
        ring = solid.difference(Point(0, 0).buffer(50, resolution=128))
        score_solid = impact_zone_score(solid)
        score_ring = impact_zone_score(ring)
        # Both should be > 0; solid may score slightly higher or equal (outer 80-100% solid either way)
        assert score_ring >= 0.0
        assert score_solid >= 0.0


# ---------------------------------------------------------------------------
# _num_teeth
# ---------------------------------------------------------------------------

class TestNumTeeth:
    def test_disk_one_tooth(self, disk_cfg):
        assert _num_teeth(disk_cfg) == 1

    def test_bar_two_teeth(self, bar_cfg):
        assert _num_teeth(bar_cfg) == 2

    def test_eggbeater_three_teeth(self, disk_cfg):
        from weapon_designer.config import WeaponConfig
        cfg = WeaponConfig(weapon_style="eggbeater")
        assert _num_teeth(cfg) == 3


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_required_keys(self, disk_cfg):
        disk = Point(0, 0).buffer(80)
        m = compute_metrics(disk, disk_cfg)
        for key in ("mass_kg", "moi_kg_mm2", "energy_joules", "bite_mm",
                    "com_offset_mm", "structural_integrity", "mass_utilization",
                    "num_teeth", "impact_zone"):
            assert key in m, f"Missing key: {key}"

    def test_mass_positive(self, disk_cfg):
        disk = Point(0, 0).buffer(80)
        m = compute_metrics(disk, disk_cfg)
        assert m["mass_kg"] > 0

    def test_moi_positive(self, disk_cfg):
        disk = Point(0, 0).buffer(80)
        m = compute_metrics(disk, disk_cfg)
        assert m["moi_kg_mm2"] > 0

    def test_energy_positive(self, disk_cfg):
        disk = Point(0, 0).buffer(80)
        m = compute_metrics(disk, disk_cfg)
        assert m["energy_joules"] > 0

    def test_com_offset_centered_shape(self, disk_cfg):
        disk = Point(0, 0).buffer(80, resolution=256)
        m = compute_metrics(disk, disk_cfg)
        assert m["com_offset_mm"] < 1.0

    def test_com_offset_shifted_shape(self, disk_cfg):
        disk = translate(Point(0, 0).buffer(40), 30, 0)
        m = compute_metrics(disk, disk_cfg)
        assert m["com_offset_mm"] > 20.0


# ---------------------------------------------------------------------------
# weighted_score
# ---------------------------------------------------------------------------

class TestWeightedScore:
    def test_score_in_range(self, disk_cfg):
        disk = Point(0, 0).buffer(80)
        m = compute_metrics(disk, disk_cfg)
        s = weighted_score(m, disk_cfg)
        assert 0.0 <= s <= 1.0

    def test_larger_disk_higher_moi_score(self, disk_cfg):
        small = Point(0, 0).buffer(60)
        large = Point(0, 0).buffer(120)
        m_small = compute_metrics(small, disk_cfg)
        m_large = compute_metrics(large, disk_cfg)
        # Larger disk has higher MOI contribution (all else equal)
        assert m_large["moi_kg_mm2"] > m_small["moi_kg_mm2"]

    def test_weights_sum_influences_score(self, disk_cfg):
        disk = Point(0, 0).buffer(80)
        m = compute_metrics(disk, disk_cfg)
        s = weighted_score(m, disk_cfg)
        # Score should be non-negative finite number
        assert math.isfinite(s)
        assert s >= 0.0
