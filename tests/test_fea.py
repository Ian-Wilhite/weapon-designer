"""Tests for fea.py — CST plane-stress FEA on known geometries."""

from __future__ import annotations

import math
import numpy as np
import pytest
from shapely.geometry import Point

from weapon_designer.fea import fea_stress_analysis


@pytest.fixture
def annular_ring():
    """Known-geometry annular ring: outer 80mm, inner 15mm."""
    outer = Point(0, 0).buffer(80, resolution=64)
    inner = Point(0, 0).buffer(15, resolution=64)
    return outer.difference(inner)


@pytest.fixture
def solid_disk():
    return Point(0, 0).buffer(80, resolution=64)


class TestFEAStressAnalysis:
    def test_returns_dict_with_required_keys(self, annular_ring):
        result = fea_stress_analysis(
            annular_ring,
            rpm=5000,
            density_kg_m3=7850,
            thickness_mm=10.0,
            yield_strength_mpa=1400.0,
            bore_diameter_mm=25.4,
        )
        for key in ("peak_stress_mpa", "safety_factor", "fea_score", "n_elements"):
            assert key in result, f"Missing key: {key}"

    def test_safe_design_positive_sf(self, annular_ring):
        """Low-speed annular ring well below yield → SF > 1."""
        result = fea_stress_analysis(
            annular_ring,
            rpm=2000,
            density_kg_m3=7850,
            thickness_mm=10.0,
            yield_strength_mpa=1400.0,
            bore_diameter_mm=25.4,
        )
        assert result["safety_factor"] > 1.0

    def test_peak_stress_positive(self, annular_ring):
        result = fea_stress_analysis(
            annular_ring,
            rpm=5000,
            density_kg_m3=7850,
            thickness_mm=10.0,
            yield_strength_mpa=1400.0,
            bore_diameter_mm=25.4,
        )
        assert result["peak_stress_mpa"] > 0

    def test_fea_score_in_range(self, annular_ring):
        result = fea_stress_analysis(
            annular_ring,
            rpm=3000,
            density_kg_m3=7850,
            thickness_mm=10.0,
            yield_strength_mpa=1400.0,
            bore_diameter_mm=25.4,
        )
        assert 0.0 <= result["fea_score"] <= 1.0

    def test_higher_rpm_higher_stress(self, annular_ring):
        """More centrifugal load at higher RPM → higher peak stress."""
        r_low = fea_stress_analysis(
            annular_ring, rpm=1000, density_kg_m3=7850,
            thickness_mm=10.0, yield_strength_mpa=1400.0, bore_diameter_mm=25.4,
        )
        r_high = fea_stress_analysis(
            annular_ring, rpm=8000, density_kg_m3=7850,
            thickness_mm=10.0, yield_strength_mpa=1400.0, bore_diameter_mm=25.4,
        )
        assert r_high["peak_stress_mpa"] > r_low["peak_stress_mpa"]

    def test_n_elements_positive(self, annular_ring):
        result = fea_stress_analysis(
            annular_ring,
            rpm=5000,
            density_kg_m3=7850,
            thickness_mm=10.0,
            yield_strength_mpa=1400.0,
            bore_diameter_mm=25.4,
        )
        assert result["n_elements"] > 0

    def test_solid_disk_safe_at_low_rpm(self, solid_disk):
        result = fea_stress_analysis(
            solid_disk, rpm=1000, density_kg_m3=7850,
            thickness_mm=10.0, yield_strength_mpa=1400.0, bore_diameter_mm=25.4,
        )
        assert result["safety_factor"] > 1.0
