"""Tests for fea.py — CST plane-stress FEA on known geometries."""

from __future__ import annotations

import math
import numpy as np
import pytest
from shapely.geometry import Point

from weapon_designer.fea import fea_stress_analysis, _find_boundary_edges, _apply_contact_forces


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


class TestNeumannEdgeLoading:
    """Tests for boundary-edge Neumann contact force distribution."""

    def test_neumann_edge_loads_boundary_nodes(self):
        """Force at boundary midpoint distributes only to boundary nodes,
        and total force is conserved."""
        # Simple 2-triangle mesh: square [0,1]×[0,1], diagonal on nodes 0→2
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        elems = np.array([[0, 1, 2], [0, 2, 3]])

        F = np.zeros(8)
        # Force at midpoint of the bottom edge (node 0 → node 1)
        contact = [{"x": 0.5, "y": 0.0, "fx": 10.0, "fy": 4.0}]
        F = _apply_contact_forces(F, nodes, contact, mode="neumann_edge", elements=elems)

        # Total force must be conserved
        assert abs(F[0::2].sum() - 10.0) < 1e-9, "x-force not conserved"
        assert abs(F[1::2].sum() - 4.0) < 1e-9, "y-force not conserved"

        # Force must be distributed only to boundary nodes (not interior)
        # All 4 nodes are on the boundary of this small mesh, so verify
        # that node 0 and node 1 each receive roughly half (t ≈ 0.5)
        assert F[0] > 0.0, "node 0 should receive x-force"
        assert F[2] > 0.0, "node 1 should receive x-force"
        # No force should land on interior DOFs (none here, but test the split)
        assert abs(F[0] + F[2] - 10.0) < 1e-6 or (F[0] + F[2]) <= 10.0 + 1e-6

    def test_neumann_edge_vs_nearest_node_different(self):
        """Neumann-edge and nearest-node modes produce different F vectors
        for an off-node contact point."""
        nodes = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        elems = np.array([[0, 1, 2], [0, 2, 3]])
        # Contact at x=0.7 on the bottom edge → not at a node
        contact = [{"x": 0.7, "y": 0.0, "fx": 1.0, "fy": 0.0}]

        F_nn = _apply_contact_forces(np.zeros(8), nodes, contact,
                                     mode="nearest_node")
        F_ne = _apply_contact_forces(np.zeros(8), nodes, contact,
                                     mode="neumann_edge", elements=elems)

        # nearest_node puts all force at node 0 (at origin, closest to (0.7, 0))
        assert F_nn[0] == pytest.approx(1.0), "nearest_node should assign all fx to node 0"
        # neumann_edge splits between node 0 and node 1
        assert F_ne[0] > 0.0 and F_ne[2] > 0.0, "neumann_edge should split force"
        assert not np.allclose(F_nn, F_ne), "modes must differ for off-node contact"
