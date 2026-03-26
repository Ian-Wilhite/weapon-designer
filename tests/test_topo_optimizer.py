"""Tests for topo_optimizer.py — SIMP topology optimisation (Phase 2 alternative).

Most tests are marked @pytest.mark.slow and are skipped in normal CI runs.
Run with:  pytest -m slow  to include them.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, MultiPolygon


# ---------------------------------------------------------------------------
# Minimal config and outer profile for quick topology tests
# ---------------------------------------------------------------------------

@pytest.fixture
def small_disk_poly():
    """60mm solid disk — small enough for fast topo mesh."""
    return Point(0, 0).buffer(60, resolution=64)


@pytest.fixture
def topo_cfg():
    """Config tuned for fast topology tests."""
    from weapon_designer.config import (
        WeaponConfig, Material, Mounting, Envelope,
        OptimizationParams, OptimizationWeights,
    )
    return WeaponConfig(
        material=Material(name="AR500", density_kg_m3=7850, yield_strength_mpa=1400),
        weapon_style="disk",
        sheet_thickness_mm=10.0,
        weight_budget_kg=0.5,   # ~56% of solid mass at 60mm radius
        rpm=5000,
        mounting=Mounting(
            bore_diameter_mm=12.0,
            bolt_circle_diameter_mm=25.0,
            num_bolts=3,
            bolt_hole_diameter_mm=4.0,
        ),
        envelope=Envelope(max_radius_mm=60.0),
        optimization=OptimizationParams(
            weights=OptimizationWeights(
                moment_of_inertia=0.35, bite=0.10,
                structural_integrity=0.25, mass_utilization=0.10,
                balance=0.10, impact_zone=0.10,
            ),
            topo_n_iter=8,              # very few iterations for speed
            topo_mesh_spacing_mm=10.0,  # coarse mesh
            topo_p_simp=3.0,
            topo_r_min_factor=2.0,
            topo_w_compliance=0.5,
            topo_frame_interval=0,      # disable GIF output in tests
        ),
    )


# ---------------------------------------------------------------------------
# Import guard — skip module entirely if scipy not available
# ---------------------------------------------------------------------------

pytest.importorskip("scipy", reason="scipy required for topo_optimizer")


# ---------------------------------------------------------------------------
# Fast unit tests (no heavy computation)
# ---------------------------------------------------------------------------

class TestTopoOptimizerImport:
    def test_module_importable(self):
        """topo_optimizer must import without error."""
        import weapon_designer.topo_optimizer  # noqa: F401

    def test_run_topo_function_exists(self):
        from weapon_designer.topo_optimizer import topology_optimize
        assert callable(topology_optimize)


class TestTopoMeshParams:
    """Verify meshing parameter logic without running full optimisation."""

    def test_volume_fraction_from_mass_budget(self, topo_cfg, small_disk_poly):
        """Volume fraction should be computable from mass_budget / solid_mass."""
        from weapon_designer.physics import polygon_mass_kg
        solid_mass = polygon_mass_kg(
            small_disk_poly,
            topo_cfg.sheet_thickness_mm,
            topo_cfg.material.density_kg_m3,
        )
        vf = topo_cfg.weight_budget_kg / solid_mass
        # vf may exceed 1 if budget > solid mass (impossible design, clamped to 1)
        assert vf > 0

    def test_config_topo_keys_exist(self, topo_cfg):
        opt = topo_cfg.optimization
        assert hasattr(opt, "topo_n_iter")
        assert hasattr(opt, "topo_mesh_spacing_mm")
        assert hasattr(opt, "topo_p_simp")
        assert hasattr(opt, "topo_r_min_factor")
        assert hasattr(opt, "topo_w_compliance")
        assert hasattr(opt, "topo_edge_offset_mm")


# ---------------------------------------------------------------------------
# Slow integration tests (full optimisation run)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTopoOptimisationIntegration:
    """Full topology optimisation smoke tests — run with pytest -m slow."""

    def _run(self, small_disk_poly, topo_cfg, tmp_path):
        """Helper: call topology_optimize and return the weapon polygon."""
        from weapon_designer.topo_optimizer import topology_optimize
        result = topology_optimize(
            solid_polygon=small_disk_poly,
            cfg=topo_cfg,
            case_dir=tmp_path,
        )
        return result, result.get("weapon_polygon")

    def test_run_returns_polygon(self, topo_cfg, small_disk_poly, tmp_path):
        """topology_optimize should return a dict with a non-empty weapon_polygon."""
        result, result_poly = self._run(small_disk_poly, topo_cfg, tmp_path)
        assert result_poly is not None
        assert isinstance(result_poly, (Polygon, MultiPolygon))
        assert not result_poly.is_empty
        assert result_poly.area > 0

    def test_result_subset_of_outer_profile(self, topo_cfg, small_disk_poly, tmp_path):
        """Topo result must be geometrically contained within the outer profile."""
        _, result_poly = self._run(small_disk_poly, topo_cfg, tmp_path)
        if result_poly is None:
            pytest.skip("topology_optimize returned no weapon_polygon")

        # Result should not extend significantly outside outer profile
        outside = result_poly.difference(small_disk_poly.buffer(2.0))
        assert outside.area < result_poly.area * 0.05  # < 5% outside

    def test_volume_fraction_approximately_met(self, topo_cfg, small_disk_poly, tmp_path):
        """Result mass should be within 15% of the weight budget."""
        from weapon_designer.physics import polygon_mass_kg

        _, result_poly = self._run(small_disk_poly, topo_cfg, tmp_path)
        if result_poly is None:
            pytest.skip("topology_optimize returned no weapon_polygon")

        result_mass = polygon_mass_kg(
            result_poly,
            topo_cfg.sheet_thickness_mm,
            topo_cfg.material.density_kg_m3,
        )
        budget = topo_cfg.weight_budget_kg
        # Allow ±15% tolerance
        assert abs(result_mass - budget) / max(budget, 1e-6) < 0.15

    def test_convergence_plot_saved(self, topo_cfg, small_disk_poly, tmp_path):
        """Convergence PNG should be written to case_dir."""
        import matplotlib
        matplotlib.use("Agg")

        self._run(small_disk_poly, topo_cfg, tmp_path)
        # Look for any convergence plot PNG
        pngs = list(tmp_path.glob("*.png")) + list(tmp_path.rglob("*.png"))
        assert len(pngs) > 0, "Expected at least one PNG output from topo optimizer"

    def test_result_is_valid_geometry(self, topo_cfg, small_disk_poly, tmp_path):
        """Result polygon should be valid or fixable with .buffer(0)."""
        _, result_poly = self._run(small_disk_poly, topo_cfg, tmp_path)
        if result_poly is None:
            pytest.skip("topology_optimize returned no weapon_polygon")

        assert result_poly.is_valid or result_poly.buffer(0).area > 0
