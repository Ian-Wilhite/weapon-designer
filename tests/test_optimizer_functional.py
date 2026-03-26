"""Tests for optimizer_functional.py — Stage A, Stage B, and entry point."""

from __future__ import annotations

import numpy as np
import pytest

from weapon_designer.config import (
    WeaponConfig, Material, Mounting, Envelope,
    OptimizationParams, OptimizationWeights,
)
from weapon_designer.functional_profiles import get_functional_bounds


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def disk_cfg_func():
    """Minimal disk config with very few iterations for fast tests."""
    return WeaponConfig(
        material=Material("AR500", 7850, 1400, 50),
        weapon_style="disk",
        sheet_thickness_mm=10.0,
        weight_budget_kg=3.0,
        rpm=8000,
        mounting=Mounting(25.4, 50, 4, 6.5),
        envelope=Envelope(max_radius_mm=100.0),
        optimization=OptimizationParams(
            max_iterations=5,
            population_size=5,
            phase1_iters=5,
            phase2_iters=3,
            n_bspline_points=8,
            fea_coarse_spacing_mm=15.0,
            fea_interval=0,
        ),
    )


# ---------------------------------------------------------------------------
# Task 1.2 tests
# ---------------------------------------------------------------------------

def test_functional_bounds_shape(disk_cfg_func):
    """get_functional_bounds() must return exactly 6 (lo, hi) tuples."""
    bounds = get_functional_bounds(disk_cfg_func)
    assert len(bounds) == 6, f"Expected 6 bounds, got {len(bounds)}"
    for lo, hi in bounds:
        assert lo < hi, f"Bound lo={lo} must be < hi={hi}"


def test_stage_a_valid_score(disk_cfg_func, tmp_path):
    """Stage A runs without exception and returns a score in [0, 1]."""
    from weapon_designer.optimizer_functional import optimize_functional_stage_a

    result = optimize_functional_stage_a(disk_cfg_func, tmp_path, log_fn=lambda x: None)
    assert "best_params" in result
    assert "best_score" in result
    assert "history" in result

    score = result["best_score"]
    assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"
    assert len(result["best_params"]) == 6, "Stage A params must have 6 elements"


def test_stage_b_lifts_to_ndim(disk_cfg_func, tmp_path):
    """Stage B output radii shape matches n_bspline_points."""
    from weapon_designer.optimizer_functional import (
        optimize_functional_stage_a,
        optimize_functional_stage_b,
    )

    result_a = optimize_functional_stage_a(disk_cfg_func, tmp_path, log_fn=lambda x: None)
    result_b = optimize_functional_stage_b(
        disk_cfg_func, result_a["best_params"], tmp_path, log_fn=lambda x: None
    )

    N = disk_cfg_func.optimization.n_bspline_points
    assert "best_radii" in result_b
    assert result_b["best_radii"].shape == (N,), (
        f"Expected radii shape ({N},), got {result_b['best_radii'].shape}"
    )
    assert "r_ref" in result_b
    assert result_b["r_ref"].shape == (N,)


# ---------------------------------------------------------------------------
# Task 5.1 trust-region test
# ---------------------------------------------------------------------------

def test_trust_region_rejects_distant_candidates(disk_cfg_func):
    """When trust_region_enabled=True and _TR_BEST_X is set, a candidate
    further than trust_region_d_max_mm from the best must return 1.0 without
    running FEA/metrics."""
    import weapon_designer.optimizer_enhanced as opt_mod
    from weapon_designer.optimizer_enhanced import _profile_objective_bspline

    import copy
    cfg = copy.deepcopy(disk_cfg_func)
    cfg.optimization.trust_region_enabled = True
    cfg.optimization.trust_region_d_max_mm = 1.0  # very tight radius

    # Set a reference best-x (all zeros in bspline space, unlikely to be built)
    N = cfg.optimization.n_bspline_points
    opt_mod._TR_BEST_X = np.zeros(N)

    # Candidate that is far away (radii all at max_r)
    max_r = cfg.envelope.max_radius_mm
    x_far = np.full(N, max_r)

    result = _profile_objective_bspline(x_far, cfg)
    assert result == 1.0, (
        f"Trust-region should reject far candidate with 1.0, got {result}"
    )

    # Cleanup
    opt_mod._TR_BEST_X = None
