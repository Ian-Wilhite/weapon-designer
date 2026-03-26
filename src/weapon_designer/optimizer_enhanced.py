"""Enhanced two-phase optimizer for research comparison.

Baseline optimizer  (optimizer.py) is left completely unchanged.
This module provides a parallel implementation with four improvements:

1. FEA-in-loop structural scoring
   Every objective evaluation runs a coarse-mesh FEA instead of the
   geometric proxy (min-wall / section-width).  The optimizer receives
   real stress feedback during search rather than waiting until the end.

2. Polar CAD cutout parameterisation  (r, φ, a, b, n)
   Phase 2 uses superellipse pockets at polar positions (r, φ).  The
   phase angle φ is a free optimizer variable; the orientation is fixed
   analytically (tangential = φ + 90°).  No symmetry is assumed — all
   asymmetric layouts are reachable.

3. Analytical mass normalisation after Phase 2
   After the optimizer finds hole positions and shapes, a closed-form
   scaling pass adjusts every hole's (a, b) semi-axes uniformly so that
   the assembled weapon hits the mass budget exactly.  Mass is thereby
   removed from the optimisation objective entirely — the optimizer
   focuses on MOI, bite, structural integrity, balance, and impact zone.
   The scaling iterates up to 3 times for accuracy (overlapping holes
   or boundary clipping break the first-order s² approximation slightly).

4. FEA frame export for GIF visualisation
   At every fea_interval-th callback step the current best solution is
   assembled, FEA'd with a medium-quality mesh, and saved as a numbered
   PNG.  After both phases, the frame sequences are stitched into GIFs.

Iteration budget
────────────────
The enhanced mode uses fewer DE iterations (50 % of baseline by default)
because each evaluation is ~3–10× more expensive (FEA in loop).
Total wall-clock time is comparable; the optimizer gets far better gradient
signal per generation.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from .config import WeaponConfig
from .parametric import build_weapon_polygon, _cutout_stride
from .parametric_cad import (
    make_cutouts_cad,
    get_cutout_bounds_cad,
    CUTOUT_STRIDE_CAD,
    make_cutouts_polar,
    get_cutout_bounds_polar,
    CUTOUT_STRIDE_POLAR,
    mass_normalize_cutouts,
)
from .bspline_profile import get_bspline_bounds, N_BSPLINE_DEFAULT
from .profile_builder import build_profile, get_profile_bounds, is_single_phase_profile
from .geometry import assemble_weapon
from .objectives_enhanced import (
    compute_metrics_enhanced,
    weighted_score_enhanced,
)
from .objectives_physical import compute_physical_score as _physical_score
from .objectives import impact_zone_score
from .constraints import validate_geometry, constraint_penalty, check_envelope, is_connected, check_min_feature_size
from .archetypes import seed_population_from_archetypes
from .optimizer import _get_profile_bounds          # reuse — unchanged
from .fea import fea_stress_analysis_with_mesh
from .fea_viz import render_fea_frame, export_gif, render_spiral_contact_frame
from .spiral_contact import analyse_contacts as _spiral_analyse_cb, contact_forces as _contact_forces_cb
from .physics import polygon_mass_kg as _poly_mass_kg
from .topo_optimizer import topology_optimize as _topo_phase2
from .staged_eval import EvalGate

# ---------------------------------------------------------------------------
# Evaluation-mode dispatcher
# ---------------------------------------------------------------------------

def _score_from_metrics(
    metrics: dict,
    poly,
    cfg: WeaponConfig,
) -> float:
    """Route to the correct scoring function based on cfg.optimization.evaluation_mode.

    "enhanced"  — original weighted-sum with kinematic bite + FEA (default)
    "physical"  — single E_transfer score with hard constraints (no weights)
    """
    mode = getattr(cfg.optimization, "evaluation_mode", "enhanced")
    if mode == "physical":
        return _physical_score(poly, cfg, metrics=metrics)
    return weighted_score_enhanced(metrics, cfg)


# Process-local gate; set by optimize_enhanced() before the DE workers start.
# Each forked worker inherits the instance, so no cross-process sharing needed.
_EVAL_GATE: EvalGate | None = None

# Trust-region: best solution vector from the callback, visible to the objective
# only when workers=1 (single-process mode forced by trust_region_enabled).
_TR_BEST_X: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Shape-distance helper (P0-C)
# ---------------------------------------------------------------------------

def _shape_distance(r1: np.ndarray, r2: np.ndarray, n_angles: int = 360) -> float:
    """RMS difference between two radii vectors resampled to n_angles uniform angles.

    Interpolates both to a common grid (periodic, linear) so vectors of
    different lengths are comparable.  Returns shape distance in mm.
    """
    from scipy.interpolate import interp1d

    def to_uniform(r_vec: np.ndarray) -> np.ndarray:
        N = len(r_vec)
        theta_in = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
        # Wrap for periodic interpolation
        theta_w = np.concatenate([theta_in - 2.0 * np.pi, theta_in, theta_in + 2.0 * np.pi])
        r_w = np.tile(r_vec, 3)
        f = interp1d(theta_w, r_w, kind="linear")
        theta_out = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
        return f(theta_out)

    return float(np.sqrt(np.mean((to_uniform(r1) - to_uniform(r2)) ** 2)))


# ---------------------------------------------------------------------------
# Phase 2 penalty helper (no mass check — mass satisfied analytically)
# ---------------------------------------------------------------------------

def _constraint_penalty_no_mass(
    weapon,
    cfg: WeaponConfig,
    cutout_polys: list | None = None,
) -> float:
    """Constraint penalty for Phase 2 (polar) objectives.

    Identical to constraint_penalty() EXCEPT the mass check is omitted.
    During Phase 2 the weapon is intentionally over the mass budget so
    the analytical normalisation pass (mass_normalize_cutouts) can set
    hole sizes correctly.  Penalising mass here would kill every valid
    solution and make Phase 2 degenerate.
    """
    if weapon.is_empty:
        return 0.0
    if not is_connected(weapon):
        return 0.0

    if cutout_polys:
        from .geometry import check_mounting_clearance
        if not check_mounting_clearance(cfg.mounting, cutout_polys):
            return 0.0

    penalty = 1.0
    if not check_envelope(weapon, cfg):
        penalty *= 0.3
    if not check_min_feature_size(weapon, cfg.optimization.min_feature_size_mm):
        penalty *= 0.5
    return float(penalty)


# ---------------------------------------------------------------------------
# Helpers: build weapon with CAD cutouts from a split parameter vector
# ---------------------------------------------------------------------------

def _build_weapon_enhanced(
    profile_params: np.ndarray,
    cutout_params: np.ndarray,
    cfg: WeaponConfig,
) -> tuple:
    """Build outer profile (Fourier) + CAD cutouts, return (outer, weapon, cutout_polys)."""
    style = cfg.weapon_style

    # Profile: reuse existing Fourier build_weapon_polygon — pass zero-length
    # cutout suffix so the profile code path is unchanged.
    C_baseline = cfg.optimization.num_cutout_pairs
    S_baseline = _cutout_stride(cfg)
    x_profile_full = np.concatenate([profile_params, np.zeros(C_baseline * S_baseline)])
    outer, params, _ = build_weapon_polygon(x_profile_full, cfg)

    # Symmetry for cutout replication
    if style == "bar":
        symmetry = 2
    elif style == "eggbeater":
        num_beaters = params.get("num_beaters", 3)
        symmetry = int(num_beaters)
    else:
        symmetry = 1

    # CAD cutouts
    C = cfg.optimization.num_cutout_pairs
    if C > 0 and cutout_params.size > 0:
        cutout_polys = make_cutouts_cad(cutout_params, C, symmetry=symmetry)
    else:
        cutout_polys = []

    weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
    weapon = validate_geometry(weapon)
    return outer, weapon, cutout_polys


# ---------------------------------------------------------------------------
# Helpers: build weapon with POLAR cutouts
# ---------------------------------------------------------------------------

def _build_weapon_polar(
    profile_params: np.ndarray,
    cutout_params: np.ndarray,
    cfg: WeaponConfig,
) -> tuple:
    """Build outer profile (Fourier) + polar superellipse cutouts.

    Returns (outer, weapon, cutout_polys).
    Symmetry is NOT applied — holes sit at their individual (r, φ) positions.
    """
    C_baseline = cfg.optimization.num_cutout_pairs
    S_baseline = _cutout_stride(cfg)
    x_profile_full = np.concatenate([profile_params, np.zeros(C_baseline * S_baseline)])
    outer, _, _ = build_weapon_polygon(x_profile_full, cfg)

    C = cfg.optimization.num_cutout_pairs
    if C > 0 and cutout_params.size > 0:
        cutout_polys = make_cutouts_polar(cutout_params, C)
    else:
        cutout_polys = []

    weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
    weapon = validate_geometry(weapon)
    return outer, weapon, cutout_polys


def _weapon_no_cutouts(profile_params: np.ndarray, cfg: WeaponConfig):
    """Build weapon with mounting holes only (no weight-reduction pockets).

    Used once before Phase 2 to establish the reference area for mass
    normalisation.  Cached as a float (area) to avoid pickling Shapely
    objects across multiprocessing workers.
    """
    C_baseline = cfg.optimization.num_cutout_pairs
    S_baseline = _cutout_stride(cfg)
    x_full = np.concatenate([profile_params, np.zeros(C_baseline * S_baseline)])
    outer, _, _ = build_weapon_polygon(x_full, cfg)
    weapon = assemble_weapon(outer, cfg.mounting, [])
    weapon = validate_geometry(weapon)
    return outer, weapon


# ---------------------------------------------------------------------------
# Helpers: B-spline profile builders
# ---------------------------------------------------------------------------

def _bspline_outer(bspline_radii: np.ndarray, cfg: WeaponConfig):
    """Build outer profile polygon from spline radii using the configured profile type.

    Returns a Shapely Polygon (or None on failure).
    Dispatches to build_profile() so that profile_type config key is respected.
    """
    profile_type = getattr(cfg.optimization, "profile_type", "bspline")
    return build_profile(profile_type, bspline_radii, cfg)


def _build_weapon_bspline_polar(
    bspline_radii: np.ndarray,
    cutout_params: np.ndarray,
    cfg: WeaponConfig,
) -> tuple:
    """Build weapon: B-spline outer profile + polar superellipse cutouts.

    Returns (outer, weapon, cutout_polys).
    No N-fold symmetry is applied — each hole sits at its individual (r, φ).
    """
    outer = _bspline_outer(bspline_radii, cfg)
    if outer is None:
        from shapely.geometry import Point
        outer = Point(0, 0).buffer(cfg.envelope.max_radius_mm * 0.5)

    C = cfg.optimization.num_cutout_pairs
    if C > 0 and cutout_params.size > 0:
        cutout_polys = make_cutouts_polar(cutout_params, C)
    else:
        cutout_polys = []

    weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
    weapon = validate_geometry(weapon)
    return outer, weapon, cutout_polys


def _weapon_no_cutouts_bspline(bspline_radii: np.ndarray, cfg: WeaponConfig):
    """Build B-spline weapon with mounting holes only (no WR pockets).

    Returns (outer, weapon).  Used once to establish the reference area
    for the analytical mass-normalisation pass.
    """
    outer = _bspline_outer(bspline_radii, cfg)
    if outer is None:
        from shapely.geometry import Point
        outer = Point(0, 0).buffer(cfg.envelope.max_radius_mm * 0.5)

    weapon = assemble_weapon(outer, cfg.mounting, [])
    weapon = validate_geometry(weapon)
    return outer, weapon


# ---------------------------------------------------------------------------
# Phase 1 objective  (B-spline profile, no cutouts, FEA structural scoring)
# ---------------------------------------------------------------------------

def _profile_objective_bspline(x: np.ndarray, cfg: WeaponConfig) -> float:
    """Phase-1 objective using a B-spline outer profile and coarse FEA."""
    try:
        # Trust-region early reject: if candidate is too far from current best,
        # return 1.0 without building geometry or running FEA.
        # Only active when workers=1 (trust_region_enabled forces single-process).
        if getattr(cfg.optimization, "trust_region_enabled", False) and _TR_BEST_X is not None:
            d_max = getattr(cfg.optimization, "trust_region_d_max_mm", 5.0)
            if _shape_distance(x, _TR_BEST_X) > d_max:
                return 1.0

        _, weapon, _ = _build_weapon_bspline_polar(x, np.zeros(0), cfg)

        if weapon.is_empty or weapon.area < 1.0:
            return 1.0

        # Staged evaluation gate: run Stage 0+1, skip FEA if below threshold
        gate = _EVAL_GATE
        if gate is not None:
            if not gate.stage0_check(weapon):
                return 1.0
            s1 = gate.stage1_score(weapon)
            if not gate.should_run_fea(s1):
                return float(-s1)   # return cheap score directly (negated)

        metrics = compute_metrics_enhanced(weapon, cfg)
        score   = _score_from_metrics(metrics, weapon, cfg)
        if not check_envelope(weapon, cfg):
            score *= 0.3
        return -score

    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Phase 2 objective  (polar cutouts on B-spline profile, mass-free)
# ---------------------------------------------------------------------------

def _cutout_objective_bspline_polar(
    x_cutout: np.ndarray,
    bspline_radii: np.ndarray,
    cfg: WeaponConfig,
) -> float:
    """Phase-2 objective: polar superellipse cutouts on a frozen B-spline profile.

    Mass is included as a *soft* penalty rather than a hard kill so the
    DE has a gradient to follow toward the budget.  After Phase 2, the
    analytical mass_normalize_cutouts() pass scales (a, b) to exactly
    hit the budget — so the residual mass error here is acceptable.

    The soft mass factor f_mass:
        mass_util ≤ 1.0  →  f_mass = mass_util   (reward more material used)
        1.0 < util ≤ 2.0 →  f_mass = 2.0 − util  (linear decay; 0 at 2× budget)
        util > 2.0        →  f_mass = 0.0
    """
    try:
        _, weapon, cutout_polys = _build_weapon_bspline_polar(bspline_radii, x_cutout, cfg)

        if weapon.is_empty or weapon.area < 1.0:
            return 1.0

        # Staged evaluation gate
        gate = _EVAL_GATE
        if gate is not None:
            if not gate.stage0_check(weapon):
                return 1.0
            s1 = gate.stage1_score(weapon)
            if not gate.should_run_fea(s1):
                return float(-s1)

        metrics = compute_metrics_enhanced(weapon, cfg)
        score   = _score_from_metrics(metrics, weapon, cfg)

        # Soft mass multiplier — gradient-preserving, rewards approaching budget.
        # Applied as a multiplier (not additive) so mass under/over-run scales the
        # whole score rather than adding an independent term.
        mu = metrics["mass_utilization"]
        if mu <= 1.0:
            f_mass = mu
        elif mu <= 2.0:
            f_mass = max(0.0, 2.0 - mu)
        else:
            f_mass = 0.0

        penalty = _constraint_penalty_no_mass(weapon, cfg, cutout_polys=cutout_polys)
        return -(score * penalty * f_mass)

    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Phase 2 objective  (POLAR cutouts, mass removed from score)
# ---------------------------------------------------------------------------

def _cutout_objective_polar(
    x_cutout: np.ndarray,
    profile_params: np.ndarray,
    cfg: WeaponConfig,
) -> float:
    """Phase-2 objective using polar cutouts and FEA structural scoring.

    Mass utilisation is intentionally excluded from this score.
    The mass target is satisfied analytically after Phase 2 completes,
    so wasting optimisation budget on it here is unnecessary.
    The optimizer can focus entirely on MOI, bite, structural integrity,
    balance, and impact zone.
    """
    try:
        _, weapon, cutout_polys = _build_weapon_polar(profile_params, x_cutout, cfg)

        if weapon.is_empty or weapon.area < 1.0:
            return 1.0

        metrics = compute_metrics_enhanced(weapon, cfg)
        score   = _score_from_metrics(metrics, weapon, cfg)

        mu = metrics["mass_utilization"]
        f_mass = mu if mu <= 1.0 else max(0.0, 2.0 - mu)

        penalty = _constraint_penalty_no_mass(weapon, cfg, cutout_polys=cutout_polys)
        return -(score * penalty * f_mass)

    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Phase 1 objective  (profile shape, no cutouts, FEA structural scoring)
# ---------------------------------------------------------------------------

def _profile_objective_enhanced(x: np.ndarray, cfg: WeaponConfig) -> float:
    """Enhanced Phase-1 objective: Fourier profile + coarse FEA structural score."""
    try:
        _, weapon, _ = _build_weapon_enhanced(x, np.zeros(0), cfg)

        if weapon.is_empty or weapon.area < 1.0:
            return 1.0

        metrics = compute_metrics_enhanced(weapon, cfg)
        score   = _score_from_metrics(metrics, weapon, cfg)
        if not check_envelope(weapon, cfg):
            score *= 0.3
        return -score

    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Phase 2 objective  (CAD cutouts with fixed profile, FEA structural scoring)
# ---------------------------------------------------------------------------

def _cutout_objective_enhanced(
    x_cutout: np.ndarray,
    profile_params: np.ndarray,
    cfg: WeaponConfig,
) -> float:
    """Enhanced Phase-2 objective: CAD cutouts + FEA structural score."""
    try:
        _, weapon, cutout_polys = _build_weapon_enhanced(profile_params, x_cutout, cfg)

        if weapon.is_empty or weapon.area < 1.0:
            return 1.0

        metrics = compute_metrics_enhanced(weapon, cfg)
        score   = _score_from_metrics(metrics, weapon, cfg)
        penalty = constraint_penalty(weapon, cfg, cutout_polys=cutout_polys)

        return -(score * penalty)

    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Callback factory with FEA frame export
# ---------------------------------------------------------------------------

class _FEACallback:
    """Optimizer callback that saves FEA stress-map frames for GIF export.

    Runs FEA on the current best solution every fea_interval steps and
    writes a numbered PNG to frame_dir.
    """

    def __init__(
        self,
        cfg: WeaponConfig,
        profile_params: np.ndarray | None,  # None during Phase 1
        frame_dir: Path,
        phase_label: str,
        log_fn,
        fea_interval: int,
        polar: bool = False,               # True → polar cutouts for Phase 2
        bspline: bool = False,             # True → B-spline profile instead of Fourier
        patience: int = 15,                # steps without improvement before early stop
        min_delta: float = 0.002,          # minimum score improvement to reset patience
        spiral_frame_dir: Path | None = None,  # directory for spiral contact frames
    ):
        self.cfg              = cfg
        self.profile_params   = profile_params
        self.frame_dir        = frame_dir
        self.phase_label      = phase_label
        self._log             = log_fn
        self.fea_interval     = fea_interval   # >0 = enabled; saved every step
        self.polar            = polar
        self.bspline          = bspline
        self.patience         = patience
        self.min_delta        = min_delta
        self.spiral_frame_dir = spiral_frame_dir
        self.step             = 0
        self.frame_idx        = 0
        self.start_time       = time.time()
        self.phase_start_time = time.time()   # Pareto: phase-local timer
        self.n_fea_calls      = 0             # Pareto: cumulative FEA call count
        self.history: list[dict] = []
        self._best_score: float = -float("inf")
        self._best_x: np.ndarray | None = None
        self._steps_without_improvement: int = 0
        self._cb_times: list[float] = []   # wall-clock ms per callback

    def __call__(self, xk: np.ndarray, convergence: float) -> bool | None:
        _cb_t0 = time.perf_counter()
        self.step += 1
        elapsed = time.time() - self.start_time

        try:
            if self.profile_params is None:
                # Phase 1: xk is the profile vector
                if self.bspline:
                    _, weapon, _ = _build_weapon_bspline_polar(xk, np.zeros(0), self.cfg)
                else:
                    _, weapon, _ = _build_weapon_enhanced(xk, np.zeros(0), self.cfg)
            elif self.polar and self.bspline:
                # Phase 2: B-spline profile + polar cutouts
                _, weapon, _ = _build_weapon_bspline_polar(self.profile_params, xk, self.cfg)
            elif self.polar:
                # Phase 2: Fourier profile + polar cutouts
                _, weapon, _ = _build_weapon_polar(self.profile_params, xk, self.cfg)
            else:
                # Phase 2: Fourier profile + Cartesian CAD cutouts
                _, weapon, _ = _build_weapon_enhanced(self.profile_params, xk, self.cfg)

            # When GIF frames are enabled, request mesh data in the same FEA call
            # so the frame renderer doesn't need a second FEA solve.
            _need_mesh = self.fea_interval > 0
            metrics = compute_metrics_enhanced(weapon, self.cfg, return_mesh=_need_mesh)
            score   = _score_from_metrics(metrics, weapon, self.cfg)
            self.n_fea_calls += 1   # Pareto: count each successful FEA evaluation

            # ── Shape distance from current best ─────────────────────────
            shape_dist = 0.0
            if self._best_x is not None:
                try:
                    shape_dist = _shape_distance(xk, self._best_x)
                except Exception:
                    shape_dist = 0.0

            # ── Constraint penalty for Pareto tracking ────────────────────
            try:
                from .constraints import constraint_penalty as _cp
                _penalty_val = round(_cp(weapon, self.cfg), 4)
            except Exception:
                _penalty_val = 1.0

            entry = {
                "phase":            self.phase_label,
                "step":             self.step,
                "elapsed_s":        round(elapsed, 1),
                "phase_elapsed_s":  round(time.time() - self.phase_start_time, 1),
                "n_fea_calls":      self.n_fea_calls,
                "score":            round(score, 6),
                "penalty":          _penalty_val,
                "moi_kg_mm2":       round(metrics["moi_kg_mm2"], 2),
                "bite_mm":          round(metrics["bite_mm"], 2),
                "n_teeth":          metrics["n_teeth"],
                "mean_tooth_height_mm": round(metrics["mean_tooth_height_mm"], 2),
                "structural":       round(metrics["structural_integrity"], 4),
                "fea_sf":           round(metrics["fea_safety_factor"], 3),
                "mass_util":        round(metrics["mass_utilization"], 4),
                "com_offset":       round(metrics["com_offset_mm"], 3),
                "mass_kg":          round(metrics["mass_kg"], 4),
                "energy_j":         round(metrics["energy_joules"], 1),
                "convergence":      round(convergence, 6),
                "shape_distance":   round(shape_dist, 4),
            }

            # ── Early-stop patience tracking ──────────────────────────────
            if score > self._best_score + self.min_delta:
                self._best_score = score
                self._best_x = xk.copy()
                self._steps_without_improvement = 0
            else:
                self._steps_without_improvement += 1

            # ── Trust-region: update module-level best ────────────────────
            if getattr(self.cfg.optimization, "trust_region_enabled", False):
                global _TR_BEST_X
                _TR_BEST_X = xk.copy()

            # ── Termination reason (early stop) ───────────────────────────
            if self._steps_without_improvement >= self.patience:
                entry["termination_reason"] = "early_stop"

            self.history.append(entry)

            if self.step % 10 == 0 or self.step <= 3:
                self._log(
                    f"  {self.phase_label} step {self.step:3d}: "
                    f"score={score:.4f} "
                    f"MOI={metrics['moi_kg_mm2']:.0f} "
                    f"bite={metrics['bite_mm']:.1f}mm({metrics['n_teeth']}t) "
                    f"SF={metrics['fea_safety_factor']:.2f} "
                    f"mass={metrics['mass_kg']:.3f}kg "
                    f"CoM={metrics['com_offset_mm']:.2f}mm "
                    f"[{elapsed:.0f}s]"
                )

            # ── FEA frame export (every step when enabled) ───────────────
            if self.fea_interval > 0:
                try:
                    # Reuse spiral contact data from metrics (already computed above)
                    _contacts   = metrics.get("_contacts", [])
                    _r_start    = metrics.get("_r_start", 0.0)

                    # Reuse the FEA mesh arrays cached by compute_metrics_enhanced
                    # (return_mesh=True was passed above) — no second FEA solve needed.
                    _nodes       = metrics.get("_fea_nodes")
                    _elements    = metrics.get("_fea_elements")
                    _vm_stresses = metrics.get("_fea_vm_stresses")

                    fea_data = {
                        "nodes":       _nodes,
                        "elements":    _elements,
                        "vm_stresses": _vm_stresses,
                        "peak_stress_mpa": metrics.get("fea_peak_stress_mpa", 0.0),
                        "safety_factor":   metrics.get("fea_safety_factor", 1.0),
                        "fea_score":       metrics.get("fea_score", 1.0),
                        "n_elements":      metrics.get("fea_n_elements", 0),
                    }
                    frame_path = self.frame_dir / f"frame_{self.frame_idx:04d}.png"
                    render_fea_frame(
                        weapon, fea_data, self.cfg,
                        step_label=f"{self.phase_label}-{self.step:03d}",
                        metrics=metrics,
                        save_path=frame_path,
                    )

                    # ── Spiral contact frame ──────────────────────────────
                    if self.spiral_frame_dir is not None and _contacts:
                        try:
                            render_spiral_contact_frame(
                                weapon, _contacts, _r_start, self.cfg,
                                step_label=f"{self.phase_label}-{self.step:03d}",
                                metrics=metrics,
                                save_path=(self.spiral_frame_dir
                                           / f"frame_{self.frame_idx:04d}.png"),
                            )
                        except Exception as e_sp:
                            self._log(f"  [spiral frame error at step {self.step}: {e_sp}]")

                    # ── Sidecar .npz (mesh + stress arrays) ──────────────
                    try:
                        # Encode holes as NaN-sentinel concatenated array
                        hole_arrays = []
                        for interior in weapon.interiors:
                            arr = np.array(list(interior.coords), dtype=float)
                            hole_arrays.append(arr)
                            hole_arrays.append(np.full((1, 2), np.nan))
                        holes_flat = (
                            np.vstack(hole_arrays)
                            if hole_arrays
                            else np.zeros((0, 2), dtype=float)
                        )
                        np.savez_compressed(
                            frame_path.with_suffix(".npz"),
                            nodes=fea_data["nodes"],
                            elements=fea_data["elements"],
                            vm_stresses=fea_data["vm_stresses"],
                            polygon_xy=np.array(list(weapon.exterior.coords), dtype=float),
                            holes_xy=holes_flat,
                        )
                    except Exception as e_npz:
                        self._log(f"  [sidecar .npz error at step {self.step}: {e_npz}]")

                    # ── Sidecar meta JSON ─────────────────────────────────
                    try:
                        cfg_snap = {
                            "yield_strength_mpa": self.cfg.material.yield_strength_mpa,
                            "density_kg_m3": self.cfg.material.density_kg_m3,
                            "sheet_thickness_mm": self.cfg.sheet_thickness_mm,
                            "rpm": self.cfg.rpm,
                            "bore_diameter_mm": self.cfg.mounting.bore_diameter_mm,
                            "profile_type": getattr(self.cfg.optimization, "profile_type", "bspline"),
                        }
                        meta = {
                            "step": self.step,
                            "phase": self.phase_label,
                            "score": round(score, 6),
                            "metrics": {
                                k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                for k, v in metrics.items()
                                if not k.startswith("_")  # skip internal/non-serialisable keys
                            },
                            "cfg_snapshot": cfg_snap,
                        }
                        frame_path.with_name(
                            frame_path.stem + "_meta.json"
                        ).write_text(json.dumps(meta, indent=2))
                    except Exception as e_json:
                        self._log(f"  [sidecar JSON error at step {self.step}: {e_json}]")

                    self.frame_idx += 1
                except Exception as e:
                    self._log(f"  [frame export error at step {self.step}: {e}]")

        except Exception as e:
            self._log(f"  {self.phase_label} step {self.step}: callback error: {e}")
            return None

        finally:
            cb_ms = (time.perf_counter() - _cb_t0) * 1000.0
            self._cb_times.append(cb_ms)
            if len(self._cb_times) % 10 == 0:
                avg_ms = sum(self._cb_times[-10:]) / 10.0
                self._log(f"  [{self.phase_label}] avg callback time (last 10): {avg_ms:.0f} ms")

        # Return True to signal scipy to halt this phase early
        if self._steps_without_improvement >= self.patience:
            self._log(
                f"  [{self.phase_label}] Early stop at step {self.step}: "
                f"no improvement > {self.min_delta:.4f} for {self.patience} steps "
                f"(best={self._best_score:.4f})"
            )
            return True
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_enhanced(
    cfg: WeaponConfig,
    case_dir: Path,
    verbose: bool = True,
) -> dict:
    """Run the enhanced two-phase optimisation with FEA-in-loop and GIF export.

    Parameters
    ----------
    cfg      : weapon configuration (evaluation_mode should be "enhanced")
    case_dir : output directory; frames and GIFs are written here
    verbose  : whether to print progress

    Returns a dict with the same keys as optimizer.optimize() plus:
        gif_phase1  : Path to Phase-1 GIF (or None)
        gif_phase2  : Path to Phase-2 GIF (or None)
        convergence_enhanced : list of per-step dicts from both phases
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    # Cap workers: actual population = popsize × n_params. Spawning more processes
    # than population individuals wastes time on IPC overhead. With the medium
    # preset (popsize=6, ~8 profile params → pop≈48) all 20 CPUs would idle most
    # of each generation. Use at most popsize workers as a proxy cap.
    _n_cpu  = max(1, os.cpu_count() or 1)
    workers = max(1, min(_n_cpu, cfg.optimization.population_size))

    # Profile bounds for Phase 1 — dispatched by profile_type config key
    profile_type   = getattr(cfg.optimization, "profile_type", "bspline")
    bspline_bounds = get_profile_bounds(profile_type, cfg)
    n_bspline      = len(bspline_bounds)

    # Reduced iteration budget: FEA per call costs more, so we use fewer
    # iterations but gain far better per-generation signal quality.
    base_iters   = cfg.optimization.max_iterations
    _p1_override = getattr(cfg.optimization, "phase1_iters", 0)
    _p2_override = getattr(cfg.optimization, "phase2_iters", 0)
    phase1_iters = _p1_override if _p1_override > 0 else max(10, int(base_iters * 0.50))
    phase2_iters = _p2_override if _p2_override > 0 else max(5,  int(base_iters * 0.25))
    pop_size     = cfg.optimization.population_size
    fea_interval = cfg.optimization.fea_interval
    patience     = getattr(cfg.optimization, "convergence_patience", 15)
    min_delta    = getattr(cfg.optimization, "convergence_min_delta", 0.002)

    logs: list[str] = []

    def _log(msg: str) -> None:
        if verbose:
            print(msg)
        logs.append(msg)

    polar_bounds_preview = get_cutout_bounds_polar(cfg)
    _log(f"[enhanced] profile_type={profile_type}, profile params={n_bspline}, "
         f"polar cutout params={len(polar_bounds_preview)}, "
         f"workers={workers}")
    _log(f"[enhanced] phase1_iters={phase1_iters}, phase2_iters={phase2_iters}, "
         f"fea_interval={fea_interval}")

    # ── GIF frame mode: force single-process so FEA mesh cache is visible ──
    # compute_metrics_enhanced(return_mesh=True) stores mesh arrays in the
    # metrics dict.  With workers>1 the objective runs in subprocesses; the
    # cached arrays would never reach the main-process callback.
    # Forcing workers=1 keeps the full call stack in-process, eliminating
    # the second FEA solve that the callback previously ran for visualization.
    if fea_interval > 0:
        workers = 1
        _log("[gif] fea_interval > 0 — forcing workers=1 to enable FEA mesh caching")

    # ── Trust-region: force single-process and reset best-x tracker ──────
    if getattr(cfg.optimization, "trust_region_enabled", False):
        global _TR_BEST_X
        _TR_BEST_X = None
        workers = 1
        _log("[trust_region] enabled — forcing workers=1")

    # ── Staged evaluation gate ────────────────────────────────────────────
    global _EVAL_GATE
    gate_frac = getattr(cfg.optimization, "staged_eval_gate", 1.0)
    _EVAL_GATE = EvalGate(cfg, gate_frac=gate_frac)
    if gate_frac < 1.0:
        _log(f"[enhanced] staged eval gate enabled: gate_frac={gate_frac:.2f} "
             f"(top {gate_frac*100:.0f}% of Stage-1 scores proceed to FEA)")

    # ── Structured population seeding ─────────────────────────────────────
    structured_frac = getattr(cfg.optimization, "structured_seed_frac", 0.0)
    p1_init: str | np.ndarray = "latinhypercube"
    if structured_frac > 0.0:
        try:
            from .seeding import mixed_init_population
            total_pop = pop_size * len(bspline_bounds)
            p1_init = mixed_init_population(
                cfg, total_pop, bspline_bounds,
                structured_frac=structured_frac,
            )
            _log(f"[seeding] structured_seed_frac={structured_frac:.2f}: "
                 f"{int(total_pop * structured_frac)} seeds from functional/archetype bank "
                 f"+ {total_pop - int(total_pop * structured_frac)} LHS fill "
                 f"(total {total_pop})")
        except Exception as exc:
            _log(f"[seeding] WARNING: structured seeding failed ({exc}), falling back to LHS")
            p1_init = "latinhypercube"

    # ── Phase 1: outer profile ────────────────────────────────────────────
    _log(f"=== Enhanced Phase 1: {profile_type.upper()} Profile ===")
    p1_frame_dir   = case_dir / "frames_p1"
    p1_spiral_dir  = case_dir / "frames_p1_spiral"
    p1_frame_dir.mkdir(parents=True, exist_ok=True)
    if fea_interval > 0:
        p1_spiral_dir.mkdir(parents=True, exist_ok=True)

    p1_cb = _FEACallback(
        cfg=cfg,
        profile_params=None,
        frame_dir=p1_frame_dir,
        phase_label="P1",
        log_fn=_log,
        fea_interval=fea_interval,
        bspline=True,
        patience=patience,
        min_delta=min_delta,
        spiral_frame_dir=p1_spiral_dir if fea_interval > 0 else None,
    )

    result1 = differential_evolution(
        _profile_objective_bspline,
        bounds=bspline_bounds,
        args=(cfg,),
        maxiter=phase1_iters,
        popsize=pop_size,
        seed=None,   # None → OS entropy → genuine replicate variance
        tol=1e-3,
        mutation=(cfg.optimization.de_mutation_lo, cfg.optimization.de_mutation_hi),
        recombination=cfg.optimization.de_recombination,
        workers=workers,
        updating="deferred",
        disp=False,
        init=p1_init,
        callback=p1_cb,
    )

    best_profile = result1.x
    _log(f"Phase 1 done: best_score={-result1.fun:.4f}, evals={result1.nfev}")

    # ── Single-phase profiles (e.g. spiral_weapon): skip Phase 2 entirely ──
    # These profiles return a complete polygon (outer + cutouts) from Phase 1.
    _single_phase = is_single_phase_profile(profile_type)
    if _single_phase:
        _log(f"[enhanced] profile_type='{profile_type}' is single-phase — skipping Phase 2")
        from .spiral_weapon import build_spiral_weapon as _build_sw
        _sp_weapon = _build_sw(best_profile, cfg)
        if _sp_weapon is None:
            _log("WARNING: spiral_weapon build returned None for best_profile — using fallback")
            from shapely.geometry import Point as _Pt
            _sp_weapon = _Pt(0, 0).buffer(float(cfg.envelope.max_radius_mm))
        weapon_norm       = _sp_weapon
        outer_ref         = _sp_weapon
        cutout_polys_norm = []
        p2_history: list[dict] = []
        gif_p2: Path | None    = None
    else:
        # ── Pre-Phase-2: compute reference area for mass normalisation ─────
        # Build weapon with mounting holes only (no weight-reduction cutouts).
        outer_ref, weapon_no_cutouts_poly = _weapon_no_cutouts_bspline(best_profile, cfg)
        no_cutout_area = weapon_no_cutouts_poly.area
        _log(f"[mass norm] reference area (no WR cutouts) = {no_cutout_area:.1f} mm²  "
             f"→ solid at budget = "
             f"{cfg.weight_budget_kg / (cfg.material.density_kg_m3 * 1e-9 * cfg.sheet_thickness_mm):.1f} mm²")

        # ── Phase 2: polar CAD cutouts OR topology optimisation ───────────
        cutout_type   = getattr(cfg.optimization, "cutout_type", "topology")
        if cutout_type != "topology":
            import warnings
            warnings.warn(
                f"cutout_type='{cutout_type}' is deprecated in enhanced mode; "
                "prefer cutout_type='topology'. Parametric DE will still run.",
                DeprecationWarning,
                stacklevel=2,
            )
            _log(
                f"[enhanced] WARNING: cutout_type='{cutout_type}' is deprecated; "
                "topology is now the default for enhanced Phase 2"
            )
        p2_frame_dir  = case_dir / "frames_p2"
        p2_spiral_dir = case_dir / "frames_p2_spiral"
        p2_frame_dir.mkdir(parents=True, exist_ok=True)
        if fea_interval > 0:
            p2_spiral_dir.mkdir(parents=True, exist_ok=True)
        p2_history: list[dict] = []
        gif_p2: Path | None = None

        if cutout_type == "topology":
            # ── SIMP topology optimisation ────────────────────────────────
            _log("=== Enhanced Phase 2: SIMP Topology Optimisation ===")
            topo_result = _topo_phase2(
                solid_polygon=weapon_no_cutouts_poly,
                cfg=cfg,
                case_dir=case_dir,
                fea_interval=fea_interval,
                log_fn=_log,
            )
            weapon_norm       = topo_result["weapon_polygon"]
            cutout_polys_norm = topo_result["cutout_polygons"]
            p2_history        = [
                {
                    "phase":      "P2-topo",
                    "step":       h["iteration"],
                    "elapsed_s":  h["elapsed_s"],
                    "score":      0.0,           # not a score-based optimiser
                    "compliance": h["compliance"],
                    "moi_kg_mm2": h["moi_kg_mm2"],
                    "mass_kg":    h["mass_kg"],
                    "v_current":  h["v_current"],
                    "v_target":   h["v_target"],
                }
                for h in topo_result["history"]
            ]
            # GIFs from topo (expose the density-evolution one as the Phase-2 GIF)
            gif_p2 = topo_result.get("gif_topo")

        else:
            # ── Polar parametric cutouts (original Phase 2) ───────────────
            polar_bounds = get_cutout_bounds_polar(cfg)
            C = cfg.optimization.num_cutout_pairs

            if C > 0 and len(polar_bounds) > 0:
                _log("=== Enhanced Phase 2: Polar CAD Cutouts on B-Spline Profile ===")

                p2_init: str | np.ndarray = "latinhypercube"
                if structured_frac > 0.0:
                    try:
                        from .seeding import mixed_init_population
                        total_pop2 = pop_size * len(polar_bounds)
                        p2_init = mixed_init_population(
                            cfg, total_pop2, polar_bounds,
                            structured_frac=structured_frac,
                        )
                        _log(f"[seeding] Phase 2: {int(total_pop2 * structured_frac)} structured + "
                             f"{total_pop2 - int(total_pop2 * structured_frac)} LHS fill")
                    except Exception as exc:
                        _log(f"[seeding] WARNING: Phase-2 seeding failed ({exc}), falling back to LHS")
                        p2_init = "latinhypercube"

                p2_cb = _FEACallback(
                    cfg=cfg,
                    profile_params=best_profile,
                    frame_dir=p2_frame_dir,
                    phase_label="P2",
                    log_fn=_log,
                    fea_interval=fea_interval,
                    polar=True,
                    bspline=True,
                    patience=patience,
                    min_delta=min_delta,
                    spiral_frame_dir=p2_spiral_dir if fea_interval > 0 else None,
                )

                result2 = differential_evolution(
                    _cutout_objective_bspline_polar,
                    bounds=polar_bounds,
                    args=(best_profile, cfg),
                    maxiter=phase2_iters,
                    popsize=pop_size,
                    seed=None,   # None → OS entropy → genuine replicate variance
                    tol=1e-3,
                    mutation=(cfg.optimization.de_mutation_lo, cfg.optimization.de_mutation_hi),
                    recombination=cfg.optimization.de_recombination,
                    workers=workers,
                    updating="deferred",
                    disp=False,
                    init=p2_init,
                    callback=p2_cb,
                )

                best_cutouts = result2.x
                p2_history   = p2_cb.history
                _log(f"Phase 2 done: best_score={-result2.fun:.4f}, evals={result2.nfev}")
            else:
                best_cutouts = np.zeros(max(C, 1) * CUTOUT_STRIDE_POLAR)

            # ── Analytical mass normalisation ─────────────────────────────
            C = cfg.optimization.num_cutout_pairs
            if C > 0 and best_cutouts.size > 0:
                _log("Applying analytical mass normalisation...")
                _, weapon_pre = _weapon_no_cutouts_bspline(best_profile, cfg)
                no_cutout_area_fresh = weapon_pre.area

                mass_before = None
                for iteration in range(3):
                    _, w_cur, _ = _build_weapon_bspline_polar(best_profile, best_cutouts, cfg)
                    best_cutouts, s = mass_normalize_cutouts(
                        best_cutouts, C,
                        w_cur.area, no_cutout_area_fresh, cfg,
                    )
                    mass_now_kg = (cfg.material.density_kg_m3 * 1e-9
                                   * cfg.sheet_thickness_mm * w_cur.area)
                    if mass_before is None:
                        mass_before = mass_now_kg
                    if abs(s - 1.0) < 0.005:
                        _log(f"  Converged after {iteration + 1} iteration(s): s={s:.4f}")
                        break
                    _log(f"  Iteration {iteration + 1}: s={s:.4f}  "
                         f"mass={mass_now_kg:.4f} kg → target {cfg.weight_budget_kg:.4f} kg")

                _, weapon_norm, cutout_polys_norm = _build_weapon_bspline_polar(
                    best_profile, best_cutouts, cfg)
                mass_after_kg = (cfg.material.density_kg_m3 * 1e-9
                                 * cfg.sheet_thickness_mm * weapon_norm.area)
                _log(f"  Mass: {mass_before:.4f} → {mass_after_kg:.4f} kg "
                     f"(budget {cfg.weight_budget_kg:.4f} kg, "
                     f"error {abs(mass_after_kg - cfg.weight_budget_kg) / cfg.weight_budget_kg * 100:.2f}%)")
            else:
                weapon_norm       = weapon_no_cutouts_poly
                cutout_polys_norm = []

    # ── Final evaluation (fine-mesh FEA on normalised weapon) ────────────
    _log("Running final fine-mesh FEA evaluation...")
    outer = outer_ref
    weapon = weapon_norm
    cutout_polys = cutout_polys_norm
    metrics = compute_metrics_enhanced(
        weapon, cfg,
        fea_spacing=cfg.optimization.fea_fine_spacing_mm,
    )
    score   = _score_from_metrics(metrics, weapon, cfg)
    penalty = constraint_penalty(weapon, cfg, cutout_polys=cutout_polys)

    _log(f"[enhanced] final score={score:.4f}, penalty={penalty:.4f}")
    _log(f"  MOI={metrics['moi_kg_mm2']:.1f} kg·mm²  "
         f"Energy={metrics['energy_joules']:.0f} J  "
         f"Bite={metrics['bite_mm']:.1f} mm  "
         f"Teeth={metrics['n_teeth']}  "
         f"Height={metrics['mean_tooth_height_mm']:.1f} mm")
    _log(f"  Mass={metrics['mass_kg']:.3f}/{cfg.weight_budget_kg:.1f} kg  "
         f"CoM={metrics['com_offset_mm']:.2f} mm  "
         f"SF={metrics['fea_safety_factor']:.2f}")

    # Save a final high-quality FEA frame (fine mesh + contact forces)
    _final_contacts  = metrics.get("_contacts", [])
    _final_r_start   = metrics.get("_r_start", 0.0)
    _final_fea_forces = metrics.get("_fea_forces", [])

    _clm_final = getattr(cfg.optimization, "contact_load_mode", "neumann_edge")
    final_fea = fea_stress_analysis_with_mesh(
        weapon,
        rpm=cfg.rpm,
        density_kg_m3=cfg.material.density_kg_m3,
        thickness_mm=cfg.sheet_thickness_mm,
        yield_strength_mpa=cfg.material.yield_strength_mpa,
        bore_diameter_mm=cfg.mounting.bore_diameter_mm,
        mesh_spacing=cfg.optimization.fea_fine_spacing_mm,
        contact_forces=_final_fea_forces,
        contact_load_mode=_clm_final,
    )
    render_fea_frame(
        weapon, final_fea, cfg,
        step_label="FINAL",
        metrics=metrics,
        save_path=case_dir / "fea_final.png",
        dpi=120,
    )

    # Final spiral contact diagram (standalone PNG)
    spiral_final_png: Path | None = None
    if _final_contacts:
        try:
            spiral_final_png = render_spiral_contact_frame(
                weapon, _final_contacts, _final_r_start, cfg,
                step_label="FINAL",
                metrics=metrics,
                save_path=case_dir / "spiral_contact_final.png",
                dpi=120,
            )
            if spiral_final_png:
                _log(f"Spiral contact diagram saved: {spiral_final_png}")
        except Exception as e_sc:
            _log(f"  [spiral final frame error: {e_sc}]")

    # ── Assemble GIFs ─────────────────────────────────────────────────────
    # FEA convergence GIFs — every frame that was saved to the frame directories
    gif_p1 = export_gif(p1_frame_dir, case_dir / "convergence_phase1.gif", fps=4)
    # gif_p2 already set for topology mode; build from frames_p2 for polar mode
    if gif_p2 is None:
        gif_p2 = export_gif(p2_frame_dir, case_dir / "convergence_phase2.gif", fps=4)

    # Spiral contact evolution GIFs (one per phase)
    gif_spiral_p1: Path | None = None
    gif_spiral_p2: Path | None = None
    if fea_interval > 0:
        gif_spiral_p1 = export_gif(
            p1_spiral_dir, case_dir / "spiral_contact_phase1.gif", fps=4
        )
        gif_spiral_p2 = export_gif(
            p2_spiral_dir, case_dir / "spiral_contact_phase2.gif", fps=4
        )

    for _g in (gif_p1, gif_p2, gif_spiral_p1, gif_spiral_p2):
        if _g:
            _log(f"GIF saved: {_g}")

    # ── DXF export (always) ───────────────────────────────────────────────
    from .exporter import export_weapon_dxf
    _dxf_stem = case_dir.name  # use directory name as stem (matches evaluate.py naming)
    dxf_path = export_weapon_dxf(weapon, cfg, case_dir, stem=_dxf_stem)
    if dxf_path:
        _log(f"DXF exported: {dxf_path}")

    return {
        "weapon_polygon":       weapon,
        "outer_profile":        outer,
        "cutout_polys":         cutout_polys,
        "metrics":              metrics,
        "score":                score,
        "penalty":              penalty,
        "convergence_p1":       p1_cb.history,
        "convergence_p2":       p2_history,
        "gif_phase1":           gif_p1,
        "gif_phase2":           gif_p2,
        "gif_spiral_phase1":    gif_spiral_p1,
        "gif_spiral_phase2":    gif_spiral_p2,
        "spiral_contact_png":   spiral_final_png,
        "dxf_path":             dxf_path,
        "logs":                 logs,
        "result_phase1":        result1,
    }
