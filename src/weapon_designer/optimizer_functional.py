"""Low-dimensional functional optimizer.

Stage A: 6-dim Differential Evolution on the analytic lobed-profile parameter
space defined by ``functional_profiles.py``.  With only 6 degrees of freedom the
DE landscape is far smoother and converges with a tiny population (20 × 6 = 120
individuals vs. 60 × 12 = 720 for B-spline).

Stage B: N-dim lift — warm-starts a full B-spline DE from the Stage-A solution
(converted via ``lobed_radii``), with a proximity penalty that decays via
score-plateau annealing.  The penalty coefficient λ starts large (strong
regularisation toward the functional shape) and halves each time the best score
stagnates for ``patience`` steps, eventually allowing the optimizer to deviate
freely from the analytic seed.

Public entry point: ``optimize_functional(cfg, case_dir, stage_b=True)``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from .config import WeaponConfig
from .functional_profiles import (
    get_functional_bounds,
    build_functional_profile,
    lobed_radii,
)
from .bspline_profile import get_bspline_bounds, build_bspline_profile
from .geometry import assemble_weapon
from .constraints import validate_geometry, constraint_penalty, check_envelope
from .objectives_enhanced import compute_metrics_enhanced, weighted_score_enhanced


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_limits(cfg: WeaponConfig) -> tuple[float, float]:
    """Return (min_r, max_r) for the outer profile."""
    if cfg.weapon_style == "bar":
        max_r = float(np.hypot(cfg.envelope.max_length_mm / 2.0,
                               cfg.envelope.max_width_mm / 2.0))
    else:
        max_r = float(cfg.envelope.max_radius_mm)

    bore_r = cfg.mounting.bore_diameter_mm / 2.0
    min_r = max(bore_r + cfg.optimization.min_wall_thickness_mm + 5.0, 15.0)
    max_r = max(max_r, min_r + 10.0)
    return min_r, max_r


def _score_poly(poly, cfg: WeaponConfig) -> float:
    """Build weapon, compute metrics, return weighted score (or 0 on failure)."""
    try:
        weapon = assemble_weapon(poly, cfg.mounting, [])
        weapon = validate_geometry(weapon)
        if weapon.is_empty:
            return 0.0
        metrics = compute_metrics_enhanced(weapon, cfg)
        score = weighted_score_enhanced(metrics, cfg)
        if not check_envelope(weapon, cfg):
            score *= 0.3
        return float(score)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Stage A — 6-dim lobed profile DE
# ---------------------------------------------------------------------------

def optimize_functional_stage_a(
    cfg: WeaponConfig,
    case_dir: Path,
    log_fn=print,
) -> dict:
    """Run Stage-A: 6-parameter lobed-profile differential evolution.

    Returns
    -------
    dict with keys:
        best_params  : (6,) array [n_teeth, R0, A1, A2, A3, phi]
        best_score   : float in [0, 1]
        history      : list of per-step dicts
        elapsed_s    : wall-clock seconds
    """
    bounds = get_functional_bounds(cfg)
    min_r, max_r = _get_limits(cfg)
    N_ctrl = getattr(cfg.optimization, "n_bspline_points", 12)

    _p1_override = getattr(cfg.optimization, "phase1_iters", 0)
    n_iter = _p1_override if _p1_override > 0 else max(10, int(cfg.optimization.max_iterations * 0.50))

    history: list[dict] = []
    start_time = time.time()
    step_counter = [0]

    def obj(params: np.ndarray) -> float:
        try:
            poly = build_functional_profile(params, max_r, min_r, N_ctrl)
            if poly is None or poly.is_empty or poly.area < 1.0:
                return 1.0
            return -_score_poly(poly, cfg)
        except Exception:
            return 1.0

    def callback(xk: np.ndarray, convergence: float) -> None:
        step_counter[0] += 1
        elapsed = time.time() - start_time
        score = max(0.0, -obj(xk))
        history.append({
            "step":        step_counter[0],
            "elapsed_s":   round(elapsed, 1),
            "score":       round(score, 6),
            "convergence": round(convergence, 6),
        })
        if step_counter[0] % 5 == 0 or step_counter[0] <= 2:
            log_fn(f"  [FuncA] step {step_counter[0]:3d}: score={score:.4f} [{elapsed:.0f}s]")

    result = differential_evolution(
        obj,
        bounds=bounds,
        maxiter=n_iter,
        popsize=20,
        seed=42,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.8,
        workers=1,
        updating="deferred",
        disp=False,
        callback=callback,
    )

    return {
        "best_params": result.x,
        "best_score":  float(-result.fun),
        "history":     history,
        "elapsed_s":   time.time() - start_time,
    }


# ---------------------------------------------------------------------------
# Stage B — N-dim lift with score-plateau proximity annealing
# ---------------------------------------------------------------------------

def optimize_functional_stage_b(
    cfg: WeaponConfig,
    stage_a_params: np.ndarray,
    case_dir: Path,
    log_fn=print,
) -> dict:
    """Run Stage-B: N-dim B-spline lift with score-plateau proximity annealing.

    Warm-starts the DE population near the Stage-A solution (converted via
    ``lobed_radii``).  The objective includes a regularisation term::

        obj(r) = -(score(r) - λ · mean((r − r_ref)²))

    λ starts at ``lambda_0 = 1.0`` and is halved whenever the best score has
    not improved by > 0.001 for ``PATIENCE`` consecutive callback steps.
    This progressively relaxes the proximity constraint, allowing the optimizer
    to explore deviations from the analytic seed.

    Returns
    -------
    dict with keys:
        best_radii   : (N,) float array of B-spline radii
        best_score   : float — score without the proximity penalty
        history      : list of per-step dicts
        r_ref        : (N,) reference radii from Stage A
        elapsed_s    : wall-clock seconds
    """
    N = getattr(cfg.optimization, "n_bspline_points", 12)
    min_r, max_r = _get_limits(cfg)

    # Reference radii from Stage-A params
    n_t = max(1, min(8, round(float(stage_a_params[0]))))
    R0, A1, A2, A3, phi = (float(stage_a_params[i]) for i in range(1, 6))
    r_ref = lobed_radii(n_t, R0, A1, A2, A3, phi, N, min_r=min_r, max_r=max_r)

    # Bounds for Stage-B (same as bspline mode)
    bspline_bounds = get_bspline_bounds(cfg)
    if not bspline_bounds:
        bspline_bounds = [(min_r, max_r)] * N

    _p2_override = getattr(cfg.optimization, "phase2_iters", 0)
    n_iter = _p2_override if _p2_override > 0 else max(5, int(cfg.optimization.max_iterations * 0.25))

    PATIENCE = max(3, getattr(cfg.optimization, "convergence_patience", 15) // 2)
    popsize = 10
    sigma = (max_r - min_r) * 0.05

    # Warm-start: population near r_ref with Gaussian noise
    rng = np.random.default_rng(42)
    pop_total = popsize * N
    init_pop = np.clip(
        r_ref[None, :] + rng.normal(0.0, sigma, size=(pop_total, N)),
        min_r, max_r,
    )

    # Mutable state for the proximity annealing schedule
    lambda_holder = [1.0]
    best_score_tracker = [-float("inf")]
    steps_without_improvement = [0]

    history: list[dict] = []
    start_time = time.time()
    step_counter = [0]

    def obj(r: np.ndarray) -> float:
        try:
            r_clipped = np.clip(r, min_r, max_r)
            poly = build_bspline_profile(r_clipped, max_r, min_r)
            if poly is None or poly.is_empty or poly.area < 1.0:
                return 1.0
            score = _score_poly(poly, cfg)
            prox_penalty = lambda_holder[0] * float(np.mean((r_clipped - r_ref) ** 2))
            return -(score - prox_penalty)
        except Exception:
            return 1.0

    def callback(xk: np.ndarray, convergence: float) -> None:
        step_counter[0] += 1
        elapsed = time.time() - start_time

        # Score without proximity penalty (for tracking progress)
        try:
            r_clipped = np.clip(xk, min_r, max_r)
            poly = build_bspline_profile(r_clipped, max_r, min_r)
            score = _score_poly(poly, cfg) if (poly is not None and not poly.is_empty) else 0.0
        except Exception:
            score = 0.0

        history.append({
            "step":        step_counter[0],
            "elapsed_s":   round(elapsed, 1),
            "score":       round(score, 6),
            "lambda":      round(lambda_holder[0], 6),
            "convergence": round(convergence, 6),
        })

        # Score-plateau annealing: halve lambda when stagnated
        if score > best_score_tracker[0] + 0.001:
            best_score_tracker[0] = score
            steps_without_improvement[0] = 0
        else:
            steps_without_improvement[0] += 1

        if steps_without_improvement[0] >= PATIENCE:
            lambda_holder[0] = max(lambda_holder[0] * 0.5, 1e-4)
            steps_without_improvement[0] = 0
            log_fn(f"  [FuncB] step {step_counter[0]}: plateau → λ halved to {lambda_holder[0]:.4f}")

        if step_counter[0] % 5 == 0 or step_counter[0] <= 2:
            log_fn(f"  [FuncB] step {step_counter[0]:3d}: score={score:.4f} "
                   f"λ={lambda_holder[0]:.4f} [{elapsed:.0f}s]")

    result = differential_evolution(
        obj,
        bounds=bspline_bounds,
        maxiter=n_iter,
        popsize=popsize,
        seed=42,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.8,
        workers=1,
        updating="deferred",
        disp=False,
        init=init_pop,
        callback=callback,
    )

    best_radii = np.clip(result.x, min_r, max_r)

    # Final score without penalty
    try:
        poly_final = build_bspline_profile(best_radii, max_r, min_r)
        best_score = _score_poly(poly_final, cfg) if poly_final is not None else 0.0
    except Exception:
        best_score = 0.0

    return {
        "best_radii": best_radii,
        "best_score": best_score,
        "history":    history,
        "r_ref":      r_ref,
        "elapsed_s":  time.time() - start_time,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def optimize_functional(
    cfg: WeaponConfig,
    case_dir: Path,
    stage_b: bool = True,
    log_fn=print,
) -> dict:
    """Run the functional optimizer: Stage A (6-dim) + optional Stage B (N-dim lift).

    Parameters
    ----------
    cfg       : weapon configuration
    case_dir  : output directory (created if missing)
    stage_b   : if True, run the N-dim proximity-annealing lift after Stage A
    log_fn    : callable for progress messages

    Returns
    -------
    dict with keys:
        weapon_polygon       : final Shapely Polygon
        metrics              : dict from compute_metrics_enhanced
        score                : float weighted score
        penalty              : float constraint penalty
        functional_params    : (6,) Stage-A best params
        convergence_stage_a  : list of per-step dicts from Stage A
        convergence_stage_b  : list of per-step dicts from Stage B (may be [])
        elapsed_s            : total wall-clock seconds
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    N = getattr(cfg.optimization, "n_bspline_points", 12)
    min_r, max_r = _get_limits(cfg)

    log_fn("[functional] Stage A: 6-dim lobed-profile DE")
    result_a = optimize_functional_stage_a(cfg, case_dir, log_fn)
    log_fn(f"[functional] Stage A done: score={result_a['best_score']:.4f}  "
           f"params={result_a['best_params'].round(3)}")

    result_b: dict = {}
    if stage_b:
        log_fn("[functional] Stage B: N-dim B-spline lift with proximity annealing")
        result_b = optimize_functional_stage_b(cfg, result_a["best_params"], case_dir, log_fn)
        log_fn(f"[functional] Stage B done: score={result_b['best_score']:.4f}")

    # Build final weapon polygon
    if stage_b and result_b:
        final_radii = result_b["best_radii"]
        final_poly = build_bspline_profile(final_radii, max_r, min_r)
    else:
        final_poly = build_functional_profile(result_a["best_params"], max_r, min_r, N)

    if final_poly is None or final_poly.is_empty:
        from shapely.geometry import Point
        final_poly = Point(0.0, 0.0).buffer(max_r * 0.5)

    weapon = assemble_weapon(final_poly, cfg.mounting, [])
    weapon = validate_geometry(weapon)

    try:
        metrics = compute_metrics_enhanced(weapon, cfg)
        score = weighted_score_enhanced(metrics, cfg)
        pen = constraint_penalty(weapon, cfg)
    except Exception:
        metrics = {}
        score = 0.0
        pen = 0.0

    return {
        "weapon_polygon":      weapon,
        "metrics":             metrics,
        "score":               score,
        "penalty":             pen,
        "functional_params":   result_a["best_params"],
        "convergence_stage_a": result_a["history"],
        "convergence_stage_b": result_b.get("history", []),
        "elapsed_s":           time.time() - start_time,
    }
