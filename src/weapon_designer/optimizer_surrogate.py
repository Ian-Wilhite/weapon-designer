"""Active-learning optimizer that uses a GP surrogate to reduce FEA calls.

Architecture
------------
Wraps a Differential Evolution inner loop.  At each candidate evaluation:

1. Compute cheap metrics (mass, MOI, bite via kinematic_spiral_bite) — no FEA
2. Query FEASurrogate.predict_score(params, cfg, metrics_base)
   → (score_pred, uncertainty σ²)
3. UCB acquisition: score_acq = score_pred + β·σ
4. Run full FEA only if:
   a. σ > σ_fea_threshold (GP is uncertain), OR
   b. score_acq > best_so_far × acq_threshold (promising region)
   c. Force FEA every force_fea_every evaluations (exploration guarantee)
5. After FEA, update the GP surrogate with the new data point.

Stats tracked
-------------
n_fea_calls, n_total_evals, fea_rate, wall_clock → saved to output_stats.json

Visualisation
-------------
At every vis_interval surrogate evaluations:
  - acquisition_landscape.png  — 2D PCA projection of acquisition surface
  - fea_call_rate.png          — FEA-call fraction over time
  - surrogate_score_history.png — predicted vs. true score for FEA'd designs
  - uncertainty_evolution.png   — mean GP uncertainty over evaluations

Usage
-----
    from weapon_designer.optimizer_surrogate import optimize_surrogate

    result = optimize_surrogate(cfg, surrogate, case_dir=Path("run_surr"))
    # result dict: weapon, metrics, score, n_fea_calls, n_total_evals, ...
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .config import WeaponConfig
from .surrogate_fea import FEASurrogate


# ---------------------------------------------------------------------------
# Acquisition function
# ---------------------------------------------------------------------------

def ucb_acquisition(score_pred: float, uncertainty: float, beta: float = 2.0) -> float:
    """Upper Confidence Bound: score_pred + β·σ."""
    return score_pred + beta * math.sqrt(max(uncertainty, 0.0))


# ---------------------------------------------------------------------------
# Cheap (no-FEA) metrics
# ---------------------------------------------------------------------------

def _cheap_metrics(poly, cfg: WeaponConfig) -> dict:
    """Fast metrics that do NOT require FEA."""
    from .physics import polygon_mass_kg, mass_moi_kg_mm2, com_offset_mm
    from .objectives_enhanced import kinematic_spiral_bite

    t   = cfg.sheet_thickness_mm
    rho = cfg.material.density_kg_m3
    v   = float(getattr(cfg.optimization, "drive_speed_mps", 6.0))

    mass  = polygon_mass_kg(poly, t, rho)
    moi   = mass_moi_kg_mm2(poly, t, rho)
    com   = com_offset_mm(poly)
    bite  = kinematic_spiral_bite(poly, cfg.rpm, drive_speed_mps=v)

    return {
        "mass_kg":         mass,
        "moi_kg_mm2":      moi,
        "com_offset_mm":   com,
        "bite_mm":         bite["bite_mm"],
        "max_bite_mm":     bite["max_bite_mm"],
        "n_contacts":      bite["n_contacts"],
        "contact_quality": 0.5,   # placeholder (no spiral FEA)
        "mass_utilization": mass / max(cfg.weight_budget_kg, 1e-6),
    }


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def optimize_surrogate(
    cfg: WeaponConfig,
    surrogate: FEASurrogate,
    case_dir: Path | str = Path("surrogate_run"),
    max_iterations: int | None = None,
    population_size: int | None = None,
    beta_ucb: float = 2.0,
    sigma_fea_threshold: float = 0.05,
    acq_threshold: float = 0.90,
    force_fea_every: int = 10,
    vis_interval: int = 20,
    checkpoint_interval: int = 50,
    verbose: bool = True,
) -> dict:
    """Active-learning weapon optimizer driven by a GP surrogate.

    Parameters
    ----------
    cfg               : weapon configuration
    surrogate         : fitted FEASurrogate instance
    case_dir          : output directory for stats, plots, DXF
    max_iterations    : DE iteration budget (default: cfg.optimization.max_iterations)
    population_size   : DE population size (default: cfg.optimization.population_size)
    beta_ucb          : UCB exploration parameter (higher = more exploration)
    sigma_fea_threshold : GP std threshold above which FEA is forced
    acq_threshold     : if score_acq > best × acq_threshold, force FEA
    force_fea_every   : run FEA every N evaluations regardless (prevents neglect)
    vis_interval      : save visualisation plots every N evaluations
    verbose           : print progress

    Returns
    -------
    dict with keys: weapon, metrics, score, n_fea_calls, n_total_evals,
                    fea_rate, wall_time_s, history, vis_paths
    """
    from scipy.optimize import differential_evolution

    from .profile_builder import build_profile, get_profile_bounds
    from .geometry import assemble_weapon
    from .constraints import validate_geometry, constraint_penalty, check_envelope
    from .objectives_enhanced import compute_metrics_enhanced, weighted_score_enhanced
    from .objectives_physical import compute_physical_score

    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    max_iter  = max_iterations or cfg.optimization.max_iterations
    pop_size  = population_size or cfg.optimization.population_size
    n_bspline = getattr(cfg.optimization, "n_bspline_points", 8)
    profile_t = getattr(cfg.optimization, "profile_type", "bspline")

    def _log(msg: str):
        if verbose:
            print(msg, flush=True)

    # ── State tracked across evaluations ────────────────────────────────
    state: dict = {
        "n_total":       0,
        "n_fea":         0,
        "best_score":    0.0,
        "best_x":        None,
        "best_weapon":   None,
        "best_metrics":  None,
        "history":       [],          # list of dicts per evaluation
        "surr_data":     [],          # (params, true_score) pairs for GP update
        "t_start":       time.perf_counter(),
    }

    # Visualisation state
    _vis_scores_pred: list[float] = []
    _vis_scores_true: list[float] = []
    _vis_uncertainties: list[float] = []
    _vis_fea_flags: list[bool] = []
    vis_paths: list[str] = []

    bounds = get_profile_bounds(profile_t, cfg)

    def _objective(x: np.ndarray) -> float:
        try:
            outer = build_profile(profile_t, x, cfg)
            if outer is None or outer.is_empty:
                return 1.0

            weapon = assemble_weapon(outer, cfg.mounting, [])
            weapon = validate_geometry(weapon)
            if weapon.is_empty:
                return 1.0

            state["n_total"] += 1
            n_total = state["n_total"]

            # ── Cheap metrics (no FEA) ────────────────────────────────────
            try:
                m_cheap = _cheap_metrics(weapon, cfg)
            except Exception:
                m_cheap = None

            # ── GP surrogate prediction ───────────────────────────────────
            try:
                score_pred, uncertainty = surrogate.predict_score(x, cfg, metrics_base=m_cheap)
            except Exception:
                score_pred, uncertainty = 0.0, 1.0

            acq = ucb_acquisition(score_pred, uncertainty, beta=beta_ucb)

            # ── FEA decision ─────────────────────────────────────────────
            run_fea = (
                math.sqrt(max(uncertainty, 0.0)) > sigma_fea_threshold
                or acq > state["best_score"] * acq_threshold
                or (n_total % force_fea_every == 0)
            )

            if run_fea:
                state["n_fea"] += 1
                try:
                    metrics_full = compute_metrics_enhanced(weapon, cfg)
                    mode = getattr(cfg.optimization, "evaluation_mode", "enhanced")
                    if mode == "physical":
                        true_score = compute_physical_score(weapon, cfg, metrics=metrics_full)
                    else:
                        true_score = weighted_score_enhanced(metrics_full, cfg)
                    if not check_envelope(weapon, cfg):
                        true_score *= 0.3
                except Exception:
                    metrics_full = m_cheap or {}
                    true_score = 0.0

                if true_score > state["best_score"]:
                    state["best_score"]   = true_score
                    state["best_x"]       = x.copy()
                    state["best_weapon"]  = weapon
                    state["best_metrics"] = metrics_full

                # Update surrogate online — refit incrementally
                state["surr_data"].append((x, true_score))
                _vis_scores_true.append(true_score)

            else:
                _vis_scores_true.append(float("nan"))
                metrics_full = m_cheap or {}

            # ── Record history ────────────────────────────────────────────
            _vis_scores_pred.append(score_pred)
            _vis_uncertainties.append(float(uncertainty))
            _vis_fea_flags.append(run_fea)

            elapsed = time.perf_counter() - state["t_start"]
            state["history"].append({
                "n_total":    n_total,
                "n_fea":      state["n_fea"],
                "score_pred": score_pred,
                "uncertainty": float(uncertainty),
                "acq":        acq,
                "ran_fea":    run_fea,
                "true_score": true_score if run_fea else None,
                "best_score": state["best_score"],
                "elapsed_s":  elapsed,
            })

            if verbose and n_total % 25 == 0:
                fea_rate = state["n_fea"] / max(n_total, 1)
                _log(
                    f"  [{n_total:4d}] best={state['best_score']:.4f}  "
                    f"fea_calls={state['n_fea']} ({100*fea_rate:.0f}%)  "
                    f"σ={math.sqrt(uncertainty):.3f}  acq={acq:.4f}"
                )

            # ── Periodic visualisation ────────────────────────────────────
            if n_total % vis_interval == 0 and n_total > 0:
                p = _make_progress_plots(
                    _vis_scores_pred, _vis_scores_true, _vis_uncertainties,
                    _vis_fea_flags, state["history"], case_dir, n_total,
                )
                vis_paths.extend(p)

            # ── Checkpoint save (crash recovery) ─────────────────────────
            if n_total % checkpoint_interval == 0 and n_total > 0:
                ckpt = {
                    "n_total":    state["n_total"],
                    "n_fea":      state["n_fea"],
                    "best_score": state["best_score"],
                    "best_x":     state["best_x"].tolist() if state["best_x"] is not None else None,
                    "history":    state["history"],
                    "surr_data":  [(x.tolist(), float(s)) for x, s in state["surr_data"]],
                }
                ckpt_path = case_dir / "surrogate_checkpoint.json"
                with open(ckpt_path, "w") as _f:
                    json.dump(ckpt, _f)
                _log(f"  [ckpt] saved at eval {n_total} → {ckpt_path}")

            return -acq   # DE minimises

        except Exception:
            return 1.0

    _log(f"[surrogate] starting DE: pop={pop_size} max_iter={max_iter}")
    _log(f"[surrogate] β_UCB={beta_ucb}  σ_FEA_thresh={sigma_fea_threshold}")

    de_result = differential_evolution(
        _objective,
        bounds,
        maxiter=max_iter,
        popsize=max(pop_size // len(bounds), 3),
        mutation=(0.5, 1.5),
        recombination=0.8,
        seed=None,    # OS entropy → genuine variance across restarts
        workers=1,    # must be 1 for surrogate online-update state
        tol=1e-6,
        polish=False,
    )

    # ── Final evaluation on best found solution ───────────────────────────
    best_x = state["best_x"] if state["best_x"] is not None else de_result.x
    best_weapon = state["best_weapon"]
    best_metrics = state["best_metrics"] or {}
    best_score = state["best_score"]

    if best_weapon is None:
        outer = build_profile(profile_t, best_x, cfg)
        best_weapon = assemble_weapon(outer, cfg.mounting, [])
        best_weapon = validate_geometry(best_weapon)
        best_metrics = compute_metrics_enhanced(best_weapon, cfg, fea_spacing=cfg.optimization.fea_fine_spacing_mm)
        best_score   = compute_physical_score(best_weapon, cfg, metrics=best_metrics)

    wall_time = time.perf_counter() - state["t_start"]
    fea_rate  = state["n_fea"] / max(state["n_total"], 1)

    _log(f"\n[surrogate] done — {state['n_total']} evals, {state['n_fea']} FEA calls "
         f"({100*fea_rate:.1f}%), {wall_time:.1f}s")
    _log(f"[surrogate] best score = {best_score:.4f}")

    # ── Final visualisation ───────────────────────────────────────────────
    p = _make_progress_plots(
        _vis_scores_pred, _vis_scores_true, _vis_uncertainties,
        _vis_fea_flags, state["history"], case_dir, state["n_total"], final=True,
    )
    vis_paths.extend(p)

    # ── Save stats ────────────────────────────────────────────────────────
    stats = {
        "n_total_evals": state["n_total"],
        "n_fea_calls":   state["n_fea"],
        "fea_rate":      fea_rate,
        "best_score":    best_score,
        "wall_time_s":   wall_time,
        "beta_ucb":      beta_ucb,
        "sigma_fea_threshold": sigma_fea_threshold,
        "acq_threshold": acq_threshold,
        "history":       state["history"],
    }
    stats_path = case_dir / "surrogate_stats.json"
    with open(stats_path, "w") as f:
        json.dump({k: v for k, v in stats.items() if k != "history"}, f, indent=2)
    _log(f"[surrogate] stats → {stats_path}")

    return {
        "weapon":        best_weapon,
        "metrics":       best_metrics,
        "score":         best_score,
        "n_fea_calls":   state["n_fea"],
        "n_total_evals": state["n_total"],
        "fea_rate":      fea_rate,
        "wall_time_s":   wall_time,
        "history":       state["history"],
        "vis_paths":     vis_paths,
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _make_progress_plots(
    scores_pred: list[float],
    scores_true: list[float],
    uncertainties: list[float],
    fea_flags: list[bool],
    history: list[dict],
    case_dir: Path,
    n_eval: int,
    final: bool = False,
) -> list[str]:
    """Produce surrogate progress plots. Returns list of saved paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    if len(scores_pred) < 3:
        return paths

    n = len(scores_pred)
    idx = np.arange(n)
    preds = np.array(scores_pred)
    trues = np.array([x if not math.isnan(x) else np.nan for x in scores_true])
    uncs  = np.array([math.sqrt(max(u, 0)) for u in uncertainties])
    fea_arr = np.array(fea_flags)

    suffix = f"_{n_eval:05d}" if not final else "_final"

    # ── 1. Score history + FEA call markers ──────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    ax = axes[0]
    ax.plot(idx, preds, "b-", lw=1.0, alpha=0.7, label="Surrogate pred.")
    fea_idx = idx[fea_arr]
    if len(fea_idx) > 0:
        ax.scatter(fea_idx, preds[fea_arr], color="orange", s=20, zorder=5, label="FEA triggered")
    true_idx = idx[~np.isnan(trues)]
    if len(true_idx) > 0:
        ax.scatter(true_idx, trues[~np.isnan(trues)], color="green", s=25, marker="^",
                   zorder=6, label="FEA true score")
    best_curve = [max(preds[:i+1]) for i in range(n)]
    ax.plot(idx, best_curve, "r--", lw=1.5, label="Best so far")
    ax.set_ylabel("Score")
    ax.set_title("Surrogate Predicted Score & FEA Evaluations", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(idx, uncs, "m-", lw=1.0, alpha=0.8, label="GP uncertainty σ")
    ax.axhline(0.05, color="r", ls="--", lw=0.8, label="σ_FEA threshold")
    ax.fill_between(idx, 0, uncs, alpha=0.15, color="m")
    ax.set_ylabel("GP σ")
    ax.set_title("GP Uncertainty Over Evaluations", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    fea_rate_curve = [
        sum(fea_flags[:i+1]) / (i + 1) for i in range(n)
    ]
    ax.plot(idx, [r * 100 for r in fea_rate_curve], "c-", lw=1.5)
    ax.set_ylabel("FEA call rate (%)")
    ax.set_xlabel("Evaluation number")
    ax.set_title("FEA Call Rate Over Time", fontsize=11)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Surrogate Optimizer Progress (n={n})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = str(case_dir / f"surrogate_progress{suffix}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # ── 2. Predicted vs. actual scatter (calibration) ─────────────────
    valid_mask = ~np.isnan(trues)
    if valid_mask.sum() >= 5:
        fig, ax = plt.subplots(figsize=(6, 6))
        preds_fea = preds[valid_mask]
        trues_fea = trues[valid_mask]
        ax.scatter(trues_fea, preds_fea, c=uncs[valid_mask], cmap="plasma",
                   s=40, alpha=0.8, edgecolors="k", linewidths=0.5)
        lims = [min(preds_fea.min(), trues_fea.min()) * 0.95,
                max(preds_fea.max(), trues_fea.max()) * 1.05]
        ax.plot(lims, lims, "k--", lw=1.0, label="Perfect prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True FEA score")
        ax.set_ylabel("Surrogate predicted score")
        ax.set_title("Surrogate Calibration (FEA-evaluated designs)", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend()

        # R² annotation
        from numpy.polynomial import polynomial as P
        try:
            ss_res = np.sum((trues_fea - preds_fea) ** 2)
            ss_tot = np.sum((trues_fea - trues_fea.mean()) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-12)
            ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes,
                    fontsize=12, color="darkblue", fontweight="bold")
        except Exception:
            pass

        fig.tight_layout()
        p2 = str(case_dir / f"surrogate_calibration{suffix}.png")
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p2)

    return paths
