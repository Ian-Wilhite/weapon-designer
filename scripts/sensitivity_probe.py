"""Sensitivity probe: perturb each control parameter ±δ and measure metric changes.

Loads a weapon configuration and a reference parameter vector (from an
output_stats.json or supplied on the command line), then perturbs each
dimension by ±δ (default 5 % of its range) and computes the change in each
individual metric component.  Outputs a table and optional bar-chart PNG ranked
by total sensitivity.

This is useful for understanding which control parameters the optimizer is most
sensitive to, and can guide decisions about reducing dimensionality or adjusting
mutation rates.

Usage
-----
    python3 scripts/sensitivity_probe.py configs/my_config.json runs/my_run/
    python3 scripts/sensitivity_probe.py configs/my_config.json runs/my_run/ --delta 0.03
    python3 scripts/sensitivity_probe.py configs/my_config.json runs/my_run/ --phase p2
    python3 scripts/sensitivity_probe.py configs/my_config.json runs/my_run/ --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_best_params(stats_path: Path, phase: str) -> np.ndarray | None:
    """Extract best parameter vector from output_stats.json."""
    with open(stats_path) as f:
        data = json.load(f)

    key = "best_bspline_params" if phase == "p1" else "best_cutout_params"
    if key in data and data[key]:
        return np.array(data[key], dtype=float)

    # Fallback: last entry in convergence history doesn't have x, use bounds midpoint
    return None


def _evaluate_one(x: np.ndarray, cfg, profile_type: str, phase: str,
                  best_profile: np.ndarray | None) -> dict:
    """Evaluate metrics for one parameter vector."""
    from weapon_designer.profile_builder import build_profile
    from weapon_designer.objectives_enhanced import compute_metrics_enhanced, weighted_score_enhanced
    from weapon_designer.parametric_cad import make_cutouts_polar, mass_normalize_cutouts, CUTOUT_STRIDE_POLAR
    from weapon_designer.geometry import assemble_weapon

    if phase == "p1":
        outer = build_profile(profile_type, x, cfg)
        if outer is None or outer.is_empty:
            return {}
        weapon = assemble_weapon(outer, cfg.mounting)
        if weapon is None or weapon.is_empty:
            return {}
    else:
        if best_profile is None:
            return {}
        from weapon_designer.profile_builder import build_profile
        outer = build_profile(profile_type, best_profile, cfg)
        if outer is None or outer.is_empty:
            return {}
        n_pairs = cfg.optimization.num_cutout_pairs
        x_norm = mass_normalize_cutouts(x, outer, cfg)
        cutouts = make_cutouts_polar(x_norm, n_pairs)
        weapon = assemble_weapon(outer, cfg.mounting, cutouts)
        if weapon is None or weapon.is_empty:
            return {}

    metrics = compute_metrics_enhanced(weapon, cfg)
    metrics["score"] = weighted_score_enhanced(metrics, cfg)
    return metrics


def sensitivity_probe(
    cfg_path: Path,
    run_dir: Path,
    delta_frac: float,
    phase: str,
    n_samples: int,
    plot: bool,
    out: Path | None,
) -> None:
    from weapon_designer.config import load_config
    from weapon_designer.profile_builder import get_profile_bounds
    from weapon_designer.parametric_cad import get_cutout_bounds_polar

    cfg = load_config(cfg_path)
    profile_type = getattr(cfg.optimization, "profile_type", "bspline")

    stats_path = run_dir / "output_stats.json"
    if not stats_path.exists():
        # Try searching
        candidates = list(run_dir.rglob("output_stats.json"))
        if candidates:
            stats_path = candidates[0]
        else:
            print(f"No output_stats.json found under {run_dir}", file=sys.stderr)
            sys.exit(1)

    # ── Load reference params ─────────────────────────────────────────────
    if phase == "p1":
        bounds = get_profile_bounds(profile_type, cfg)
        best_profile = None
    else:
        bounds = get_cutout_bounds_polar(cfg)
        # Load P1 best to build outer profile
        try:
            with open(stats_path) as f:
                data = json.load(f)
            best_profile_raw = data.get("best_bspline_params")
            best_profile = np.array(best_profile_raw, dtype=float) if best_profile_raw else None
        except Exception:
            best_profile = None

    x0 = _load_best_params(stats_path, phase)
    if x0 is None:
        # Use midpoint of bounds
        x0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds])
        print(f"Warning: best params not found, using bounds midpoint.", file=sys.stderr)

    n_dims = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    ranges = hi - lo

    # ── Evaluate baseline ─────────────────────────────────────────────────
    print(f"Evaluating baseline ({phase.upper()}, {n_dims} dims)...", flush=True)
    base_metrics = _evaluate_one(x0, cfg, profile_type, phase, best_profile)
    if not base_metrics:
        print("Baseline evaluation failed — check config and params.", file=sys.stderr)
        sys.exit(1)

    metric_keys = ["score", "moi_kg_mm2", "bite_mm", "structural_integrity",
                   "fea_safety_factor", "mass_utilization", "com_offset_mm"]
    metric_keys = [k for k in metric_keys if k in base_metrics]

    print(f"\nBaseline metrics:")
    for k in metric_keys:
        print(f"  {k:30s}: {base_metrics[k]:.4f}")

    # ── Perturb each dimension ────────────────────────────────────────────
    delta = delta_frac * ranges
    sensitivities: dict[int, dict[str, float]] = {}

    print(f"\nPerturbing {n_dims} parameters by ±{delta_frac*100:.1f}% of range...")
    for i in range(n_dims):
        sens_i: dict[str, float] = {}
        for sign in (+1.0, -1.0):
            x_pert = x0.copy()
            x_pert[i] = np.clip(x0[i] + sign * delta[i], lo[i], hi[i])
            m = _evaluate_one(x_pert, cfg, profile_type, phase, best_profile)
            if not m:
                continue
            for k in metric_keys:
                base_v = base_metrics.get(k, 0.0)
                pert_v = m.get(k, 0.0)
                diff = abs(pert_v - base_v)
                sens_i[k] = max(sens_i.get(k, 0.0), diff)

        sensitivities[i] = sens_i
        if (i + 1) % 5 == 0 or i == n_dims - 1:
            print(f"  [{i+1}/{n_dims}]  dim {i:3d}  "
                  f"score_sensitivity={sens_i.get('score', 0.0):.5f}", flush=True)

    # ── Rank by total sensitivity ─────────────────────────────────────────
    total_sens = {i: sum(sensitivities[i].values()) for i in range(n_dims)}
    ranked = sorted(total_sens.items(), key=lambda kv: kv[1], reverse=True)

    print("\n── Top 20 most-sensitive parameters ────────────────────────────")
    print(f"{'dim':>5}  {'lo':>8}  {'hi':>8}  {'x0':>8}  {'total_sens':>12}  "
          + "  ".join(f"{k[:10]:>10}" for k in metric_keys))
    for rank, (i, total) in enumerate(ranked[:20]):
        row = (f"{i:>5d}  {lo[i]:>8.2f}  {hi[i]:>8.2f}  {x0[i]:>8.2f}  {total:>12.5f}  "
               + "  ".join(f"{sensitivities[i].get(k, 0.0):>10.5f}" for k in metric_keys))
        print(row)

    # ── Save summary JSON ─────────────────────────────────────────────────
    out_dir = out or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": phase,
        "delta_frac": delta_frac,
        "n_dims": n_dims,
        "baseline": {k: base_metrics.get(k) for k in metric_keys},
        "sensitivities": {
            str(i): {k: sensitivities[i].get(k, 0.0) for k in metric_keys}
            for i in range(n_dims)
        },
        "ranked": [[i, total] for i, total in ranked],
    }
    json_path = out_dir / f"sensitivity_{phase}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {json_path}")

    # ── Optional bar chart ────────────────────────────────────────────────
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        top_n = min(30, n_dims)
        top_dims = [i for i, _ in ranked[:top_n]]
        top_scores = [sensitivities[i].get("score", 0.0) for i in top_dims]

        bars = ax.bar(range(top_n), top_scores, color="steelblue", alpha=0.85)
        ax.set_xticks(range(top_n))
        ax.set_xticklabels([f"d{i}" for i in top_dims], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Parameter index (ranked by total sensitivity)")
        ax.set_ylabel("|Δscore| from ±δ perturbation")
        ax.set_title(f"Parameter Sensitivity — {phase.upper()} ({profile_type}, δ={delta_frac*100:.0f}%)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        png_path = out_dir / f"sensitivity_{phase}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        del fig
        print(f"Saved: {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config",  type=Path, help="Weapon config JSON path")
    parser.add_argument("run_dir", type=Path, help="Run output directory (contains output_stats.json)")
    parser.add_argument("--delta",   type=float, default=0.05,
                        help="Perturbation as fraction of parameter range (default: 0.05 = 5%%)")
    parser.add_argument("--phase",   choices=["p1", "p2"], default="p1",
                        help="Which phase to probe (default: p1)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Not yet used (reserved for future Monte Carlo probing)")
    parser.add_argument("--plot",    action="store_true", help="Generate bar-chart PNG")
    parser.add_argument("--out",     type=Path, default=None,
                        help="Output directory for JSON/PNG (default: run_dir)")
    args = parser.parse_args()

    sensitivity_probe(
        cfg_path=args.config,
        run_dir=args.run_dir,
        delta_frac=args.delta,
        phase=args.phase,
        n_samples=args.samples,
        plot=args.plot,
        out=args.out,
    )


if __name__ == "__main__":
    main()
