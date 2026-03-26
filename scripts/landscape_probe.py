"""2D objective landscape slice: vary two parameters, plot score heatmap.

Fixes all parameters except two (selected by index), then sweeps those two
over their bounds on a regular grid and evaluates the Stage-1 cheap objective
(no FEA) at each grid point.  Produces a heatmap PNG showing the objective
landscape for the chosen 2D slice.

This is useful for:
  • Checking whether the landscape has many local optima (multi-modal)
  • Understanding parameter interactions (ridges, valleys)
  • Verifying that promising designs lie in smooth basins

Stage-1 scoring (no FEA) is used so the probe runs in seconds/minutes rather
than hours.  FEA scoring can be enabled with --fea.

Usage
-----
    python3 scripts/landscape_probe.py configs/my_config.json
    python3 scripts/landscape_probe.py configs/my_config.json --dim1 0 --dim2 1
    python3 scripts/landscape_probe.py configs/my_config.json --resolution 40 --fea
    python3 scripts/landscape_probe.py configs/my_config.json --run-dir runs/my_run/ --phase p2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _stage1_score(weapon, cfg) -> float:
    """Cheap Stage-1 score without FEA (same as EvalGate.stage1_score)."""
    from weapon_designer.staged_eval import EvalGate
    gate = EvalGate(cfg, gate_frac=1.0)  # no gating, just use as scorer
    return gate.stage1_score(weapon)


def _full_score(weapon, cfg) -> float:
    """Full score with FEA."""
    from weapon_designer.objectives_enhanced import compute_metrics_enhanced, weighted_score_enhanced
    metrics = compute_metrics_enhanced(weapon, cfg)
    return weighted_score_enhanced(metrics, cfg)


def _build_weapon(x: np.ndarray, cfg, profile_type: str,
                  phase: str, best_profile: np.ndarray | None):
    """Build weapon polygon from parameter vector."""
    from weapon_designer.profile_builder import build_profile
    from weapon_designer.geometry import assemble_weapon
    from weapon_designer.parametric_cad import make_cutouts_polar, mass_normalize_cutouts

    if phase == "p1":
        outer = build_profile(profile_type, x, cfg)
        if outer is None or outer.is_empty:
            return None
        return assemble_weapon(outer, cfg.mounting)
    else:
        if best_profile is None:
            return None
        outer = build_profile(profile_type, best_profile, cfg)
        if outer is None or outer.is_empty:
            return None
        n_pairs = cfg.optimization.num_cutout_pairs
        x_norm = mass_normalize_cutouts(x, outer, cfg)
        cutouts = make_cutouts_polar(x_norm, n_pairs)
        return assemble_weapon(outer, cfg.mounting, cutouts)


def landscape_probe(
    cfg_path: Path,
    run_dir: Path | None,
    dim1: int,
    dim2: int,
    resolution: int,
    phase: str,
    use_fea: bool,
    out: Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    from weapon_designer.config import load_config
    from weapon_designer.profile_builder import get_profile_bounds
    from weapon_designer.parametric_cad import get_cutout_bounds_polar

    cfg = load_config(cfg_path)
    profile_type = getattr(cfg.optimization, "profile_type", "bspline")

    if phase == "p1":
        bounds = get_profile_bounds(profile_type, cfg)
        best_profile = None
    else:
        bounds = get_cutout_bounds_polar(cfg)
        best_profile = None
        if run_dir:
            stats_candidates = list(run_dir.rglob("output_stats.json"))
            if stats_candidates:
                try:
                    with open(stats_candidates[0]) as f:
                        data = json.load(f)
                    bp = data.get("best_bspline_params")
                    if bp:
                        best_profile = np.array(bp, dtype=float)
                except Exception:
                    pass

    n_dims = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    if dim1 >= n_dims or dim2 >= n_dims:
        print(f"Error: dim1={dim1}, dim2={dim2} but n_dims={n_dims}", file=sys.stderr)
        sys.exit(1)
    if dim1 == dim2:
        print("Error: dim1 and dim2 must be different.", file=sys.stderr)
        sys.exit(1)

    # ── Reference point: bounds midpoint or loaded best ────────────────────
    x0 = (lo + hi) / 2.0
    if run_dir:
        stats_candidates = list(run_dir.rglob("output_stats.json"))
        if stats_candidates:
            try:
                with open(stats_candidates[0]) as f:
                    data = json.load(f)
                key = "best_bspline_params" if phase == "p1" else "best_cutout_params"
                raw = data.get(key)
                if raw and len(raw) == n_dims:
                    x0 = np.array(raw, dtype=float)
                    print(f"Using loaded best params as reference point.")
            except Exception:
                pass

    # ── Grid ───────────────────────────────────────────────────────────────
    v1 = np.linspace(lo[dim1], hi[dim1], resolution)
    v2 = np.linspace(lo[dim2], hi[dim2], resolution)
    grid = np.zeros((resolution, resolution))
    scorer = _full_score if use_fea else _stage1_score
    mode_label = "FEA" if use_fea else "Stage-1 (no FEA)"

    print(f"Sweeping dim {dim1} × dim {dim2}  ({resolution}×{resolution} = {resolution**2} evals)")
    print(f"Scoring mode: {mode_label}")
    print(f"dim {dim1}: [{lo[dim1]:.1f}, {hi[dim1]:.1f}]")
    print(f"dim {dim2}: [{lo[dim2]:.1f}, {hi[dim2]:.1f}]")

    for i, d1 in enumerate(v1):
        for j, d2 in enumerate(v2):
            x = x0.copy()
            x[dim1] = d1
            x[dim2] = d2
            try:
                weapon = _build_weapon(x, cfg, profile_type, phase, best_profile)
                if weapon is None or weapon.is_empty:
                    grid[j, i] = 0.0
                else:
                    grid[j, i] = scorer(weapon, cfg)
            except Exception:
                grid[j, i] = 0.0
        if (i + 1) % 5 == 0 or i == resolution - 1:
            print(f"  [{i+1}/{resolution}]  max_score_so_far={grid.max():.4f}", flush=True)

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.contourf(v1, v2, grid, levels=30, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Score")

    # Mark reference point
    ax.plot(x0[dim1], x0[dim2], "r*", markersize=12, label="reference")
    # Mark optimum
    peak = np.unravel_index(grid.argmax(), grid.shape)
    ax.plot(v1[peak[1]], v2[peak[0]], "w^", markersize=10, label=f"max={grid.max():.4f}")

    ax.set_xlabel(f"dim {dim1}  [{lo[dim1]:.1f}, {hi[dim1]:.1f}]")
    ax.set_ylabel(f"dim {dim2}  [{lo[dim2]:.1f}, {hi[dim2]:.1f}]")
    ax.set_title(f"Landscape Slice — {phase.upper()}  |  {mode_label}\n"
                 f"dim {dim1} × dim {dim2}  (others fixed at reference)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_dir = out or (run_dir or Path("."))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"landscape_{phase}_d{dim1}xd{dim2}.png"
    fig.savefig(png_path, dpi=150)
    print(f"\nSaved: {png_path}")

    # Save grid as npy for further analysis
    npy_path = out_dir / f"landscape_{phase}_d{dim1}xd{dim2}.npy"
    np.save(npy_path, grid)
    print(f"Saved: {npy_path}")

    if show:
        plt.show()
    plt.close(fig)
    del fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="Weapon config JSON path")
    parser.add_argument("--run-dir",    type=Path, default=None,
                        help="Run output directory (used to load reference params)")
    parser.add_argument("--dim1",       type=int, default=0,
                        help="First parameter dimension to sweep (default: 0)")
    parser.add_argument("--dim2",       type=int, default=1,
                        help="Second parameter dimension to sweep (default: 1)")
    parser.add_argument("--resolution", type=int, default=30,
                        help="Grid resolution per dimension (default: 30)")
    parser.add_argument("--phase",      choices=["p1", "p2"], default="p1",
                        help="Phase to probe (default: p1)")
    parser.add_argument("--fea",        action="store_true",
                        help="Use full FEA scoring instead of Stage-1 (slow)")
    parser.add_argument("--out",        type=Path, default=None,
                        help="Output directory for PNG/npy (default: run_dir or .)")
    parser.add_argument("--show",       action="store_true",
                        help="Show interactive plot")
    args = parser.parse_args()

    landscape_probe(
        cfg_path=args.config,
        run_dir=args.run_dir,
        dim1=args.dim1,
        dim2=args.dim2,
        resolution=args.resolution,
        phase=args.phase,
        use_fea=args.fea,
        out=args.out,
        show=args.show,
    )


if __name__ == "__main__":
    main()
