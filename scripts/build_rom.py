#!/usr/bin/env python3
"""Build the POD/GP surrogate from the FEA database.

Steps
-----
1. Load all design_NNNN.npz files from the database directory
2. Assemble stress data matrix X: (N × N_ref_elements)
3. Mean-centre and compute truncated SVD (POD decomposition)
4. Retain k modes capturing ≥ 99% variance (or --n-modes explicit)
5. Fit k independent GP regressors (one per POD coefficient)
6. Save the fitted FEASurrogate to fea_surrogate.pkl
7. Produce visualisation: variance curve, mode shapes, GP training summary

Outputs
-------
  <out-dir>/fea_surrogate.pkl          — fitted FEASurrogate (use with validate_rom.py)
  <out-dir>/variance_explained.png     — cumulative variance curve
  <out-dir>/mode_shapes.png            — first 4 POD mode shape images
  <out-dir>/gp_alpha_fits.png          — GP fit quality per mode (predicted vs. actual)
  <out-dir>/rom_summary.json           — metadata: k, variance, N, timings

Usage
-----
    python scripts/build_rom.py --db-dir fea_database
    python scripts/build_rom.py --db-dir fea_database --n-modes 20 --out-dir rom_output
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from weapon_designer.surrogate_fea import FEASurrogate

# Shared database loader — defined in build_fea_database.py
sys.path.insert(0, str(Path(__file__).parent))
from build_fea_database import load_database  # noqa: E402


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_variance_explained(
    surrogate: FEASurrogate,
    out_dir: Path,
):
    """Cumulative variance explained curve with k marked."""
    fig, ax = plt.subplots(figsize=(8, 5))

    cumvar = surrogate.variance_explained_ * 100
    modes  = np.arange(1, len(cumvar) + 1)

    ax.plot(modes, cumvar, "b-o", ms=4, lw=1.5, label="Cumulative variance")
    ax.axhline(99, color="r", ls="--", lw=1.0, label="99% threshold")
    ax.axvline(surrogate.k_, color="g", ls="--", lw=1.0, label=f"k={surrogate.k_} selected")

    ax.fill_between(modes, 0, cumvar, alpha=0.1, color="b")
    ax.set_xlim(1, min(len(cumvar), 50))
    ax.set_ylim(0, 102)
    ax.set_xlabel("Number of POD modes", fontsize=12)
    ax.set_ylabel("Cumulative variance explained (%)", fontsize=12)
    ax.set_title("POD/SVD Mode Decomposition — Variance Explained", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Annotate singular values drop-off
    S = surrogate.singular_values_
    ax2 = ax.twinx()
    ax2.semilogy(modes, S[:len(cumvar)], "m-", lw=1.0, alpha=0.5, label="Singular values")
    ax2.set_ylabel("Singular value (log scale)", color="m")
    ax2.tick_params(axis="y", labelcolor="m")

    path = out_dir / "variance_explained.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[build_rom] variance plot → {path}")


def plot_mode_shapes(
    surrogate: FEASurrogate,
    ref_nodes: np.ndarray,
    ref_elements: np.ndarray,
    out_dir: Path,
    n_modes: int = 4,
):
    """Visualise first n POD mode shapes as filled contour plots."""
    k_show = min(n_modes, surrogate.k_)
    if k_show == 0:
        return

    fig, axes = plt.subplots(1, k_show, figsize=(5 * k_show, 4.5))
    if k_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        mode_vals = surrogate.basis_[i]   # (N_ref_elements,)

        # One value per element → interpolate to nodes for smooth plot
        node_vals = np.zeros(len(ref_nodes))
        node_cnt  = np.zeros(len(ref_nodes))
        for el in ref_elements:
            for n in el:
                node_vals[n] += mode_vals[len(ref_elements) - 1] if i >= len(mode_vals) else mode_vals[min(i, len(mode_vals)-1)]
                node_cnt[n]  += 1

        # Use element centroid values directly for tricontourf
        tri = mtri.Triangulation(ref_nodes[:, 0], ref_nodes[:, 1], ref_elements)
        # Map element mode values to per-node via averaging
        node_mode = np.zeros(len(ref_nodes))
        node_cnt2 = np.zeros(len(ref_nodes))
        for el_idx, el in enumerate(ref_elements):
            v = mode_vals[el_idx] if el_idx < len(mode_vals) else 0.0
            for n in el:
                node_mode[n] += v
                node_cnt2[n] += 1
        mask2 = node_cnt2 > 0
        node_mode[mask2] /= node_cnt2[mask2]

        vmax = np.abs(node_mode).max()
        cf = ax.tricontourf(tri, node_mode, levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(cf, ax=ax, label="Mode amplitude")

        var_pct = float(
            (surrogate.singular_values_[i] ** 2) / max((surrogate.singular_values_ ** 2).sum(), 1e-12)
        ) * 100
        ax.set_title(f"Mode {i+1}  ({var_pct:.1f}% var)", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    fig.suptitle("POD Mode Shapes — von-Mises Stress Basis Functions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "mode_shapes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[build_rom] mode shape plot → {path}")


def plot_gp_alpha_fits(
    surrogate: FEASurrogate,
    params: np.ndarray,
    stresses: np.ndarray,
    out_dir: Path,
    n_modes: int = 4,
):
    """For each of the first n POD modes, plot GP predicted vs. true α_i."""
    k_show = min(n_modes, surrogate.k_)
    if k_show == 0:
        return

    from weapon_designer.surrogate_fea import _normalise_params

    # True POD coefficients
    X_tilde = stresses - surrogate.mean_stress_
    alpha_true = (X_tilde @ surrogate.basis_.T)   # (N, k)

    X_norm = _normalise_params(params, surrogate.param_lo_, surrogate.param_hi_)

    n_cols = 2
    n_rows = (k_show + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes[:k_show]):
        mu, std = surrogate.gps_[i].predict(X_norm, return_std=True)
        true = alpha_true[:, i]

        ax.errorbar(true, mu, yerr=2*std, fmt="o", ms=4, alpha=0.6, color="#2196F3",
                    ecolor="#90CAF9", capsize=2, lw=0.5, label="pred ± 2σ")

        lims = [min(true.min(), mu.min()), max(true.max(), mu.max())]
        ax.plot(lims, lims, "k--", lw=1.0, label="perfect")

        ss_res = np.sum((true - mu) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)

        ax.set_xlabel(f"True α_{i+1}")
        ax.set_ylabel(f"GP predicted α_{i+1}")
        ax.set_title(f"Mode {i+1}  R²={r2:.3f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[k_show:]:
        ax.set_visible(False)

    fig.suptitle("GP Fit Quality per POD Mode", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "gp_alpha_fits.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[build_rom] GP fit plot → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build POD/GP surrogate from FEA database")
    parser.add_argument("--db-dir",        default="fea_database",  help="FEA database directory")
    parser.add_argument("--out-dir",        default=None,            help="Output directory (default: db-dir)")
    parser.add_argument("--n-modes",        type=int, default=None,  help="Number of POD modes (default: 99%% variance)")
    parser.add_argument("--variance-threshold", type=float, default=0.99, help="Variance threshold for auto k selection")
    parser.add_argument("--n-restarts",     type=int, default=3,     help="GP kernel hyperparameter restarts")
    args = parser.parse_args()

    db_dir  = Path(args.db_dir)
    out_dir = Path(args.out_dir) if args.out_dir else db_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ROM BUILD — POD/SVD + GP surrogate")
    print(f"  Database: {db_dir}")
    print(f"  Output:   {out_dir}")
    print("=" * 60)

    # ── Load database ─────────────────────────────────────────────────────
    params, stresses = load_database(db_dir, verbose=True)
    N, d = params.shape

    # ── Load reference mesh for visualisation ─────────────────────────────
    ref_mesh_path = db_dir / "ref_mesh.npz"
    if ref_mesh_path.exists():
        ref_data = np.load(ref_mesh_path)
        ref_nodes, ref_elements = ref_data["nodes"], ref_data["elements"]
    else:
        ref_nodes, ref_elements = None, None
        print("[warn] ref_mesh.npz not found — mode shape plots skipped")

    # ── Fit surrogate ─────────────────────────────────────────────────────
    print(f"\nFitting surrogate (N={N}, d={d}, n_modes={args.n_modes or 'auto'})...")
    t0 = time.perf_counter()

    surrogate = FEASurrogate(
        n_modes=args.n_modes,
        variance_threshold=args.variance_threshold,
        kernel_params={"length_scale": 0.5, "noise_level": 1e-3},
    )
    surrogate.fit(params, stresses)

    dt_fit = time.perf_counter() - t0
    print(f"  k={surrogate.k_} modes  ({surrogate.variance_explained_[surrogate.k_-1]*100:.1f}% variance)")
    print(f"  GP fitting time: {dt_fit:.1f}s")

    # ── Save surrogate ────────────────────────────────────────────────────
    surr_path = out_dir / "fea_surrogate.pkl"
    surrogate.save(surr_path)

    # ── Visualisations ────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_variance_explained(surrogate, out_dir)

    if ref_nodes is not None and ref_elements is not None:
        plot_mode_shapes(surrogate, ref_nodes, ref_elements, out_dir, n_modes=min(4, surrogate.k_))
    else:
        print("[skip] mode shape plot (no ref_mesh.npz)")

    plot_gp_alpha_fits(surrogate, params, stresses, out_dir, n_modes=min(4, surrogate.k_))

    # ── ROM summary JSON ──────────────────────────────────────────────────
    summary = {
        "n_designs":          N,
        "n_params":           d,
        "n_ref_elements":     stresses.shape[1],
        "k_modes":            surrogate.k_,
        "variance_at_k":      float(surrogate.variance_explained_[surrogate.k_-1]),
        "variance_threshold": args.variance_threshold,
        "singular_values":    surrogate.singular_values_[:min(20, len(surrogate.singular_values_))].tolist(),
        "fit_time_s":         dt_fit,
        "surrogate_path":     str(surr_path),
    }
    with open(out_dir / "rom_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[build_rom] done — surrogate → {surr_path}")
    print(f"[build_rom] k={surrogate.k_} modes, variance={summary['variance_at_k']*100:.1f}%")


if __name__ == "__main__":
    main()
