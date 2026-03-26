#!/usr/bin/env python3
"""Generate ROM / surrogate summary figure for the report.

Four-panel figure:
  A. POD variance explained curve (cumulative) with k=63 and k=187 marked
  B. LOO residuals: predicted vs true peak stress for each design
  C. GP calibration: log(predicted σ) vs log(true error)
  D. Sample database designs coloured by stress magnitude

Output: docs/figures/rom_summary.png
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DB_DIR  = ROOT / "fea_database"
OUT_DIR = Path(__file__).parent


def load_surrogate(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_database(db_dir: Path):
    """Load params + stresses from the database directory."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_fea_database import load_database as _load
    return _load(db_dir, verbose=False)


def main():
    surr_path = DB_DIR / "fea_surrogate.pkl"
    if not surr_path.exists():
        print(f"[skip] surrogate not found at {surr_path}")
        return

    surr = load_surrogate(surr_path)
    sv   = surr.singular_values_
    cumvar = np.cumsum(sv**2) / (sv**2).sum() * 100

    # k thresholds
    k95  = int(np.searchsorted(cumvar, 95.0)) + 1
    k99  = int(np.searchsorted(cumvar, 99.0)) + 1
    k_used = surr.k_

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38)
    ax_var  = fig.add_subplot(gs[0])
    ax_loo  = fig.add_subplot(gs[1])
    ax_cal  = fig.add_subplot(gs[2])
    ax_db   = fig.add_subplot(gs[3])

    # ── Panel A: Variance explained ────────────────────────────────────────
    modes = np.arange(1, min(len(cumvar), 300) + 1)
    ax_var.plot(modes, cumvar[:len(modes)], "b-", lw=1.5, label="Cumulative var.")
    ax_var.axhline(95, color="orange", ls="--", lw=1.2, label="95% threshold")
    ax_var.axhline(99, color="red",    ls="--", lw=1.2, label="99% threshold")

    if k95 <= len(modes):
        ax_var.axvline(k95, color="orange", ls=":", lw=1.0, alpha=0.8)
        ax_var.text(k95 + 2, 40, f"k={k95}", color="orange", fontsize=8)
    if k99 <= len(modes):
        ax_var.axvline(k99, color="red",    ls=":", lw=1.0, alpha=0.8)
        ax_var.text(k99 + 2, 55, f"k={k99}", color="red",    fontsize=8)

    ax_var.fill_between(modes, 0, cumvar[:len(modes)], alpha=0.08, color="blue")
    ax_var.set_xlim(1, min(250, len(cumvar)))
    ax_var.set_ylim(0, 102)
    ax_var.set_xlabel("Number of POD modes", fontsize=9)
    ax_var.set_ylabel("Cumulative variance (%)", fontsize=9)
    ax_var.set_title("(a) POD variance decay\n$N=500$, $d=8$", fontsize=10)
    ax_var.legend(fontsize=8, loc="lower right")
    ax_var.grid(alpha=0.3)

    # Inset: singular values log scale
    ax_in = ax_var.inset_axes([0.55, 0.05, 0.42, 0.40])
    ax_in.semilogy(modes, sv[:len(modes)], "m-", lw=1.0)
    ax_in.set_xlabel("mode", fontsize=6)
    ax_in.set_ylabel("σ", fontsize=6)
    ax_in.tick_params(labelsize=6)
    ax_in.set_xlim(1, min(250, len(modes)))
    ax_in.grid(alpha=0.3)

    # ── Panel B: LOO residuals ─────────────────────────────────────────────
    loo_path = DB_DIR / "loo_residuals.png"
    if loo_path.exists():
        img_loo = plt.imread(str(loo_path))
        ax_loo.imshow(img_loo)
        ax_loo.axis("off")
        ax_loo.set_title("(b) LOO residuals\n(99% threshold, k=187)", fontsize=10)
    else:
        ax_loo.text(0.5, 0.5, "LOO plot\nnot available", ha="center", va="center",
                    transform=ax_loo.transAxes, fontsize=10, color="gray")
        ax_loo.set_title("(b) LOO residuals", fontsize=10)
        ax_loo.axis("off")

    # ── Panel C: Calibration ───────────────────────────────────────────────
    cal_path = DB_DIR / "calibration.png"
    if cal_path.exists():
        img_cal = plt.imread(str(cal_path))
        ax_cal.imshow(img_cal)
        ax_cal.axis("off")
        ax_cal.set_title("(c) GP calibration\n(predicted σ² vs. true error²)", fontsize=10)
    else:
        ax_cal.text(0.5, 0.5, "Calibration plot\nnot available", ha="center", va="center",
                    transform=ax_cal.transAxes, fontsize=10, color="gray")
        ax_cal.set_title("(c) GP calibration", fontsize=10)
        ax_cal.axis("off")

    # ── Panel D: Database sample designs ──────────────────────────────────
    db_path = DB_DIR / "sample_designs.png"
    if db_path.exists():
        img_db = plt.imread(str(db_path))
        ax_db.imshow(img_db)
        ax_db.axis("off")
        ax_db.set_title("(d) Sample database designs\n(Sobol-sampled B-splines)", fontsize=10)
    else:
        ax_db.text(0.5, 0.5, "Sample designs\nnot available", ha="center", va="center",
                   transform=ax_db.transAxes, fontsize=10, color="gray")
        ax_db.set_title("(d) Sample database", fontsize=10)
        ax_db.axis("off")

    fig.suptitle(
        f"ROM Infrastructure: POD/SVD + GP Surrogate  "
        f"(N=500 training designs, "
        f"d=8 B-spline params, k={k_used} modes at {getattr(surr, 'variance_threshold', 0.99)*100:.0f}% variance)",
        fontsize=11, fontweight="bold", y=1.01
    )

    out_path = OUT_DIR / "rom_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure_rom_summary] → {out_path}")


if __name__ == "__main__":
    main()
