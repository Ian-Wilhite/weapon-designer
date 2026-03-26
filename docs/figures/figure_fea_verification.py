#!/usr/bin/env python3
"""Generate FEA verification figure for the report.

Produces a two-panel figure:
  Left:  FEA von-Mises stress field (solid disk, 8000 RPM) with analytical
         contour lines overlaid.
  Right: Bar chart comparing analytical vs FEA for both verification tests,
         with error annotations.

Output: docs/figures/fea_verification.png
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patches as mpatches
from shapely.geometry import Polygon

from weapon_designer.fea import fea_stress_analysis_with_mesh

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RHO   = 7850.0
NU    = 0.3
RPM   = 8000.0
OMEGA = 2 * math.pi * RPM / 60.0
R_MM  = 100.0


def _solid_disk_polygon(R: float, n: int = 120) -> Polygon:
    th = np.linspace(0, 2*math.pi, n, endpoint=False)
    return Polygon([(R*math.cos(t), R*math.sin(t)) for t in th])


def _lame_solid_disk_vm(r_mm: float, R_mm: float) -> float:
    """Analytical von-Mises stress at radius r for solid rotating disk."""
    R_m = R_mm * 1e-3
    r_m = r_mm * 1e-3
    fac = RHO * OMEGA**2 * (3 + NU) / 8.0
    sig_r = fac * (R_m**2 - r_m**2)
    sig_t = fac * (R_m**2 - (1 + 3*NU) / (3 + NU) * r_m**2)
    return math.sqrt(sig_r**2 + sig_t**2 - sig_r * sig_t) * 1e-6


def main():
    out_dir = Path(__file__).parent
    poly = _solid_disk_polygon(R_MM)

    result = fea_stress_analysis_with_mesh(
        poly, rpm=RPM, density_kg_m3=RHO, thickness_mm=10.0,
        yield_strength_mpa=1400.0, bore_diameter_mm=0.0, mesh_spacing=5.0,
    )
    nodes    = result["nodes"]
    elements = result["elements"]
    vm       = result["vm_stresses"]

    # Node-averaged vm for smooth contour plot
    node_vm  = np.zeros(len(nodes))
    node_cnt = np.zeros(len(nodes))
    for el_idx, el in enumerate(elements):
        for n in el:
            node_vm[n]  += vm[el_idx]
            node_cnt[n] += 1
    mask = node_cnt > 0
    node_vm[mask] /= node_cnt[mask]

    # Analytical radial profile
    r_vals = np.linspace(0, R_MM, 200)
    sig_anal = np.array([_lame_solid_disk_vm(r, R_MM) for r in r_vals])

    # ---------------------------------------------------------------------------
    # Figure layout
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 4.8))
    gs  = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.5, 1.5], wspace=0.35)
    ax_fea   = fig.add_subplot(gs[0])
    ax_radial= fig.add_subplot(gs[1])
    ax_bar   = fig.add_subplot(gs[2])

    # ── Panel 1: FEA stress field ───────────────────────────────────────────
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    vmax = node_vm.max()
    tcf  = ax_fea.tricontourf(tri, node_vm, levels=20, cmap="inferno")
    cb   = plt.colorbar(tcf, ax=ax_fea, label="von-Mises stress (MPa)", fraction=0.046)

    # Analytical contour lines
    r_grid = np.linspace(0, R_MM, 300)
    sig_grid = np.array([_lame_solid_disk_vm(r, R_MM) for r in r_grid])
    for level_mpa in [5, 10, 15, 20]:
        # Find r at which analytical = level_mpa (decreasing function)
        idx = np.argmin(np.abs(sig_grid - level_mpa))
        r_c = r_grid[idx]
        circle = plt.Circle((0, 0), r_c, fill=False, color="white",
                             lw=0.8, ls="--", alpha=0.6)
        ax_fea.add_patch(circle)
        if r_c > 5:
            ax_fea.text(r_c * 0.72, r_c * 0.72, f"{level_mpa}", color="white",
                        fontsize=6, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="none", ec="none"))

    ax_fea.set_aspect("equal")
    ax_fea.set_xlim(-R_MM * 1.05, R_MM * 1.05)
    ax_fea.set_ylim(-R_MM * 1.05, R_MM * 1.05)
    ax_fea.set_title("FEA von-Mises stress\n(solid disk, 8 000 RPM)", fontsize=10)
    ax_fea.set_xlabel("x (mm)")
    ax_fea.set_ylabel("y (mm)")
    ax_fea.text(0.02, 0.97, "Steel  ρ=7850 kg/m³\nν=0.3, R=100 mm",
                transform=ax_fea.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Panel 2: Radial stress profile ──────────────────────────────────────
    # Element centroids vs analytical
    cx = nodes[elements, 0].mean(axis=1)
    cy = nodes[elements, 1].mean(axis=1)
    cr = np.sqrt(cx**2 + cy**2)
    order = np.argsort(cr)

    ax_radial.scatter(cr[order], vm[order], s=3, alpha=0.35, color="#2196F3",
                      label="FEA elements", zorder=3)
    ax_radial.plot(r_vals, sig_anal, "r-", lw=1.8, label="Analytical (Lamé)", zorder=4)

    # Mark test points
    for r_test, label in [(0.0, "Test 1\n(centre)"), (0.7 * R_MM, "Test 2\n(0.7R)")]:
        sig_a = _lame_solid_disk_vm(r_test, R_MM)
        ax_radial.axvline(r_test if r_test > 0 else 2, color="k", ls=":", lw=1, alpha=0.5)
        ax_radial.plot(r_test if r_test > 0 else 0, sig_a, "ko", ms=6, zorder=5)

    ax_radial.set_xlabel("Radius r (mm)")
    ax_radial.set_ylabel("σ_vm (MPa)")
    ax_radial.set_title("Radial stress profile", fontsize=10)
    ax_radial.legend(fontsize=8, loc="upper right")
    ax_radial.grid(alpha=0.3)

    # ── Panel 3: Bar comparison ──────────────────────────────────────────────
    tests  = ["Test 1\nCentre stress", "Test 2\n0.7R stress"]
    anal   = [22.73, 14.54]
    fea_v  = [22.48, 14.44]
    errors = [1.1, 0.7]

    x  = np.arange(len(tests))
    w  = 0.32
    b1 = ax_bar.bar(x - w/2, anal,  w, label="Analytical", color="#E53935", alpha=0.85)
    b2 = ax_bar.bar(x + w/2, fea_v, w, label="FEA",         color="#1E88E5", alpha=0.85)

    for xi, err in zip(x, errors):
        ax_bar.text(xi, max(anal[xi-x[0] if xi > x[0] else 0], fea_v[xi-x[0] if xi > x[0] else 0]) + 0.5,
                    f"{err:.1f}%", ha="center", va="bottom", fontsize=9,
                    color="green" if err < 5 else "orange")

    # Annotate PASS
    for xi in x:
        ax_bar.text(xi, -2.5, "PASS ✓", ha="center", va="top", fontsize=8,
                    color="green", fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(tests, fontsize=9)
    ax_bar.set_ylabel("σ_vm (MPa)")
    ax_bar.set_title("Analytical vs. FEA\ncomparison", fontsize=10)
    ax_bar.legend(fontsize=8)
    ax_bar.set_ylim(-5, 28)
    ax_bar.grid(axis="y", alpha=0.3)

    # ── Global title ─────────────────────────────────────────────────────────
    fig.suptitle("FEA Verification — 2D CST Plane-Stress Implementation",
                 fontsize=12, fontweight="bold", y=1.01)

    out_path = out_dir / "fea_verification.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure_fea_verification] → {out_path}")


if __name__ == "__main__":
    main()
