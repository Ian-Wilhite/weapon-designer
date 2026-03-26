#!/usr/bin/env python3
"""Speed tradeoff figures: stacked stress snapshots + RPM/force Pareto plot.

Two figures:

  1. figure_speed_stacked.png
     Filmstrip: one row per archetype weapon, 5 columns of stress-field snapshots
     at representative RPM values (min, 25%, 50%, FOS=2, failure).  The weapon
     outline is shown in the first column.

  2. figure_speed_pareto.png
     Pareto-style plot: RPM (x) vs. impact force in N (y) for each archetype.
     Each weapon has two lines:
       — solid line   : region from 0 to RPM_fail (structural failure, SF=1)
       — dotted line  : region from 0 to RPM_FOS2 (conservative limit, SF=2)
     Vertical markers indicate RPM_fail and RPM_FOS2 per weapon.
     Different colours per weapon geometry.

Usage
-----
    python docs/figures/figure_speed_pareto.py
    python docs/figures/figure_speed_pareto.py \\
        --sweep-dir outputs/spiral_speed_sweep --dpi 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.tri as mtri
from matplotlib.lines import Line2D

SWEEP_DIR = ROOT / "outputs" / "spiral_speed_sweep"
OUT_DIR   = Path(__file__).parent

YIELD_MPa = 1600.0   # S7 Tool Steel — matches spiral weapon build config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def elements_to_nodes(
    elem_vals: np.ndarray,
    elements: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Average per-element scalar values to nodes (vectorised)."""
    node_vals = np.zeros(n_nodes)
    node_cnt  = np.zeros(n_nodes)
    np.add.at(node_vals, elements.ravel(), np.repeat(elem_vals, 3))
    np.add.at(node_cnt,  elements.ravel(), 1)
    return node_vals / np.maximum(node_cnt, 1)


def load_sweeps(sweep_dir: Path) -> list[dict]:
    """Load all *_sweep.npz files from sweep_dir, sorted by nc."""
    npz_files = sorted(sweep_dir.glob("*_sweep.npz"))
    if not npz_files:
        return []

    sweeps = []
    for p in npz_files:
        d = np.load(p, allow_pickle=True)
        n_starts = int(d["n_starts"][0])
        label = f"n_starts={n_starts}"
        sweeps.append({
            "label":           label,
            "nc":              n_starts,
            "file":            p.name,
            "rpms":            d["rpms"],
            "sf_vals":         d["sf_vals"],
            "peak_stress_mpa": d["peak_stress_mpa"],
            "f_impact_n":      d["f_impact_n"],
            "snap_indices":    d["snap_indices"],
            "snap_rpms":       d["snap_rpms"],
            "snap_stresses":   d["snap_stresses"],
            "snap_nodes":      d["snap_nodes"],
            "snap_elements":   d["snap_elements"].astype(int),
            "exterior_coords": d["exterior_coords"],
            "mass_kg":         float(d["mass_kg"][0]),
        })

    # Sort by nc (champion = -1 → goes first)
    sweeps.sort(key=lambda s: (s["nc"] if s["nc"] >= 0 else -999))
    return sweeps


def _rpm_at_sf(rpms: np.ndarray, sf_vals: np.ndarray, target_sf: float) -> float:
    """Interpolate to find RPM where SF first crosses target_sf from above."""
    valid = (sf_vals > 0) & np.isfinite(sf_vals)
    if not valid.any():
        return float(rpms[-1])
    r = rpms[valid]
    s = sf_vals[valid]
    above = s >= target_sf
    if above.all():
        return float(r[-1])
    if not above.any():
        return float(r[0])
    i = int(np.where(~above)[0][0])
    if i == 0:
        return float(r[0])
    # Linear interpolation
    t = (target_sf - s[i - 1]) / (s[i] - s[i - 1])
    return float(r[i - 1] + t * (r[i] - r[i - 1]))


# ---------------------------------------------------------------------------
# Figure 1: stacked stress snapshots
# ---------------------------------------------------------------------------

def plot_stacked_snapshots(
    sweeps: list[dict],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """One row per weapon, 5 cols = weapon outline + 4 stress snapshots."""
    n_rows = len(sweeps)
    n_cols = 5   # col 0 = weapon outline, cols 1-4 = stress snapshots (skip 1 if needed)

    fig = plt.figure(figsize=(20, 4.5 * n_rows + 1.0))
    gs  = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        wspace=0.04,
        hspace=0.50,
        left=0.06, right=0.98, top=0.93, bottom=0.04,
    )

    colors = plt.cm.tab10(np.linspace(0, 0.8, max(n_rows, 1)))

    for row_idx, sw in enumerate(sweeps):
        nc     = sw["nc"]
        label  = sw["label"]
        color  = colors[row_idx]
        snaps  = sw["snap_stresses"]         # (n_snaps, M_elem)
        srpms  = sw["snap_rpms"]             # (n_snaps,)
        nodes  = sw["snap_nodes"]            # (N_nodes, 2)
        elems  = sw["snap_elements"]         # (M_elem, 3)
        ext    = sw["exterior_coords"]       # (M, 2)

        has_mesh = (len(nodes) > 0 and len(elems) > 0)

        # Shared stress colour scale: yield_strength / 1.5 (stress at SF=1.5)
        vmax_stress = YIELD_MPa / 1.5

        # ── Col 0: weapon profile (no stress, just outline fill) ──────────
        ax0 = fig.add_subplot(gs[row_idx, 0])
        if has_mesh:
            # Show mean stress field from lowest-RPM snapshot as context
            sm = snaps[0]
            valid_mask = np.isfinite(sm)
            if valid_mask.any():
                sm_clean = np.where(valid_mask, sm, 0.0)
                try:
                    tri  = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems)
                    nv   = elements_to_nodes(sm_clean, elems, len(nodes))
                    ax0.tricontourf(tri, nv, levels=20, cmap="plasma",
                                    vmin=0, vmax=vmax_stress, alpha=0.7)
                except Exception:
                    pass
        # Weapon outline
        ax0.plot(ext[:, 0], ext[:, 1], "-", color=color, lw=2.0, zorder=5)
        ax0.set_aspect("equal")
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_ylabel(f"{label}\nmass={sw['mass_kg']:.3f}kg",
                       fontsize=10, fontweight="bold", color=color, labelpad=8)
        ax0.set_title("Profile\n(low RPM stress)", fontsize=8.5)

        # ── Cols 1-4: stress snapshots ────────────────────────────────────
        # Choose up to 4 non-zero snapshots
        n_snaps    = len(srpms)
        n_show     = min(4, n_snaps)
        show_idxs  = np.round(np.linspace(0, n_snaps - 1, n_show)).astype(int)

        # Get SF values at snapshot RPMs
        snap_sfs = []
        for srpm in srpms:
            # Find closest RPM in full sweep
            idx = int(np.argmin(np.abs(sw["rpms"] - srpm)))
            snap_sfs.append(float(sw["sf_vals"][idx]))

        snap_axes = []
        for col_offset in range(1, 5):
            ax = fig.add_subplot(gs[row_idx, col_offset])
            snap_axes.append(ax)
            si = col_offset - 1   # index into show_idxs

            if si >= n_show or not has_mesh:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, color="#aaaaaa")
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            snap_idx = show_idxs[si]
            rpm_i    = int(srpms[snap_idx])
            sf_i     = snap_sfs[snap_idx]
            sm       = snaps[snap_idx]

            valid = np.isfinite(sm)
            if not valid.any():
                ax.text(0.5, 0.5, "FEA\nfailed", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#cc3333")
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{rpm_i} RPM\nSF={sf_i:.2f}", fontsize=8.5)
                continue

            sm_clean = np.where(valid, sm, 0.0)
            try:
                tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems)
                nv  = elements_to_nodes(sm_clean, elems, len(nodes))
                cf  = ax.tricontourf(tri, nv, levels=20, cmap="plasma",
                                     vmin=0, vmax=vmax_stress)
                if col_offset == 4:
                    cb = plt.colorbar(cf, ax=snap_axes, shrink=0.85, pad=0.02)
                    cb.ax.tick_params(labelsize=6)
                    cb.set_label("σ_vM (MPa)", fontsize=7)
            except Exception:
                ax.text(0.5, 0.5, "mesh\nerr", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#cc3333")

            # Weapon outline overlay
            ax.plot(ext[:, 0], ext[:, 1], "w-", lw=0.9, alpha=0.8, zorder=5)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])

            # Colour-code title by safety factor
            sf_color = "#cc3333" if sf_i < 1.5 else ("#bb8800" if sf_i < 2.0 else "#226622")
            ax.set_title(f"{rpm_i} RPM\nSF={sf_i:.2f}", fontsize=8.5, color=sf_color)

    fig.suptitle(
        "FEA Stress Snapshots at Representative RPM Values\n"
        "(one row per archetype weapon; σ_vM coloured by plasma, vmax = σ_yield / 1.5)",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure_speed_pareto] stacked snapshots → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: RPM vs. impact force Pareto
# ---------------------------------------------------------------------------

def plot_speed_pareto(
    sweeps: list[dict],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Pareto-style: RPM (x) vs. impact force (y), two lines per weapon."""
    fig, ax = plt.subplots(figsize=(11, 7))

    colors = plt.cm.tab10(np.linspace(0, 0.8, max(len(sweeps), 1)))

    for sw, color in zip(sweeps, colors):
        rpms    = sw["rpms"]
        sf_vals = sw["sf_vals"]
        f_imp   = sw["f_impact_n"]
        label   = sw["label"]
        mass_kg = sw["mass_kg"]

        # Find RPM thresholds
        rpm_fail = _rpm_at_sf(rpms, sf_vals, 1.0)
        rpm_fos2 = _rpm_at_sf(rpms, sf_vals, 2.0)

        # Convert N → kN for readability
        f_kn = f_imp / 1000.0

        # Mask valid (positive SF) region
        valid = (sf_vals > 0) & np.isfinite(sf_vals) & (f_imp > 0)
        if not valid.any():
            continue

        r_v = rpms[valid]
        f_v = f_kn[valid]

        # Dotted line: 0 → RPM_FOS2 (safe operating envelope, FOS=2)
        fos2_mask = r_v <= rpm_fos2
        if fos2_mask.any():
            ax.plot(r_v[fos2_mask], f_v[fos2_mask],
                    "--", color=color, lw=2.5, alpha=0.75)

        # Solid line: 0 → RPM_fail (full structural range to failure)
        fail_mask = r_v <= rpm_fail
        if fail_mask.any():
            ax.plot(r_v[fail_mask], f_v[fail_mask],
                    "-", color=color, lw=2.5,
                    label=f"{label}  m={mass_kg:.3f}kg")

        # Vertical markers
        if rpm_fos2 < rpm_fail * 0.99:
            f_at_fos2 = float(0.5 * sw["mass_kg"]
                              * (2 * np.pi * rpm_fos2 / 60) ** 2
                              * sw["snap_nodes"].shape[0] * 0)   # recompute cleanly
            # Use linear interp for exact force value at rpm_fos2
            f_at_fos2 = float(np.interp(rpm_fos2, rpms[valid], f_v))
            ax.axvline(rpm_fos2, color=color, ls=":", lw=1.0, alpha=0.55)
            ax.annotate(
                f"FOS=2\n{rpm_fos2:.0f}",
                xy=(rpm_fos2, f_at_fos2),
                xytext=(rpm_fos2 + 80, f_at_fos2 * 1.05),
                fontsize=7, color=color, alpha=0.85,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.6),
            )

        f_at_fail = float(np.interp(rpm_fail, rpms[valid], f_v))
        ax.plot(rpm_fail, f_at_fail, "x", color=color, ms=10, mew=2.0, zorder=10)

    # Shared formatting
    ax.set_xlabel("Spin speed (RPM)", fontsize=11)
    ax.set_ylabel("Impact force F_contact (kN)\n"
                  r"$F = \mu \, v_{\rm tip} / t_{\rm contact}$"
                  "\n(reduced-mass impulse model)",
                  fontsize=10)
    ax.set_title(
        "Operating Regime: Maximum RPM vs. Contact Impact Force\n"
        "Solid line = range to structural failure (SF=1)  |  "
        "Dotted line = safe operating envelope (FOS=2)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Custom legend: weapon lines + style explanation
    handles, labels = ax.get_legend_handles_labels()
    style_handles = [
        Line2D([0], [0], color="gray", lw=2.5, ls="-",  label="→ failure point (SF=1)"),
        Line2D([0], [0], color="gray", lw=2.5, ls="--", label="→ FOS=2 limit"),
        Line2D([0], [0], color="gray", lw=0,   ls="",   marker="x", ms=9, mew=2,
               label="× failure RPM"),
    ]
    ax.legend(handles=handles + style_handles,
              labels=labels + [h.get_label() for h in style_handles],
              fontsize=9, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure_speed_pareto] pareto → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate speed sweep stacked snapshot + Pareto figures")
    parser.add_argument("--sweep-dir", default=str(SWEEP_DIR),
                        help="Directory containing *_sweep.npz files")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"[error] sweep directory not found: {sweep_dir}")
        print("  Run  python scripts/fea_archetype_sweep.py  first.")
        sys.exit(1)

    sweeps = load_sweeps(sweep_dir)
    if not sweeps:
        print(f"[error] No *_sweep.npz files found in {sweep_dir}")
        sys.exit(1)

    print(f"[load] found {len(sweeps)} archetype sweep(s): "
          f"{[s['label'] for s in sweeps]}")

    snap_path  = OUT_DIR / "figure_speed_stacked.png"
    pareto_path = OUT_DIR / "figure_speed_pareto.png"

    plot_stacked_snapshots(sweeps, snap_path,  dpi=args.dpi)
    plot_speed_pareto(sweeps,       pareto_path, dpi=args.dpi)

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
