#!/usr/bin/env python3
"""POD mode table + subspace similarity figures.

Two figures:
  1. pod_contact_modes_table.png  — grid: rows = nc groups, cols = [best design, mode 1-3]
  2. pod_subspace_similarity.png  — pairwise geodesic distance heatmap + angle box plots

Usage
-----
    python docs/figures/figure_pod_contact_table.py
    python docs/figures/figure_pod_contact_table.py --pod-dir outputs/pod_stratified \\
        --nc 1,2,3 --dpi 150
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

POD_DIR = ROOT / "outputs" / "pod_stratified"
DB_DIR  = ROOT / "outputs" / "fea_database"
OUT_DIR = Path(__file__).parent


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


def load_pod_results(pod_dir: Path, nc_values: list[int]) -> dict:
    """Load nc{n}_basis.npz files + summary.json."""
    results = {}
    for nc in nc_values:
        p = pod_dir / f"nc{nc}_basis.npz"
        if not p.exists():
            print(f"[warn] {p} not found — skipping nc={nc}")
            continue
        d = np.load(p, allow_pickle=True)
        results[nc] = {
            "basis":            d["basis"],            # (M_ref, k_99)
            "singular_values":  d["singular_values"],  # (k_99,)
            "mean_stress":      d["mean_stress"],       # (M_ref,)
            "variance_explained": d["variance_explained"],
            "k_99":             int(d["k_99"][0]),
            "k_95":             int(d["k_95"][0]),
            "N":                int(d["N"][0]),
            "best_exterior_coords": d["best_exterior_coords"],  # (M, 2)
            "best_vm_stresses":     d["best_vm_stresses"],      # (M_ref,)
            "safety_factors":       d["safety_factors"],        # (N,)
            "best_design_idx":      int(d["best_design_idx"][0]),
        }

    summary_path = pod_dir / "summary.json"
    similarity = None
    if summary_path.exists():
        with open(summary_path) as f:
            raw = json.load(f)
        summary_groups = raw.get("groups", {})
        similarity     = raw.get("similarity", {})
        for nc in results:
            g = summary_groups.get(str(nc), {})
            results[nc]["var_3modes"]      = g.get("var_3modes", 0.0)
            results[nc]["proj_rmse_k3"]    = g.get("proj_rmse_k3_mpa", float("nan"))
            results[nc]["best_design_file"] = g.get("best_design_file", "")
    else:
        print("[warn] summary.json not found — annotations will be incomplete")
        for nc in results:
            sv = results[nc]["singular_values"]
            s2 = sv ** 2
            cumvar = np.cumsum(s2) / s2.sum()
            results[nc]["var_3modes"]   = float(cumvar[min(2, len(cumvar)-1)])
            results[nc]["proj_rmse_k3"] = float("nan")

    return results, similarity


# ---------------------------------------------------------------------------
# Figure 1: Mode table
# ---------------------------------------------------------------------------

def plot_mode_table(
    pod_results: dict,
    ref_nodes: np.ndarray,
    ref_elements: np.ndarray,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Grid figure: rows = nc groups, cols = [best design, mode 1, mode 2, mode 3]."""
    nc_list = sorted(pod_results.keys())
    n_rows  = len(nc_list)
    n_cols  = 4

    fig = plt.figure(figsize=(18, 4.5 * n_rows + 1.0))
    gs  = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        wspace=0.06,
        hspace=0.45,
        left=0.06, right=0.97, top=0.93, bottom=0.05,
    )

    # Build the shared triangulation once
    tri = mtri.Triangulation(ref_nodes[:, 0], ref_nodes[:, 1], ref_elements)
    n_nodes = len(ref_nodes)

    # Qualitative color for each row's stress map (plasma) and row label
    row_colors = plt.cm.tab10(np.linspace(0, 0.4, n_rows))

    for row_idx, nc in enumerate(nc_list):
        pr = pod_results[nc]
        var_3  = pr.get("var_3modes", float("nan")) * 100
        rmse3  = pr.get("proj_rmse_k3", float("nan"))
        k99    = pr["k_99"]
        N      = pr["N"]
        sf_vals = pr["safety_factors"]
        finite  = sf_vals[np.isfinite(sf_vals)]
        best_sf = float(finite.max()) if len(finite) else float("nan")

        # ── Column 0: Best design + actual stress field ────────────────────
        ax0 = fig.add_subplot(gs[row_idx, 0])

        vm = pr["best_vm_stresses"]                    # (M_ref,)
        nv = elements_to_nodes(vm, ref_elements, n_nodes)
        vmax_stress = float(np.percentile(vm, 95))
        vmax_stress = max(vmax_stress, 1.0)

        cf0 = ax0.tricontourf(tri, nv, levels=20, cmap="plasma",
                              vmin=0, vmax=vmax_stress)
        cb0 = plt.colorbar(cf0, ax=ax0, shrink=0.85, pad=0.02)
        cb0.ax.tick_params(labelsize=6)
        cb0.set_label("σ_vM (MPa)", fontsize=7)

        # Weapon outline overlay
        coords = pr["best_exterior_coords"]
        ax0.plot(coords[:, 0], coords[:, 1], "w-", lw=1.4, alpha=0.9, zorder=5)

        ax0.set_aspect("equal")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_ylabel(f"nc = {nc}", fontsize=12, fontweight="bold", labelpad=8,
                       color=row_colors[row_idx])
        ax0.set_title(f"Best design\n(SF={best_sf:.2f})", fontsize=9)

        # Annotation strip below col 0
        annotation = (
            f"N={N}  |  k@99%={k99}  |  3-mode var: {var_3:.1f}%  |  "
            f"proj. RMSE (k=3): {rmse3:.2f} MPa"
        )
        ax0.text(0.0, -0.14, annotation, transform=ax0.transAxes,
                 fontsize=7.5, ha="left", va="top", color="#333333",
                 style="italic")

        # ── Columns 1-3: POD modes ─────────────────────────────────────────
        basis = pr["basis"]                 # (M_ref, k_99)
        sv    = pr["singular_values"]
        s2    = sv ** 2
        cumvar_modes = np.cumsum(s2) / s2.sum()

        mode_axes = []
        for col_idx in range(1, 4):
            mode_i = col_idx - 1   # 0-based mode index
            ax = fig.add_subplot(gs[row_idx, col_idx])
            mode_axes.append(ax)

            if mode_i >= basis.shape[1]:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, color="gray")
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            mode_elem = basis[:, mode_i]               # (M_ref,)
            mode_node = elements_to_nodes(mode_elem, ref_elements, n_nodes)

            vmax_m = float(np.abs(mode_node).max())
            vmax_m = max(vmax_m, 1e-9)

            cf = ax.tricontourf(tri, mode_node, levels=20, cmap="RdBu_r",
                                vmin=-vmax_m, vmax=vmax_m)

            pct_var = float(s2[mode_i] / s2.sum()) * 100
            ax.set_title(f"Mode {col_idx}  ({pct_var:.1f}% var.)", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            # Shared colorbar on the last mode column of each row
            if col_idx == 3:
                cb = plt.colorbar(cf, ax=mode_axes, shrink=0.85, pad=0.02)
                cb.ax.tick_params(labelsize=6)
                cb.set_label("Mode amplitude (norm.)", fontsize=7)

    fig.suptitle(
        "Stratified POD Analysis by Contact Count\n"
        "(B-spline weapon designs, FEA database N=2500)",
        fontsize=13, fontweight="bold",
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure_pod_contact_table] mode table → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Subspace similarity
# ---------------------------------------------------------------------------

def plot_subspace_similarity(
    similarity: dict,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Two-panel: geodesic distance heatmap + principal angle box plots."""
    if similarity is None:
        print("[warn] no similarity data — skipping subspace similarity figure")
        return

    labels   = similarity.get("nc_labels", [])
    geo_mat  = np.array(similarity.get("geodesic_distance", []))
    pairs    = similarity.get("pairs", {})
    k_used   = similarity.get("k_used", 10)

    if len(labels) < 2 or geo_mat.size == 0:
        print("[warn] insufficient groups for similarity plot")
        return

    n = len(labels)

    fig = plt.figure(figsize=(10, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.8, 1.2],
                            wspace=0.4, left=0.08, right=0.97, top=0.87, bottom=0.12)
    ax_heat  = fig.add_subplot(gs[0])
    ax_box   = fig.add_subplot(gs[1])

    # ── Panel A: geodesic distance heatmap ────────────────────────────────
    im = ax_heat.imshow(geo_mat, cmap="YlOrRd", vmin=0,
                        vmax=geo_mat.max() * 1.05 if geo_mat.max() > 0 else 1.0,
                        aspect="auto")
    plt.colorbar(im, ax=ax_heat, shrink=0.8, label="Geodesic distance (rad)")

    tick_labels = [f"nc={nc}" for nc in labels]
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(tick_labels, fontsize=9)
    ax_heat.set_yticklabels(tick_labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = geo_mat[i, j]
            color = "white" if val > geo_mat.max() * 0.6 else "black"
            ax_heat.text(j, i, f"{val:.3f}", ha="center", va="center",
                         fontsize=11, fontweight="bold", color=color)

    ax_heat.set_title(
        f"Geodesic subspace distance\n"
        f"(Grassmannian, k={k_used} modes)",
        fontsize=10,
    )

    max_theoretical = float(np.pi / 2 * np.sqrt(k_used))
    ax_heat.text(0.01, -0.08,
                 f"Range: 0 (identical) → {max_theoretical:.2f} (maximally distinct)",
                 transform=ax_heat.transAxes, fontsize=7.5, color="#555555", style="italic")

    # ── Panel B: principal angle box plots ────────────────────────────────
    box_data   = []
    box_labels = []
    for key, pdata in sorted(pairs.items()):
        angs_deg = np.degrees(np.array(pdata["principal_angles_rad"]))
        box_data.append(angs_deg)
        # e.g. "nc1_vs_nc2" → "nc=1 vs nc=2"
        pretty = key.replace("nc", "nc=").replace("_vs_", " vs ")
        box_labels.append(pretty)

    if box_data:
        bp = ax_box.boxplot(box_data, vert=False, patch_artist=True,
                            medianprops=dict(color="black", lw=1.5),
                            boxprops=dict(facecolor="#AED6F1", alpha=0.7))
        ax_box.axvline(45, color="gray", ls="--", lw=1.2, alpha=0.7, label="45° reference")
        ax_box.set_yticks(range(1, len(box_labels) + 1))
        ax_box.set_yticklabels(box_labels, fontsize=9)
        ax_box.set_xlabel("Principal angle (degrees)", fontsize=9)
        ax_box.set_title(f"Principal angle distribution\n(k={k_used} modes per pair)",
                         fontsize=10)
        ax_box.legend(fontsize=8, loc="lower right")
        ax_box.grid(alpha=0.3, axis="x")
        ax_box.set_xlim(0, 92)

    fig.suptitle(
        "Subspace Similarity between POD Bases\n"
        "(B-spline weapon designs grouped by contact count)",
        fontsize=12, fontweight="bold",
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure_pod_contact_table] similarity → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate POD contact table figures")
    parser.add_argument("--pod-dir", default=str(POD_DIR), help="Pod stratified outputs dir")
    parser.add_argument("--db-dir",  default=str(DB_DIR),  help="FEA database dir (for ref mesh)")
    parser.add_argument("--nc",      default="1,2,3",      help="n_contacts groups to plot")
    parser.add_argument("--dpi",     type=int, default=150, help="Output DPI")
    args = parser.parse_args()

    pod_dir  = Path(args.pod_dir)
    db_dir   = Path(args.db_dir)
    nc_values = [int(x.strip()) for x in args.nc.split(",")]

    # Load reference mesh
    ref_mesh_path = db_dir / "ref_mesh.npz"
    if not ref_mesh_path.exists():
        print(f"[error] ref_mesh.npz not found at {ref_mesh_path}")
        sys.exit(1)
    ref_data = np.load(ref_mesh_path)
    ref_nodes    = ref_data["nodes"]     # (691, 2)
    ref_elements = ref_data["elements"]  # (1173, 3)
    print(f"[load] reference mesh: {len(ref_nodes)} nodes, {len(ref_elements)} elements")

    # Load POD results
    pod_results, similarity = load_pod_results(pod_dir, nc_values)
    if not pod_results:
        print("[error] No POD results found — run pod_contact_analysis.py first")
        sys.exit(1)

    print(f"[load] loaded POD bases for nc ∈ {sorted(pod_results.keys())}")

    # Figure 1: mode table
    table_path = OUT_DIR / "pod_contact_modes_table.png"
    plot_mode_table(pod_results, ref_nodes, ref_elements, table_path, dpi=args.dpi)

    # Figure 2: subspace similarity
    sim_path = OUT_DIR / "pod_subspace_similarity.png"
    plot_subspace_similarity(similarity, sim_path, dpi=args.dpi)

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
