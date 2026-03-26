#!/usr/bin/env python3
"""Stratified POD comparison: Archimedean spiral weapon, n_starts = 1..4.

Uses the spiral_weapon parametrisation (same as spiral_preview/ images):
  - Outer profile: Archimedean spiral (n_starts arcs, each r(θ)=R_min+pitch*frac)
  - Interior: parametric ribs (t_rim, t_hub, n_supports, t_support, r_fillet)
  - 7 parameters total: [spiral_pitch, n_starts, t_rim, t_hub, n_supports, t_support, r_fillet]

No B-spline, no control points, no harmonic series.

Rows (5):  all  |  n_starts=1  |  n_starts=2  |  n_starts=3  |  n_starts=4
Cols (5):  FEA stress field  |  POD mode 1  |  mode 2  |  mode 3  |  mode 4

For each group with fixed n_starts, the remaining 6 free parameters
(spiral_pitch + 5 cutout params) are sampled with Sobol.

Usage
-----
    python docs/figures/figure_functional_pod_table.py
    python docs/figures/figure_functional_pod_table.py --n-per-group 150 --yes
    python docs/figures/figure_functional_pod_table.py --force-regen --yes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.tri as mtri
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from weapon_designer.config import (
    WeaponConfig, Material, Mounting, Envelope, OptimizationParams,
)
from weapon_designer.geometry import assemble_weapon
from weapon_designer.fea import fea_stress_analysis_with_mesh
from weapon_designer.spiral_weapon import (
    build_spiral_weapon,
    get_spiral_weapon_bounds,
)
from weapon_designer.spiral_cutouts import get_spiral_cutout_bounds
from weapon_designer.spiral_contact import (
    analyse_contacts,
    contact_forces as spiral_contact_forces,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = ROOT / "outputs" / "spiral_pod_cache_v2"
REF_MESH  = ROOT / "outputs" / "fea_database" / "ref_mesh.npz"
OUT_DIR   = ROOT / "docs" / "figures"
N_PER_GROUP_DEFAULT = 150
DPI_DEFAULT = 150
MESH_SPACING = 10.0  # mm, coarse mesh for speed


def _make_cfg() -> WeaponConfig:
    return WeaponConfig(
        material=Material(
            name="S7_Tool_Steel", density_kg_m3=7750,
            yield_strength_mpa=1600, hardness_hrc=56,
        ),
        weapon_style="disk",
        sheet_thickness_mm=6,
        weight_budget_kg=0.5,
        rpm=10000,
        mounting=Mounting(
            bore_diameter_mm=12.0, bolt_circle_diameter_mm=25,
            num_bolts=3, bolt_hole_diameter_mm=4.0,
        ),
        envelope=Envelope(max_radius_mm=100),
        optimization=OptimizationParams(
            fea_coarse_spacing_mm=MESH_SPACING,
        ),
    )


# ---------------------------------------------------------------------------
# Sobol sampling
# ---------------------------------------------------------------------------

def _sobol(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Return (n, d) Sobol sample in [0,1]^d."""
    from scipy.stats.qmc import Sobol
    m = 1
    while 2**m < n:
        m += 1
    sampler = Sobol(d=d, scramble=True, seed=seed)
    return sampler.random_base2(m)[:n]


# ---------------------------------------------------------------------------
# Reference mesh loader
# ---------------------------------------------------------------------------

def _load_ref_mesh(path: Path):
    """Load the fixed reference mesh (nodes, elements)."""
    data = np.load(path)
    nodes    = data["nodes"]    # (N_nodes, 2)
    elements = data["elements"] # (N_elem, 3)
    return nodes, elements


# ---------------------------------------------------------------------------
# FEA for one spiral design, interpolated to reference mesh
# ---------------------------------------------------------------------------

def _run_design(
    params: np.ndarray,
    cfg: WeaponConfig,
    ref_nodes: np.ndarray,
    ref_elements: np.ndarray,
) -> tuple[np.ndarray | None, float, float]:
    """
    Build spiral weapon, run FEA, interpolate vm_stresses to reference mesh.

    Returns
    -------
    vm_ref   : (N_ref_elem,) stress array on reference mesh, or None on failure
    sf       : safety factor (inf if stress≈0)
    peak_mpa : peak von-Mises stress
    """
    weapon = build_spiral_weapon(params, cfg)
    if weapon is None or weapon.is_empty:
        return None, 0.0, 0.0

    assembled = assemble_weapon(weapon, cfg.mounting)
    if assembled is None or assembled.is_empty:
        return None, 0.0, 0.0

    # Contact forces: opponent impact (reduced-mass impulse model)
    # m_eff = I / r_contact²,  mu = m_eff * m_opp / (m_eff + m_opp)
    # F = mu * v_tip / t_contact  (F scales with RPM via v_tip = omega * r_contact)
    try:
        from weapon_designer.physics import mass_moi_kg_mm2
        M_OPP_KG   = 0.68    # ~1.5 lb featherweight opponent
        T_CONT_S   = 1e-3    # ~1 ms steel-on-steel contact
        V_APPR_MS  = 3.0     # approach speed for contact geometry
        omega = 2.0 * 3.141592653589793 * cfg.rpm / 60.0
        contacts, _ = analyse_contacts(
            weapon, n_spirals=6, v_ms=V_APPR_MS, rpm=cfg.rpm, n_eval=360,
        )
        if contacts:
            r_c_m  = float(sum(c.r_contact for c in contacts) / len(contacts)) * 1e-3
            I_kg_m2 = mass_moi_kg_mm2(
                assembled, cfg.sheet_thickness_mm, cfg.material.density_kg_m3,
            ) * 1e-6
            m_eff  = I_kg_m2 / max(r_c_m ** 2, 1e-6)
            mu     = m_eff * M_OPP_KG / (m_eff + M_OPP_KG)
            f_mag  = mu * (omega * r_c_m) / T_CONT_S
            fea_forces = spiral_contact_forces(contacts, f_mag, scale_by_angle=True)
        else:
            fea_forces = None
    except Exception:
        fea_forces = None

    try:
        result = fea_stress_analysis_with_mesh(
            assembled,
            rpm=cfg.rpm,
            density_kg_m3=cfg.material.density_kg_m3,
            thickness_mm=cfg.sheet_thickness_mm,
            yield_strength_mpa=cfg.material.yield_strength_mpa,
            bore_diameter_mm=cfg.mounting.bore_diameter_mm,
            mesh_spacing=MESH_SPACING,
            contact_forces=fea_forces,
        )
    except Exception:
        return None, 0.0, 0.0

    if result is None:
        return None, 0.0, 0.0

    sf         = result["safety_factor"]
    peak_mpa   = result["peak_stress_mpa"]
    vm_stresses = result["vm_stresses"]
    nodes      = result["nodes"]
    elements   = result["elements"]

    if vm_stresses is None or len(vm_stresses) == 0:
        return None, float(sf), float(peak_mpa)

    # Centroids of design mesh elements
    src_cents = nodes[elements].mean(axis=1)  # (M, 2)

    # Centroids of reference mesh elements
    ref_cents = ref_nodes[ref_elements].mean(axis=1)  # (N_ref, 2)

    # Interpolate
    interp = LinearNDInterpolator(src_cents, vm_stresses)
    vm_ref = interp(ref_cents)

    # Fill NaN with nearest-neighbor
    nan_mask = np.isnan(vm_ref)
    if nan_mask.any():
        near = NearestNDInterpolator(src_cents, vm_stresses)
        vm_ref[nan_mask] = near(ref_cents[nan_mask])

    return vm_ref, float(sf), float(peak_mpa)


# ---------------------------------------------------------------------------
# Generate & cache one group
# ---------------------------------------------------------------------------

def _generate_group(
    n_starts_fixed: int | None,
    cfg: WeaponConfig,
    ref_nodes: np.ndarray,
    ref_elements: np.ndarray,
    n_per_group: int,
    cache_path: Path,
) -> dict:
    """
    Generate n_per_group designs for a fixed n_starts (or all n_starts if None).

    Returns dict with keys: stresses (N,M), sf_vals (N,), params (N,7), n_starts_fixed
    """
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return {
            "stresses":      data["stresses"],
            "sf_vals":       data["sf_vals"],
            "params":        data["params"],
            "n_starts_fixed": int(data["n_starts_fixed"]) if "n_starts_fixed" in data else None,
        }

    # Parameter bounds: [spiral_pitch, n_starts, t_rim, t_hub, n_supports, t_support, r_fillet]
    all_bounds = get_spiral_weapon_bounds(cfg)  # 7 bounds
    lo = np.array([b[0] for b in all_bounds])
    hi = np.array([b[1] for b in all_bounds])

    # Double the pitch range for more pronounced spiral geometry
    R_outer = float(cfg.envelope.max_radius_mm)
    R_bore  = float(cfg.mounting.bore_diameter_mm) / 2.0
    max_safe_pitch = R_outer - R_bore * 2.0 - 2.0
    lo[0] = lo[0] * 2.0
    hi[0] = min(hi[0] * 2.0, max_safe_pitch)

    # Double the rim thickness range for a larger contact region
    lo[2] = lo[2] * 2.0
    hi[2] = min(hi[2] * 2.0, R_outer * 0.30)

    # For fixed n_starts groups: sample 6 free params (exclude n_starts at index 1)
    free_idx = [0, 2, 3, 4, 5, 6]  # pitch + 5 cutout params
    n_free = len(free_idx)          # 6

    n_target = n_per_group
    sobol_raw = _sobol(n_target, n_free)
    lo_free = lo[free_idx]
    hi_free = hi[free_idx]
    free_samples = lo_free + sobol_raw * (hi_free - lo_free)  # (n_target, 6)

    # Assemble full param vectors
    if n_starts_fixed is not None:
        # Fix n_starts; cycle through all n_starts values for "all" group later
        n_starts_col = np.full((n_target, 1), float(n_starts_fixed))
    else:
        # "all" group: cycle n_starts = 1,2,3,4 evenly
        ns_vals = np.tile([1.0, 2.0, 3.0, 4.0], n_target // 4 + 1)[:n_target]
        n_starts_col = ns_vals.reshape(-1, 1)

    params_matrix = np.empty((n_target, 7))
    params_matrix[:, 0]    = free_samples[:, 0]   # spiral_pitch
    params_matrix[:, 1]    = n_starts_col[:, 0]   # n_starts
    params_matrix[:, 2:7]  = free_samples[:, 1:]  # cutout params

    stresses_list = []
    sf_list       = []
    valid_params  = []

    t0 = time.time()
    fail = 0
    for i, p in enumerate(params_matrix):
        vm, sf, _ = _run_design(p, cfg, ref_nodes, ref_elements)
        if vm is None:
            fail += 1
            continue
        stresses_list.append(vm)
        sf_list.append(sf)
        valid_params.append(p)
        done = len(stresses_list)
        if done % 30 == 0:
            print(f"    [{done:4d}/{n_target}]  valid={done}  fail={fail}  "
                  f"({time.time()-t0:.1f}s)")

    stresses = np.array(stresses_list, dtype=np.float32)
    sf_vals  = np.array(sf_list)
    params_arr = np.array(valid_params)
    label = n_starts_fixed if n_starts_fixed is not None else -1

    np.savez_compressed(
        cache_path,
        stresses=stresses,
        sf_vals=sf_vals,
        params=params_arr,
        n_starts_fixed=np.array(label),
    )
    return {
        "stresses": stresses,
        "sf_vals":  sf_vals,
        "params":   params_arr,
        "n_starts_fixed": n_starts_fixed,
    }


# ---------------------------------------------------------------------------
# POD analysis
# ---------------------------------------------------------------------------

def compute_pod(stresses: np.ndarray, sf_vals: np.ndarray) -> dict:
    """
    Centered SVD of stress matrix.

    Returns dict with: mean_stress, basis, singular_values, variance_explained,
                       k_99, k_95, best_idx, rmse_k4
    """
    X = stresses.astype(np.float64)
    mean_s = X.mean(axis=0)
    X_c = X - mean_s

    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)

    var = (S ** 2) / (S ** 2).sum()
    cum_var = np.cumsum(var)
    k_99 = int(np.searchsorted(cum_var, 0.99)) + 1
    k_95 = int(np.searchsorted(cum_var, 0.95)) + 1
    k4_var = float(cum_var[min(3, len(cum_var) - 1)] * 100)

    # Reconstruction error with 4 modes
    X_rec4 = U[:, :4] @ np.diag(S[:4]) @ Vt[:4, :] + mean_s
    rmse_k4 = float(np.sqrt(np.mean((X - X_rec4) ** 2)))

    # Best design: highest finite SF
    finite_mask = np.isfinite(sf_vals)
    if finite_mask.any():
        best_idx = int(np.where(finite_mask)[0][np.argmax(sf_vals[finite_mask])])
    else:
        best_idx = 0

    return {
        "mean_stress":        mean_s,
        "basis":              Vt.T,        # (M_ref, k) right singular vectors
        "singular_values":    S,
        "variance_explained": cum_var,
        "k_99":               k_99,
        "k_95":               k_95,
        "k4_var":             k4_var,
        "best_idx":           best_idx,
        "rmse_k4":            rmse_k4,
    }


# ---------------------------------------------------------------------------
# Element→node averaging helper
# ---------------------------------------------------------------------------

def elements_to_nodes(elem_vals: np.ndarray, elements: np.ndarray, n_nodes: int) -> np.ndarray:
    node_vals = np.zeros(n_nodes)
    node_cnt  = np.zeros(n_nodes)
    np.add.at(node_vals, elements.ravel(), np.repeat(elem_vals, 3))
    np.add.at(node_cnt,  elements.ravel(), 1)
    return node_vals / np.maximum(node_cnt, 1)


# ---------------------------------------------------------------------------
# Raw FEA re-run for best design (figure time only, not cached)
# ---------------------------------------------------------------------------

def _raw_fea(params: np.ndarray, cfg: WeaponConfig) -> dict | None:
    """Re-run FEA on weapon's own mesh for the best design in a group."""
    weapon = build_spiral_weapon(params, cfg)
    if weapon is None:
        return None
    assembled = assemble_weapon(weapon, cfg.mounting)
    if assembled is None:
        return None
    try:
        result = fea_stress_analysis_with_mesh(
            assembled,
            rpm=cfg.rpm,
            density_kg_m3=cfg.material.density_kg_m3,
            thickness_mm=cfg.sheet_thickness_mm,
            yield_strength_mpa=cfg.material.yield_strength_mpa,
            bore_diameter_mm=cfg.mounting.bore_diameter_mm,
            mesh_spacing=MESH_SPACING,
        )
        return result if result else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(
    groups: list[dict],
    pod_results: list[dict],
    ref_nodes: np.ndarray,
    ref_elements: np.ndarray,
    cfg: WeaponConfig,
    out_path: Path,
    dpi: int,
):
    n_nodes = ref_nodes.shape[0]
    tri_ref = mtri.Triangulation(ref_nodes[:, 0], ref_nodes[:, 1], ref_elements)

    row_labels = ["n_starts=1", "n_starts=2", "n_starts=3", "n_starts=4"]
    col_titles = ["weapon shape\n(best design)",
                  "FEA — raw mesh",
                  "FEA → ref mesh\n(interp.)",
                  "POD mode 1", "POD mode 2", "POD mode 3", "POD mode 4"]
    n_rows, n_cols = 4, 7

    # Leave right margin for shared POD colorbar
    fig = plt.figure(figsize=(28, 13))

    outer_gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[0.04, 1.0],
        hspace=0.01,
        left=0.06, right=0.94,
        top=0.95, bottom=0.05,
    )
    header_gs = gridspec.GridSpecFromSubplotSpec(
        1, n_cols, subplot_spec=outer_gs[0], wspace=0.05,
    )
    data_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=outer_gs[1],
        wspace=0.05, hspace=0.35,
    )

    # Column headers
    for c, title in enumerate(col_titles):
        ax_h = fig.add_subplot(header_gs[0, c])
        ax_h.axis("off")
        ax_h.text(0.5, 0.5, title, ha="center", va="center",
                  fontsize=9, fontweight="bold",
                  transform=ax_h.transAxes)

    # Shared POD colorbar axes (far right, outside data_gs)
    cbar_ax = fig.add_axes([0.955, 0.10, 0.012, 0.75])

    # Dummy mappable for shared RdBu_r colorbar (modes normalised to [-1, 1])
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    pod_sm = mcm.ScalarMappable(
        cmap="RdBu_r",
        norm=mcolors.Normalize(vmin=-1, vmax=1),
    )
    pod_sm.set_array([])
    fig.colorbar(pod_sm, cax=cbar_ax, label="Normalised mode amplitude")

    for row_idx, (grp, pod) in enumerate(zip(groups, pod_results)):
        stresses   = grp["stresses"]
        sf_vals    = grp["sf_vals"]
        best_idx   = pod["best_idx"]
        basis      = pod["basis"]

        params_best = grp["params"][best_idx]
        try:
            weapon_best    = build_spiral_weapon(params_best, cfg)
            assembled_best = assemble_weapon(weapon_best, cfg.mounting) if weapon_best else None
        except Exception:
            weapon_best = assembled_best = None

        def _draw_outline(ax):
            if assembled_best is None:
                return
            polys = ([assembled_best] if assembled_best.geom_type == "Polygon"
                     else list(assembled_best.geoms))
            for p in polys:
                ax.plot(*np.array(p.exterior.coords).T, "w-", lw=0.7, alpha=0.8)

        R = float(cfg.envelope.max_radius_mm)

        # ── Col 0: weapon shape ────────────────────────────────────────────
        ax_shape = fig.add_subplot(data_gs[row_idx, 0])
        ax_shape.set_aspect("equal"); ax_shape.axis("off")
        ax_shape.set_xlim(-R * 1.05, R * 1.05)
        ax_shape.set_ylim(-R * 1.05, R * 1.05)
        if assembled_best is not None:
            polys = ([assembled_best] if assembled_best.geom_type == "Polygon"
                     else list(assembled_best.geoms))
            for p in polys:
                ext = np.array(p.exterior.coords)
                ax_shape.fill(ext[:, 0], ext[:, 1], color="#4a90d9", alpha=0.85)
                ax_shape.plot(ext[:, 0], ext[:, 1], "k-", lw=0.5)
                for ring in p.interiors:
                    rint = np.array(ring.coords)
                    ax_shape.fill(rint[:, 0], rint[:, 1], color="white")
                    ax_shape.plot(rint[:, 0], rint[:, 1], "k-", lw=0.4)
        n_st = int(round(float(params_best[1])))
        ax_shape.set_title(f"n={n_st}  pitch={params_best[0]:.0f}mm",
                           fontsize=7, pad=2)
        ax_shape.set_ylabel(row_labels[row_idx], fontsize=9, fontweight="bold",
                            labelpad=4)

        # ── Col 1: raw FEA on weapon mesh ─────────────────────────────────
        ax_raw = fig.add_subplot(data_gs[row_idx, 1])
        ax_raw.set_aspect("equal"); ax_raw.axis("off")
        raw = _raw_fea(params_best, cfg)
        if raw is not None:
            r_nodes = raw["nodes"]; r_elems = raw["elements"]; r_vm = raw["vm_stresses"]
            vmax_raw = float(np.percentile(r_vm[r_vm > 0], 95)) if (r_vm > 0).any() else 1.0
            tri_raw  = mtri.Triangulation(r_nodes[:, 0], r_nodes[:, 1], r_elems)
            # tripcolor: one colour per triangle, no interpolation
            tcp_raw  = ax_raw.tripcolor(tri_raw, r_vm, cmap="plasma",
                                        vmin=0, vmax=vmax_raw, shading="flat")
            sf_raw   = raw["safety_factor"]
            sf_lbl   = f"SF={sf_raw:.2f}" if np.isfinite(sf_raw) else "SF=∞"
            ax_raw.set_title(sf_lbl, fontsize=7, pad=2)
            _draw_outline(ax_raw)
            plt.colorbar(tcp_raw, ax=ax_raw, fraction=0.046, pad=0.04,
                         label="MPa", format="%.0f")

        # ── Col 2: FEA interpolated to reference mesh ──────────────────────
        ax_ref = fig.add_subplot(data_gs[row_idx, 2])
        vm_best = stresses[best_idx].astype(np.float64)
        vmax_ref = float(np.percentile(vm_best[vm_best > 0], 95)) if (vm_best > 0).any() else 1.0
        nv_ref   = elements_to_nodes(vm_best, ref_elements, n_nodes)
        tcf_ref  = ax_ref.tricontourf(tri_ref, nv_ref, levels=20, cmap="plasma",
                                      vmin=0, vmax=vmax_ref)
        ax_ref.set_aspect("equal"); ax_ref.axis("off")
        sf_label = f"SF={sf_vals[best_idx]:.2f}" if np.isfinite(sf_vals[best_idx]) else "SF=∞"
        ax_ref.set_title(sf_label, fontsize=7, pad=2)
        _draw_outline(ax_ref)
        plt.colorbar(tcf_ref, ax=ax_ref, fraction=0.046, pad=0.04,
                     label="MPa", format="%.0f")

        # ── Cols 3–6: POD modes (normalised to [-1, 1]) ───────────────────
        for mode_idx in range(4):
            ax = fig.add_subplot(data_gs[row_idx, mode_idx + 3])
            if mode_idx < basis.shape[1]:
                mode_elem = basis[:, mode_idx]
                vabs = float(np.abs(mode_elem).max()) or 1.0
                mode_norm = mode_elem / vabs          # normalise to [-1, 1]
                node_mode = elements_to_nodes(mode_norm, ref_elements, n_nodes)
                ax.tricontourf(tri_ref, node_mode, levels=20, cmap="RdBu_r",
                               vmin=-1, vmax=1)
                var_pct = float(pod["variance_explained"][mode_idx]) * 100
                ax.set_title(f"{var_pct:.1f}% var", fontsize=7, pad=2)
            ax.set_aspect("equal"); ax.axis("off")

    fig.suptitle(
        "Stratified POD — Archimedean Spiral Weapon  (n_starts = 1, 2, 3, 4)\n"
        "Col 0: geometry  |  Col 1: FEA on design mesh  |  Col 2: FEA on reference mesh  "
        "|  Cols 3–6: POD modes (normalised, RdBu_r)",
        fontsize=10, y=0.98,
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] → {out_path}")


# ---------------------------------------------------------------------------
# Statistics / improvement figure
# ---------------------------------------------------------------------------

def make_stats_figure(
    groups: list[dict],
    pod_results: list[dict],
    labels: list[str],
    out_path: Path,
    dpi: int,
):
    """
    Two-panel figure showing stratification benefit vs. a global (all-n) basis.

    Panel A: Reconstruction RMSE(k) curves — per-n basis vs. global basis
             applied to held-in data of each group.
    Panel B: Summary bar chart — k@99% and k@95% for global vs per-n.
    """
    # Build global basis from all groups combined
    X_all = np.vstack([g["stresses"].astype(np.float64) for g in groups])
    mean_all = X_all.mean(axis=0)
    X_all_c  = X_all - mean_all
    _, S_g, Vt_g = np.linalg.svd(X_all_c, full_matrices=False)
    V_global = Vt_g.T   # (M, K_global)

    var_g    = (S_g**2) / (S_g**2).sum()
    cum_g    = np.cumsum(var_g)
    k99_g    = int(np.searchsorted(cum_g, 0.99)) + 1
    k95_g    = int(np.searchsorted(cum_g, 0.95)) + 1

    colors = plt.cm.tab10(np.linspace(0, 0.4, len(groups)))
    k_max  = min(120, V_global.shape[1])
    ks     = np.arange(1, k_max + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel A: RMSE curves ──────────────────────────────────────────────
    ax = axes[0]
    for grp, pod, lbl, col in zip(groups, pod_results, labels, colors):
        X = grp["stresses"].astype(np.float64)
        X_c = X - X.mean(axis=0)          # centre on group mean
        V_n = pod["basis"]                 # per-n basis

        rmse_global = np.zeros(k_max)
        rmse_pern   = np.zeros(k_max)
        total_var   = float(np.mean(X_c**2))

        for k in ks:
            Vg_k = V_global[:, :k]
            recon_g = (X_c @ Vg_k) @ Vg_k.T
            rmse_global[k-1] = np.sqrt(np.mean((X_c - recon_g)**2))

            Vn_k = V_n[:, :min(k, V_n.shape[1])]
            recon_n = (X_c @ Vn_k) @ Vn_k.T
            rmse_pern[k-1] = np.sqrt(np.mean((X_c - recon_n)**2))

        ax.plot(ks, rmse_global, color=col, ls="--", lw=1.5, alpha=0.7,
                label=f"{lbl} global")
        ax.plot(ks, rmse_pern,   color=col, ls="-",  lw=2.0,
                label=f"{lbl} per-n")

    ax.set_xlabel("Number of POD modes  k")
    ax.set_ylabel("Reconstruction RMSE  (MPa)")
    ax.set_title("Reconstruction error: global basis (--) vs. per-n basis (—)")
    ax.set_xlim(1, k_max)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Panel B: k summary bars ───────────────────────────────────────────
    ax2 = axes[1]
    # Groups: n=1, n=2, n=3, n=4, global
    bar_labels = labels + ["global"]
    n_bars = len(bar_labels)
    x      = np.arange(n_bars)
    width  = 0.35

    k99_vals = [pod["k_99"] for pod in pod_results] + [k99_g]
    k95_vals = [pod["k_95"] for pod in pod_results] + [k95_g]

    # Colors: per-n groups use tab10, global uses gray
    bar_colors = list(colors) + [np.array([0.5, 0.5, 0.5, 1.0])]

    bars1 = ax2.bar(x - width/2, k99_vals, width, label="k@99%",
                    color=bar_colors, alpha=0.9, edgecolor="k", lw=0.5)
    bars2 = ax2.bar(x + width/2, k95_vals, width, label="k@95%",
                    color=bar_colors, alpha=0.5, edgecolor="k", lw=0.5, hatch="//")

    # Reference lines for global basis (kept as requested)
    ax2.axhline(k99_g, color="gray", ls="--", lw=1.5, label=f"k@99% global ({k99_g})")
    ax2.axhline(k95_g, color="gray", ls=":",  lw=1.5, label=f"k@95% global ({k95_g})")

    ax2.set_xticks(x); ax2.set_xticklabels(bar_labels)
    ax2.set_ylabel("Modes required  k")
    ax2.set_title("Basis dimension: per-n stratified vs. global")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Annotate bars with values
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)),
                 ha="center", va="bottom", fontsize=7)

    # Summary table below
    rows_txt = ["N", "k@99% (per-n)", "k@99% (global)", "k@95% (per-n)",
                "k@95% (global)", "4-mode var%", "RMSE_k4 (MPa)"]
    col_data = []
    for grp, pod in zip(groups, pod_results):
        col_data.append([
            str(len(grp["stresses"])),
            str(pod["k_99"]),
            str(k99_g),
            str(pod["k_95"]),
            str(k95_g),
            f"{pod['k4_var']:.1f}",
            f"{pod['rmse_k4']:.2f}",
        ])

    table_ax = fig.add_axes([0.0, -0.28, 1.0, 0.25])
    table_ax.axis("off")
    tbl = table_ax.table(
        cellText=list(map(list, zip(*col_data))),
        rowLabels=rows_txt,
        colLabels=labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)

    fig.suptitle("POD Stratification Benefit — n_starts groups vs. global mixed basis",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] stats → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-group", type=int, default=N_PER_GROUP_DEFAULT,
                        help=f"Designs per n_starts group (default {N_PER_GROUP_DEFAULT})")
    parser.add_argument("--yes", action="store_true",
                        help="Skip compute confirmation prompt")
    parser.add_argument("--force-regen", action="store_true",
                        help="Delete and recompute all caches")
    parser.add_argument("--dpi", type=int, default=DPI_DEFAULT)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load reference mesh
    if not REF_MESH.exists():
        print(f"ERROR: reference mesh not found at {REF_MESH}")
        sys.exit(1)
    ref_nodes, ref_elements = _load_ref_mesh(REF_MESH)
    n_ref_elem = ref_elements.shape[0]
    print(f"[load] ref mesh: {ref_nodes.shape[0]} nodes, {n_ref_elem} elements")

    cfg = _make_cfg()

    # Estimate compute
    n_total = args.n_per_group * 4  # 4 groups
    print(f"\n  Need FEA for n_starts ∈ [1,2,3,4]  (~{n_total} calls, ~{n_total*8//60} min)")

    if not args.yes:
        ans = input("\n  Proceed? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    # Group definitions: (label, n_starts_fixed, cache_name)
    group_defs = [
        ("n=1",    1,    "n1"),
        ("n=2",    2,    "n2"),
        ("n=3",    3,    "n3"),
        ("n=4",    4,    "n4"),
    ]

    print("\n[1/3] Datasets ...")
    groups = []
    for label, n_fixed, cache_key in group_defs:
        cache_path = CACHE_DIR / f"{cache_key}_N{args.n_per_group}.npz"
        if args.force_regen and cache_path.exists():
            cache_path.unlink()

        print(f"  {label}: generating {args.n_per_group} designs ...")
        t0 = time.time()
        grp = _generate_group(
            n_starts_fixed=n_fixed,
            cfg=cfg,
            ref_nodes=ref_nodes,
            ref_elements=ref_elements,
            n_per_group=args.n_per_group,
            cache_path=cache_path,
        )
        elapsed = time.time() - t0
        print(f"    {label}: {len(grp['stresses'])}/{args.n_per_group} valid ({elapsed:.1f}s)")
        if len(grp['stresses']) < 10:
            print(f"    WARNING: too few valid designs for {label}, skipping")
            continue
        print(f"  [cache] saved → {cache_path.name}")
        groups.append(grp)

    print("\n[2/3] POD bases ...")
    pod_results = []
    row_labels  = []
    for (label, _, _), grp in zip(group_defs, groups):
        t0 = time.time()
        pod = compute_pod(grp["stresses"], grp["sf_vals"])
        elapsed = time.time() - t0
        print(f"  [{label:>5}]  N={len(grp['stresses']):4d}  "
              f"k@99%={pod['k_99']:4d}  k@95%={pod['k_95']:3d}  "
              f"4-mode var={pod['k4_var']:.1f}%  "
              f"RMSE_k4={pod['rmse_k4']:.2f}MPa  ({elapsed:.1f}s)")
        pod_results.append(pod)
        row_labels.append(label)

    # Pairwise geodesic distances
    if len(pod_results) >= 2:
        print("\n  Geodesic distances (k=10 modes):")
        from scipy.linalg import subspace_angles
        header = "        " + "".join(f"  {l:>6}" for l in row_labels)
        print(f"  {header}")
        for la, pa in zip(row_labels, pod_results):
            row = f"  {la:>6}:"
            for lb, pb in zip(row_labels, pod_results):
                k = min(10, pa["basis"].shape[1], pb["basis"].shape[1])
                angles = subspace_angles(pa["basis"][:, :k], pb["basis"][:, :k])
                dist = float(np.sqrt((angles**2).sum()))
                row += f"  {dist:6.3f}"
            print(row)

    print("\n[3/3] Figures ...")
    make_figure(
        groups, pod_results, ref_nodes, ref_elements, cfg,
        out_path=OUT_DIR / "pod_spiral_modes_table.png",
        dpi=args.dpi,
    )
    make_stats_figure(
        groups, pod_results, row_labels,
        out_path=OUT_DIR / "pod_spiral_stats.png",
        dpi=args.dpi,
    )
    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
