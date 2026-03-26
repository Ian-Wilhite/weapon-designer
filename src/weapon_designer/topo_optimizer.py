"""Topology optimisation (SIMP) for weapon weight-reduction — Phase 2 alternative.

Replaces the parametric polar-cutout Phase 2 with a continuum topology
optimisation that discovers the optimal material layout within the Phase-1
outer profile.

Method
------
SIMP — Solid Isotropic Material with Penalization:
    E_eff(ρ_e) = E_min + ρ_e^p · (E₀ − E_min)

Objective (per iteration, minimised):
    J = w_C · C / C₀  −  w_MOI · MOI / MOI₀

    C   = u^T F                (structural compliance, lower = stiffer)
    MOI = ρ_mat·t· Σ ρ_e·A_e·r_e²  (moment of inertia, higher = more energetic)
    C₀, MOI₀ = normalisation constants from iteration 1

Constraint:
    Σ ρ_e·A_e / A_total = V_f  (volume fraction = mass_budget / mass_solid)

Update:
    Optimality Criteria (OC) with bisection on the Lagrange multiplier.

Filter:
    Heuristic sensitivity averaging over a radius r_min (prevents checkerboard).

Post-processing:
    Elements with ρ_e < 0.5 are union'd to form void regions, then lightly
    smoothed and subtracted from the outer profile to yield the final polygon.

Units: mm, N, MPa, tonne  (consistent with the rest of the package).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from .config import WeaponConfig
from .fea import _triangulate_polygon, _apply_boundary_conditions
from .fea_viz import export_gif, render_fea_frame
from .fea import fea_stress_analysis_with_mesh


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIMP_P_DEFAULT   = 3.0      # SIMP penalisation exponent
_E_MIN_REL        = 1e-4     # void stiffness relative to E₀  (avoids singularity)
_RHO_MIN          = 1e-3     # minimum density (numerical floor)
_MOVE_LIMIT       = 0.20     # OC move limit per iteration
_OC_ETA           = 0.50     # OC damping exponent
_OC_BISECT_ITERS  = 60       # bisection iterations for Lagrange multiplier


# ---------------------------------------------------------------------------
# Pre-computation of element geometry
# ---------------------------------------------------------------------------

def _precompute_elements(
    nodes: np.ndarray,
    elements: np.ndarray,
    nu: float,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute per-element unit-modulus stiffness matrices, areas, centroids.

    Returns
    -------
    K_elem  : (n_elem, 6, 6) float array  — unit-modulus CST stiffness
    areas   : (n_elem,)       float array  — element areas in mm²
    centroids : (n_elem, 2)  float array  — element centroid positions
    """
    n_elem = len(elements)
    K_elem    = np.zeros((n_elem, 6, 6))
    areas     = np.zeros(n_elem)
    centroids = np.zeros((n_elem, 2))

    # Unit-modulus constitutive matrix (D without E factor)
    D_unit = (1.0 / (1.0 - nu**2)) * np.array([
        [1.0, nu,          0.0],
        [nu,  1.0,         0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])

    for i, elem in enumerate(elements):
        tri = nodes[elem]           # (3, 2)
        x0, y0 = tri[0]
        x1, y1 = tri[1]
        x2, y2 = tri[2]

        A2   = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        area = A2 / 2.0
        areas[i]       = area
        centroids[i]   = tri.mean(axis=0)

        if area < 1e-12:
            continue

        b = np.array([y1 - y2, y2 - y0, y0 - y1])
        c = np.array([x2 - x1, x0 - x2, x1 - x0])

        B = np.zeros((3, 6))
        for j in range(3):
            B[0, 2 * j]     = b[j]
            B[1, 2 * j + 1] = c[j]
            B[2, 2 * j]     = c[j]
            B[2, 2 * j + 1] = b[j]
        B /= A2

        K_elem[i] = thickness * area * (B.T @ D_unit @ B)

    return K_elem, areas, centroids


# ---------------------------------------------------------------------------
# SIMP stiffness assembly (vectorised)
# ---------------------------------------------------------------------------

def _assemble_simp_K(
    nodes: np.ndarray,
    elements: np.ndarray,
    K_elem: np.ndarray,
    densities: np.ndarray,
    E0: float,
    E_min: float,
    p: float,
) -> sparse.csr_matrix:
    """Assemble global stiffness with SIMP-interpolated per-element moduli.

    K_global = Σ_e  [E_min + ρ_e^p (E₀−E_min)] · K_e^unit
    """
    n_nodes = len(nodes)
    n_dof   = 2 * n_nodes
    n_elem  = len(elements)

    # Effective Young's modulus per element
    E_eff = E_min + densities**p * (E0 - E_min)         # (n_elem,)

    # DOF index arrays: shape (n_elem, 6)
    dofs = np.empty((n_elem, 6), dtype=int)
    for j in range(3):
        dofs[:, 2 * j]     = 2 * elements[:, j]
        dofs[:, 2 * j + 1] = 2 * elements[:, j] + 1

    # Scaled stiffness: (n_elem, 6, 6)
    K_scaled = K_elem * E_eff[:, np.newaxis, np.newaxis]

    # COO arrays — each element contributes a 6×6 block (36 entries)
    rows = np.repeat(dofs[:, :, np.newaxis], 6, axis=2).ravel()   # (n_elem·36,)
    cols = np.repeat(dofs[:, np.newaxis, :], 6, axis=1).ravel()
    vals = K_scaled.ravel()

    K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return K.tocsr()


# ---------------------------------------------------------------------------
# Density-scaled centrifugal load
# ---------------------------------------------------------------------------

def _centrifugal_load_simp(
    nodes: np.ndarray,
    elements: np.ndarray,
    omega: float,
    density_tonne_mm3: float,
    thickness: float,
    densities: np.ndarray,
) -> np.ndarray:
    """Centrifugal body-force vector scaled by element densities.

    Body force: f_e = ρ_material · ρ_e · ω² · r_e · A_e · t  (N per element)
    Distributed equally to the 3 element nodes.
    """
    n_dof = 2 * len(nodes)
    F     = np.zeros(n_dof)

    tri = nodes[elements]                                  # (n_elem, 3, 2)
    cx  = tri[:, :, 0].mean(axis=1)                       # (n_elem,)
    cy  = tri[:, :, 1].mean(axis=1)
    r   = np.hypot(cx, cy)

    x, y  = tri[:, :, 0], tri[:, :, 1]
    areas = 0.5 * np.abs(
        (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
        - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    )

    valid = r > 1e-6
    f_mag = np.where(
        valid,
        densities * density_tonne_mm3 * omega**2 * r * areas * thickness,
        0.0,
    )
    safe_r = np.where(valid, r, 1.0)
    fx = f_mag * cx / safe_r
    fy = f_mag * cy / safe_r

    # Scatter to nodes (1/3 each)
    for j in range(3):
        nids = elements[:, j]
        np.add.at(F, 2 * nids,     fx / 3.0)
        np.add.at(F, 2 * nids + 1, fy / 3.0)

    return F


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def _compliance_sensitivities(
    elements: np.ndarray,
    K_elem: np.ndarray,
    u: np.ndarray,
    densities: np.ndarray,
    p: float,
    E0: float,
    E_min: float,
) -> np.ndarray:
    """Per-element compliance sensitivity ∂C/∂ρ_e (always ≤ 0).

    ∂C/∂ρ_e = −p · ρ_e^(p−1) · (E₀−E_min) · u_e^T · K_e^unit · u_e
    """
    n_elem = len(elements)

    # DOF indices: (n_elem, 6)
    dofs = np.empty((n_elem, 6), dtype=int)
    for j in range(3):
        dofs[:, 2 * j]     = 2 * elements[:, j]
        dofs[:, 2 * j + 1] = 2 * elements[:, j] + 1

    u_elem = u[dofs]                                    # (n_elem, 6)
    Ku     = np.einsum("ijk,ik->ij", K_elem, u_elem)   # (n_elem, 6)
    se     = np.einsum("ij,ij->i",   u_elem, Ku)       # element strain energy

    return -p * densities**(p - 1) * (E0 - E_min) * se   # (n_elem,), ≤ 0


def _moi_sensitivities(
    areas: np.ndarray,
    centroids: np.ndarray,
    density_tonne_mm3: float,
    thickness: float,
) -> np.ndarray:
    """Per-element MOI sensitivity ∂MOI/∂ρ_e (always ≥ 0).

    ∂MOI/∂ρ_e = ρ_material · t · A_e · r_e²
    """
    r_sq = centroids[:, 0]**2 + centroids[:, 1]**2
    return density_tonne_mm3 * thickness * areas * r_sq   # (n_elem,), ≥ 0


# ---------------------------------------------------------------------------
# Sensitivity filter  (heuristic — prevents checkerboard artefacts)
# ---------------------------------------------------------------------------

def _density_filter(
    sensitivities: np.ndarray,
    areas: np.ndarray,
    centroids: np.ndarray,
    r_min: float,
) -> np.ndarray:
    """Weighted-average sensitivity filter over circular neighbourhood r_min.

    H_ef = max(0, r_min − ||c_e − c_f||)
    s̃_e = Σ_f H_ef A_f s_f  /  Σ_f H_ef A_f
    """
    tree = cKDTree(centroids)
    neighbor_lists = tree.query_ball_point(centroids, r_min)

    filtered = np.empty_like(sensitivities)
    for e, nbrs in enumerate(neighbor_lists):
        if not nbrs:
            filtered[e] = sensitivities[e]
            continue
        nb   = np.array(nbrs)
        dist = np.hypot(centroids[nb, 0] - centroids[e, 0],
                        centroids[nb, 1] - centroids[e, 1])
        H    = np.maximum(0.0, r_min - dist)
        w    = H * areas[nb]
        den  = w.sum()
        filtered[e] = (w * sensitivities[nb]).sum() / den if den > 1e-14 else sensitivities[e]

    return filtered


# ---------------------------------------------------------------------------
# Optimality Criteria update
# ---------------------------------------------------------------------------

def _oc_update(
    densities: np.ndarray,
    sens_filtered: np.ndarray,
    areas: np.ndarray,
    v_target: float,
    fixed_void: np.ndarray,
    fixed_solid: np.ndarray,
    rho_min: float = _RHO_MIN,
    move: float    = _MOVE_LIMIT,
    eta: float     = _OC_ETA,
) -> np.ndarray:
    """Optimality Criteria density update with bisection on volume constraint.

    B_e = sqrt(−∂J/∂ρ_e / (λ · ∂g/∂ρ_e))
    ρ_e_new = clip(ρ_e · B_e^η,  ρ_e−move,  ρ_e+move)
    Volume constraint g = Σ ρ_e A_e / A_total − V_f = 0  →  bisect on λ.
    """
    total_area = areas.sum()

    # Clamp sensitivities: must be negative for OC to work
    sens = np.minimum(sens_filtered, -1e-30)
    # dg/drho_e = area_e / total_area
    dg = areas / (total_area + 1e-12)

    def _rho_new(lam: float) -> np.ndarray:
        B   = np.sqrt(-sens / (lam * dg + 1e-30))
        rho = densities * B**eta
        rho = np.clip(rho,
                      np.maximum(rho_min, densities - move),
                      np.minimum(1.0,     densities + move))
        rho[fixed_void]  = rho_min
        rho[fixed_solid] = 1.0
        return rho

    # Bisection on λ
    l1, l2 = 1e-10, 1e10
    for _ in range(_OC_BISECT_ITERS):
        lam_mid = 0.5 * (l1 + l2)
        rho_mid = _rho_new(lam_mid)
        vf_mid  = (rho_mid * areas).sum() / total_area
        if vf_mid > v_target:
            l1 = lam_mid
        else:
            l2 = lam_mid
        if (l2 - l1) < 1e-8 * l1:
            break

    return _rho_new(0.5 * (l1 + l2))


# ---------------------------------------------------------------------------
# Frame visualisation
# ---------------------------------------------------------------------------

def render_topo_frame(
    nodes: np.ndarray,
    elements: np.ndarray,
    densities: np.ndarray,
    strain_energy: np.ndarray,
    outer_poly: Polygon | MultiPolygon,
    iteration: int,
    v_current: float,
    v_target: float,
    compliance: float,
    moi_mm2: float,
    mass_kg: float,
    cfg: WeaponConfig,
    save_path: Path | str,
    dpi: int = 90,
    allowable_void_poly: Polygon | MultiPolygon | None = None,
) -> Path | None:
    """Save a two-panel topology-optimisation progress frame.

    Left  — element density field (white = solid, black = void)
    Right — element strain-energy density (hot: bright = load-bearing)

    Parameters
    ----------
    strain_energy : per-element u_e^T K_e u_e (absolute, before normalisation)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import LogNorm, Normalize
    except ImportError:
        return None

    BG = "#12122a"
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG)

    triangles = nodes[elements]        # (n_elem, 3, 2)

    # ── LEFT: density field ──────────────────────────────────────────────
    ax0.set_facecolor(BG)
    ax0.set_aspect("equal")

    col0 = PolyCollection(
        triangles,
        array=densities,
        cmap="bone",          # dark=void, white=solid
        norm=Normalize(vmin=0.0, vmax=1.0),
        edgecolors="none",
        antialiased=False,
    )
    ax0.add_collection(col0)

    # Draw outer profile outline
    if not outer_poly.is_empty:
        if isinstance(outer_poly, MultiPolygon):
            for sub in outer_poly.geoms:
                x, y = sub.exterior.xy
                ax0.plot(x, y, color="#4a90e2", linewidth=0.8, alpha=0.6)
        else:
            x, y = outer_poly.exterior.xy
            ax0.plot(x, y, color="#4a90e2", linewidth=0.8, alpha=0.6)

    # Draw allowable void region boundary (dashed cyan)
    if allowable_void_poly is not None and not allowable_void_poly.is_empty:
        for geom in (allowable_void_poly.geoms if hasattr(allowable_void_poly, "geoms")
                     else [allowable_void_poly]):
            x, y = geom.exterior.xy
            ax0.plot(x, y, color="#00e5ff", linewidth=0.7, linestyle="--", alpha=0.55)

    cbar0 = fig.colorbar(col0, ax=ax0, fraction=0.03, pad=0.03)
    cbar0.set_label("Density ρ", color="white", fontsize=8)
    cbar0.ax.tick_params(colors="white", labelsize=7)
    plt.setp(plt.getp(cbar0.ax.axes, "yticklabels"), color="white")

    ax0.set_title(
        f"Iter {iteration:03d}  ·  Density field\n"
        f"Vf = {v_current:.3f} / {v_target:.3f}  ·  "
        f"Mass = {mass_kg:.3f} kg",
        color="white", fontsize=8.5, pad=5,
    )
    ax0.tick_params(colors="white", labelsize=7)
    ax0.set_xlabel("mm", color="white", fontsize=8)
    ax0.set_ylabel("mm", color="white", fontsize=8)
    for sp in ax0.spines.values():
        sp.set_edgecolor("#333355")
    ax0.autoscale()

    # ── RIGHT: strain energy density ─────────────────────────────────────
    ax1.set_facecolor(BG)
    ax1.set_aspect("equal")

    se_plot = strain_energy.copy()
    se_plot = np.maximum(se_plot, 1e-30)

    # Log-normalised so both load-bearing elements and lightly-stressed
    # elements are visible simultaneously.
    se_max  = se_plot.max()
    se_norm = se_plot / (se_max + 1e-30)

    col1 = PolyCollection(
        triangles,
        array=se_norm,
        cmap="inferno",
        norm=Normalize(vmin=0.0, vmax=1.0),
        edgecolors="none",
        antialiased=False,
    )
    ax1.add_collection(col1)

    if not outer_poly.is_empty:
        if isinstance(outer_poly, MultiPolygon):
            for sub in outer_poly.geoms:
                x, y = sub.exterior.xy
                ax1.plot(x, y, color="#cccccc", linewidth=0.5, alpha=0.5)
        else:
            x, y = outer_poly.exterior.xy
            ax1.plot(x, y, color="#cccccc", linewidth=0.5, alpha=0.5)

    # Draw allowable void region boundary (dashed cyan)
    if allowable_void_poly is not None and not allowable_void_poly.is_empty:
        for geom in (allowable_void_poly.geoms if hasattr(allowable_void_poly, "geoms")
                     else [allowable_void_poly]):
            x, y = geom.exterior.xy
            ax1.plot(x, y, color="#00e5ff", linewidth=0.7, linestyle="--", alpha=0.55)

    cbar1 = fig.colorbar(col1, ax=ax1, fraction=0.03, pad=0.03)
    cbar1.set_label("Strain energy (norm.)", color="white", fontsize=8)
    cbar1.ax.tick_params(colors="white", labelsize=7)
    plt.setp(plt.getp(cbar1.ax.axes, "yticklabels"), color="white")

    ax1.set_title(
        f"Strain-energy density  ·  C = {compliance:.3e}\n"
        f"MOI = {moi_mm2:.0f} kg·mm²",
        color="white", fontsize=8.5, pad=5,
    )
    ax1.tick_params(colors="white", labelsize=7)
    ax1.set_xlabel("mm", color="white", fontsize=8)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#333355")
    ax1.autoscale()

    plt.tight_layout(pad=1.2)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


def render_binary_frame(
    nodes: np.ndarray,
    elements: np.ndarray,
    densities: np.ndarray,
    outer_poly: Polygon | MultiPolygon,
    extracted_poly: Polygon | MultiPolygon | None,
    iteration: int,
    v_current: float,
    v_target: float,
    compliance: float,
    cfg: WeaponConfig,
    save_path: Path | str,
    dpi: int = 90,
    allowable_void_poly: Polygon | MultiPolygon | None = None,
) -> Path | None:
    """Binary threshold view: left = thresholded mesh, right = extracted polygon."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import Normalize
    except ImportError:
        return None

    BG = "#12122a"
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    triangles = nodes[elements]

    # ── LEFT: binary element mesh ─────────────────────────────────────────
    ax0.set_facecolor(BG)
    ax0.set_aspect("equal")
    binary = (densities >= 0.5).astype(float)
    col0 = PolyCollection(
        triangles,
        array=binary,
        cmap="bone",
        norm=Normalize(vmin=0.0, vmax=1.0),
        edgecolors="none",
        antialiased=False,
    )
    ax0.add_collection(col0)

    # Draw allowable void region boundary on binary mesh panel (dashed cyan)
    if allowable_void_poly is not None and not allowable_void_poly.is_empty:
        for geom in (allowable_void_poly.geoms if hasattr(allowable_void_poly, "geoms")
                     else [allowable_void_poly]):
            x, y = geom.exterior.xy
            ax0.plot(x, y, color="#00e5ff", linewidth=0.7, linestyle="--", alpha=0.55)

    ax0.set_title(
        f"Iter {iteration:03d}  ·  Binary design (ρ ≥ 0.5)\n"
        f"Vf = {v_current:.3f} / {v_target:.3f}",
        color="white", fontsize=8.5, pad=5,
    )
    ax0.tick_params(colors="white", labelsize=7)
    ax0.set_xlabel("mm", color="white", fontsize=8)
    ax0.set_ylabel("mm", color="white", fontsize=8)
    for sp in ax0.spines.values():
        sp.set_edgecolor("#333355")
    ax0.autoscale()

    # ── RIGHT: extracted Shapely polygon ─────────────────────────────────
    ax1.set_facecolor(BG)
    ax1.set_aspect("equal")

    def _draw(ax, poly, face="#4a90e2", edge="#2c5f8a"):
        if poly is None or poly.is_empty:
            return
        if isinstance(poly, MultiPolygon):
            for sub in poly.geoms:
                _draw(ax, sub, face, edge)
            return
        x, y = poly.exterior.xy
        ax.fill(x, y, fc=face, ec=edge, linewidth=0.6, alpha=0.85)
        for interior in poly.interiors:
            xi, yi = interior.xy
            ax.fill(xi, yi, fc="black", ec="#cc3333", linewidth=0.5, alpha=1.0)

    if extracted_poly is not None:
        _draw(ax1, extracted_poly)
    else:
        _draw(ax1, outer_poly)

    # Draw allowable void region boundary on extracted polygon panel (dashed cyan)
    if allowable_void_poly is not None and not allowable_void_poly.is_empty:
        for geom in (allowable_void_poly.geoms if hasattr(allowable_void_poly, "geoms")
                     else [allowable_void_poly]):
            x, y = geom.exterior.xy
            ax1.plot(x, y, color="#00e5ff", linewidth=0.7, linestyle="--", alpha=0.55)

    ax1.set_title(
        f"Extracted weapon polygon\nC = {compliance:.3e}",
        color="white", fontsize=8.5, pad=5,
    )
    ax1.tick_params(colors="white", labelsize=7)
    ax1.set_xlabel("mm", color="white", fontsize=8)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#333355")
    ax1.autoscale()

    plt.tight_layout(pad=1.2)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


def save_convergence_plot(
    history: list[dict],
    save_path: Path | str,
    dpi: int = 110,
) -> Path | None:
    """Save a convergence summary plot (compliance, volume fraction, MOI)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not history:
        return None

    iters  = [h["iteration"] for h in history]
    comps  = [h["compliance"] for h in history]
    vfs    = [h["v_current"] for h in history]
    mois   = [h["moi_kg_mm2"] for h in history]
    change = [h.get("max_density_change", 0.0) for h in history]

    BG = "#12122a"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Topology Optimisation Convergence", color="white", fontsize=12)

    def _styled_ax(ax, ys, label, color, ylabel):
        ax.set_facecolor("#1a1a2e")
        ax.plot(iters, ys, color=color, linewidth=1.5)
        ax.set_xlabel("Iteration", color="white", fontsize=8)
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.set_title(label, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        ax.yaxis.label.set_color("white")

    _styled_ax(axes[0, 0], comps,  "Compliance (lower = stiffer)",
               "#e05c5c", "Compliance (N·mm)")
    _styled_ax(axes[0, 1], vfs,    "Volume fraction",
               "#5ce0a0", "Vf (target shown)")
    axes[0, 1].axhline(vfs[-1] if vfs else 0.5,
                       color="white", linestyle="--", alpha=0.4, linewidth=0.8)
    _styled_ax(axes[1, 0], mois,   "Moment of Inertia",
               "#5c9ce0", "MOI (kg·mm²)")
    _styled_ax(axes[1, 1], change, "Max density change per iter",
               "#e0c05c", "Δρ_max")

    plt.tight_layout(pad=1.5)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Post-processing: densities → cutout polygons
# ---------------------------------------------------------------------------

def densities_to_cutout_polygons(
    nodes: np.ndarray,
    elements: np.ndarray,
    densities: np.ndarray,
    outer_polygon: Polygon | MultiPolygon,
    threshold: float   = 0.50,
    min_area_mm2: float = 30.0,
    smooth_mm: float    = 1.5,
) -> list[Polygon]:
    """Convert void elements (ρ < threshold) to smooth Shapely cutout polygons.

    Triangles below the threshold are union'd, lightly smoothed (buffer in/out),
    then filtered by minimum area and clipped to the outer profile.

    Returns a list of Polygon objects suitable for assemble_weapon().
    """
    void_tris: list[Polygon] = []
    for e, elem in enumerate(elements):
        if densities[e] < threshold:
            pts = [nodes[elem[0]].tolist(),
                   nodes[elem[1]].tolist(),
                   nodes[elem[2]].tolist()]
            try:
                tri = Polygon(pts)
                if tri.is_valid and tri.area > 1e-6:
                    void_tris.append(tri)
            except Exception:
                pass

    if not void_tris:
        return []

    void_union = unary_union(void_tris)

    # Smooth: fill tiny holes + merge barely-touching regions
    void_smooth = void_union.buffer(smooth_mm * 1.2).buffer(-smooth_mm)

    # Clip to outer profile (don't remove material outside the envelope)
    void_clipped = void_smooth.intersection(outer_polygon)

    # Collect individual polygons
    if void_clipped.geom_type == "Polygon":
        polys = [void_clipped]
    elif void_clipped.geom_type == "MultiPolygon":
        polys = list(void_clipped.geoms)
    else:
        polys = [g for g in getattr(void_clipped, "geoms", [])
                 if g.geom_type == "Polygon"]

    # Filter fragments
    polys = [p for p in polys if p.is_valid and p.area >= min_area_mm2]
    return polys


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def topology_optimize(
    solid_polygon: Polygon | MultiPolygon,
    cfg: WeaponConfig,
    case_dir: Path,
    fea_interval: int      = 5,
    log_fn                  = None,
) -> dict:
    """Run SIMP topology optimisation as a Phase-2 alternative.

    Parameters
    ----------
    solid_polygon : weapon polygon from Phase 1 (bore + bolt holes subtracted,
                    no weight-reduction cutouts).  This is the design domain.
    cfg           : full weapon configuration.
    case_dir      : output directory for frames and GIFs.
    fea_interval  : save an FEA stress frame every N topo iterations (0 = off).
    log_fn        : callable(str) for progress messages.

    Returns
    -------
    dict with keys:
        weapon_polygon   : final assembled polygon
        cutout_polygons  : list of void regions extracted from the density field
        history          : list of per-iteration metric dicts
        gif_topo         : Path to density-evolution GIF (or None)
        gif_binary       : Path to binary-design GIF (or None)
        gif_fea          : Path to FEA stress-frame GIF (or None)
        convergence_plot : Path to static convergence PNG (or None)
        logs             : list of log strings
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    logs: list[str] = []
    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)
        logs.append(msg)

    # ── config parameters ─────────────────────────────────────────────────
    opt = cfg.optimization
    n_iter         = getattr(opt, "topo_n_iter",          60)
    mesh_spacing   = getattr(opt, "topo_mesh_spacing_mm", 6.0)
    p_simp         = getattr(opt, "topo_p_simp",          _SIMP_P_DEFAULT)
    r_min_factor   = getattr(opt, "topo_r_min_factor",    2.5)
    w_compliance   = getattr(opt, "topo_w_compliance",    0.5)
    frame_interval = getattr(opt, "topo_frame_interval",  2)
    edge_offset    = getattr(opt, "topo_edge_offset_mm",   5.0)

    w_moi      = 1.0 - w_compliance
    r_min      = r_min_factor * mesh_spacing
    E0         = cfg.material.density_kg_m3 * 0.0  # placeholder — use material E
    E0         = 200_000.0   # MPa — structural steel default
    E_min      = E0 * _E_MIN_REL
    nu         = 0.3
    thickness  = cfg.sheet_thickness_mm
    rpm        = cfg.rpm
    omega      = rpm * 2.0 * np.pi / 60.0
    rho_tonne  = cfg.material.density_kg_m3 * 1e-12    # tonne/mm³
    bore_r     = cfg.mounting.bore_diameter_mm / 2.0
    max_r      = cfg.envelope.max_radius_mm
    yield_mpa  = cfg.material.yield_strength_mpa

    # Target volume fraction from mass budget
    solid_area   = solid_polygon.area
    target_area  = cfg.weight_budget_kg / (cfg.material.density_kg_m3 * 1e-9 * thickness)
    v_target     = float(np.clip(target_area / max(solid_area, 1.0), 0.05, 0.99))

    _log(f"[topo] solid_area={solid_area:.0f} mm²  target_area={target_area:.0f} mm²  "
         f"V_f={v_target:.3f}")
    _log(f"[topo] mesh_spacing={mesh_spacing} mm  r_min={r_min:.1f} mm  "
         f"p={p_simp}  w_C={w_compliance:.2f}  w_MOI={w_moi:.2f}  n_iter={n_iter}")

    # ── mesh the solid polygon ────────────────────────────────────────────
    max_area = mesh_spacing**2
    nodes, elements = _triangulate_polygon(solid_polygon, max_area=max_area)

    if len(elements) < 10:
        _log("[topo] ERROR: mesh too coarse — falling back to unmodified polygon")
        return {
            "weapon_polygon":   solid_polygon,
            "cutout_polygons":  [],
            "history":          [],
            "gif_topo":         None,
            "gif_binary":       None,
            "gif_fea":          None,
            "convergence_plot": None,
            "logs":             logs,
        }

    n_elem = len(elements)
    _log(f"[topo] mesh: {len(nodes)} nodes, {n_elem} elements")

    # ── pre-compute element geometry ──────────────────────────────────────
    K_elem, areas, centroids = _precompute_elements(nodes, elements, nu, thickness)

    # ── fixed element classification (offset-based) ────────────────────────
    # Allowable void region = solid polygon eroded inward by edge_offset_mm.
    # Negative buffer simultaneously shrinks the outer boundary inward (rim stays
    # solid) and expands interior holes outward (hub/bolt zones stay solid).
    from shapely.prepared import prep as _shp_prep
    allowable_void_poly = solid_polygon.buffer(-edge_offset)

    if allowable_void_poly.is_empty or allowable_void_poly.area < solid_polygon.area * 0.05:
        _log(
            f"[topo] WARNING: topo_edge_offset_mm={edge_offset} too large; "
            f"halving to {edge_offset / 2.0:.1f}"
        )
        allowable_void_poly = solid_polygon.buffer(-edge_offset / 2.0)

    if not allowable_void_poly.is_empty:
        _prep = _shp_prep(allowable_void_poly)
        in_allowable = np.fromiter(
            (_prep.contains(Point(c)) for c in centroids),
            dtype=bool,
            count=n_elem,
        )
    else:
        in_allowable = np.zeros(n_elem, dtype=bool)

    fixed_solid = ~in_allowable   # elements outside the eroded region stay solid
    fixed_void  = np.zeros(n_elem, dtype=bool)

    _log(
        f"[topo] edge_offset={edge_offset} mm  "
        f"allowable_area={allowable_void_poly.area:.0f} mm²  "
        f"free_elements={in_allowable.sum()}/{n_elem}"
    )

    # ── boundary conditions (for FEA solve) ──────────────────────────────
    _, F_dummy, free_dofs = _apply_boundary_conditions(
        sparse.eye(2 * len(nodes), format="csr"),
        np.zeros(2 * len(nodes)),
        nodes, bore_r,
    )
    n_dof = 2 * len(nodes)

    # ── initialise densities at v_target ─────────────────────────────────
    densities          = np.full(n_elem, v_target)
    densities[fixed_solid] = 1.0
    densities[fixed_void]  = _RHO_MIN

    # Enforce v_target after fixed elements
    free_mask   = ~(fixed_solid | fixed_void)
    fixed_area  = (areas[fixed_solid]).sum()
    free_area   = areas[free_mask].sum()
    target_free = max(0.0, v_target * solid_area - fixed_area) / max(free_area, 1e-6)
    target_free = float(np.clip(target_free, _RHO_MIN, 1.0))
    densities[free_mask] = target_free

    # ── output directories ────────────────────────────────────────────────
    topo_dir   = case_dir / "frames_topo"
    binary_dir = case_dir / "frames_topo_binary"
    fea_dir    = case_dir / "frames_topo_fea"
    topo_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    if fea_interval > 0:
        fea_dir.mkdir(parents=True, exist_ok=True)

    # ── normalisation constants (from first valid solve) ─────────────────
    C_norm   = None
    moi_norm = None

    history: list[dict] = []
    frame_idx = 0

    t0 = time.time()

    for it in range(1, n_iter + 1):
        # ── assemble SIMP stiffness ───────────────────────────────────────
        K_simp = _assemble_simp_K(nodes, elements, K_elem, densities, E0, E_min, p_simp)

        # ── centrifugal load (density-scaled) ────────────────────────────
        F_cent = _centrifugal_load_simp(nodes, elements, omega, rho_tonne,
                                         thickness, densities)

        # ── solve K u = F ─────────────────────────────────────────────────
        u = np.zeros(n_dof)
        try:
            K_free  = K_simp[np.ix_(free_dofs, free_dofs)]
            F_free  = F_cent[free_dofs]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u_free = spsolve(K_free, F_free)
            if np.isfinite(u_free).all():
                u[free_dofs] = u_free
        except Exception as exc:
            _log(f"  [topo iter {it}] solver error: {exc}")
            continue

        # ── compliance ───────────────────────────────────────────────────
        compliance = float(u @ F_cent)
        if C_norm is None or C_norm < 1e-12:
            C_norm = max(abs(compliance), 1e-12)

        # ── MOI estimate (sum of rho_e * A_e * r_e²) ─────────────────────
        moi_mm2 = float((densities * areas * (centroids[:, 0]**2 + centroids[:, 1]**2)).sum()
                        * rho_tonne * thickness * 1e12 / 1e12)
        # Convert tonne·mm⁴ → kg·mm²:  rho_tonne·A·t·r² [tonne·mm²] × 1e3 → kg·mm²
        moi_kg_mm2 = float((densities * areas * (centroids[:, 0]**2 + centroids[:, 1]**2)).sum()
                           * rho_tonne * thickness * 1e3)
        if moi_norm is None or moi_norm < 1e-12:
            moi_norm = max(moi_kg_mm2, 1e-12)

        # ── mass ─────────────────────────────────────────────────────────
        mass_kg = float((densities * areas).sum() * rho_tonne * thickness * 1e3)
        v_current = float((densities * areas).sum() / max(solid_area, 1.0))

        # ── per-element strain energy (for vis) ──────────────────────────
        dofs_arr = np.empty((n_elem, 6), dtype=int)
        for j in range(3):
            dofs_arr[:, 2 * j]     = 2 * elements[:, j]
            dofs_arr[:, 2 * j + 1] = 2 * elements[:, j] + 1
        u_elem   = u[dofs_arr]
        Ku_e     = np.einsum("ijk,ik->ij", K_elem, u_elem)
        se_elem  = np.einsum("ij,ij->i",   u_elem, Ku_e)   # per-element strain energy

        # ── combined sensitivity ──────────────────────────────────────────
        comp_sens = _compliance_sensitivities(
            elements, K_elem, u, densities, p_simp, E0, E_min)
        moi_sens  = _moi_sensitivities(areas, centroids, rho_tonne, thickness)

        # Normalise and combine: both terms push in same direction (negative)
        c_max = max(np.abs(comp_sens).max(), 1e-30)
        m_max = max(moi_sens.max(), 1e-30)
        J_sens = (w_compliance * (comp_sens / c_max)
                  - w_moi      * (moi_sens  / m_max))

        # ── sensitivity filter ────────────────────────────────────────────
        J_filtered = _density_filter(J_sens, areas, centroids, r_min)

        # ── OC update ─────────────────────────────────────────────────────
        densities_new = _oc_update(
            densities, J_filtered, areas, v_target,
            fixed_void, fixed_solid,
        )
        max_delta = float(np.abs(densities_new - densities).max())
        densities = densities_new

        # ── log ───────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        entry = {
            "iteration":         it,
            "elapsed_s":         round(elapsed, 1),
            "compliance":        float(compliance),
            "moi_kg_mm2":        round(moi_kg_mm2, 2),
            "mass_kg":           round(mass_kg, 4),
            "v_current":         round(v_current, 4),
            "v_target":          round(v_target, 4),
            "max_density_change": round(max_delta, 5),
        }
        history.append(entry)

        if it % 10 == 0 or it <= 3 or it == n_iter:
            _log(
                f"  [topo] iter {it:3d}/{n_iter}  "
                f"C={compliance:.3e}  Vf={v_current:.3f}/{v_target:.3f}  "
                f"MOI={moi_kg_mm2:.0f}  mass={mass_kg:.3f}kg  "
                f"Δρ={max_delta:.4f}  [{elapsed:.0f}s]"
            )

        # ── save frames ───────────────────────────────────────────────────
        save_this = (frame_interval <= 0) or (it % max(1, frame_interval) == 0) or it == 1 or it == n_iter

        if save_this:
            render_topo_frame(
                nodes, elements, densities, se_elem,
                solid_polygon,
                iteration=it,
                v_current=v_current,
                v_target=v_target,
                compliance=compliance,
                moi_mm2=moi_mm2,
                mass_kg=mass_kg,
                cfg=cfg,
                save_path=topo_dir / f"topo_{frame_idx:04d}.png",
                allowable_void_poly=allowable_void_poly,
            )

            # Adaptive threshold for intermediate binary view
            lo_b, hi_b = 0.0, 1.0
            for _ in range(30):
                tau_b    = 0.5 * (lo_b + hi_b)
                vf_b     = areas[densities < tau_b].sum() / max(areas.sum(), 1.0)
                if vf_b < (1.0 - v_target):
                    lo_b = tau_b
                else:
                    hi_b = tau_b
            cur_threshold = float(np.clip(0.5 * (lo_b + hi_b), 0.1, 0.9))

            # Extract current polygon for binary view
            curr_cutouts = densities_to_cutout_polygons(
                nodes, elements, densities, solid_polygon,
                threshold=cur_threshold, min_area_mm2=10.0, smooth_mm=1.0,
            )
            from .geometry import assemble_weapon
            from .constraints import validate_geometry
            if curr_cutouts:
                curr_poly = assemble_weapon(solid_polygon, cfg.mounting, curr_cutouts)
                curr_poly = validate_geometry(curr_poly)
            else:
                curr_poly = solid_polygon

            render_binary_frame(
                nodes, elements, densities,
                solid_polygon, curr_poly,
                iteration=it,
                v_current=v_current,
                v_target=v_target,
                compliance=compliance,
                cfg=cfg,
                save_path=binary_dir / f"binary_{frame_idx:04d}.png",
                allowable_void_poly=allowable_void_poly,
            )

            # Save sidecar JSON
            try:
                meta = {
                    "iteration": it,
                    "compliance": float(compliance),
                    "moi_kg_mm2": round(moi_kg_mm2, 2),
                    "mass_kg": round(mass_kg, 4),
                    "v_current": round(v_current, 4),
                    "v_target": round(v_target, 4),
                    "max_density_change": round(max_delta, 5),
                }
                (topo_dir / f"topo_{frame_idx:04d}_meta.json").write_text(
                    json.dumps(meta, indent=2))
            except Exception:
                pass

            frame_idx += 1

        # ── optional FEA frame ────────────────────────────────────────────
        if fea_interval > 0 and (it % fea_interval == 0 or it == n_iter):
            try:
                from .geometry import assemble_weapon
                from .constraints import validate_geometry
                fea_cutouts = densities_to_cutout_polygons(
                    nodes, elements, densities, solid_polygon,
                    threshold=0.5, min_area_mm2=20.0, smooth_mm=1.5,
                )
                fea_poly = assemble_weapon(solid_polygon, cfg.mounting, fea_cutouts)
                fea_poly = validate_geometry(fea_poly)
                fea_data = fea_stress_analysis_with_mesh(
                    fea_poly,
                    rpm=rpm,
                    density_kg_m3=cfg.material.density_kg_m3,
                    thickness_mm=thickness,
                    yield_strength_mpa=yield_mpa,
                    bore_diameter_mm=cfg.mounting.bore_diameter_mm,
                    mesh_spacing=cfg.optimization.fea_fine_spacing_mm,
                )
                render_fea_frame(
                    fea_poly, fea_data, cfg,
                    step_label=f"Topo-{it:03d}",
                    metrics={
                        "mass_kg":     mass_kg,
                        "moi_kg_mm2":  moi_kg_mm2,
                        "energy_joules": 0.5 * moi_kg_mm2 * omega**2 * 1e-6,
                        "bite_mm":     0.0,
                        "n_teeth":     0,
                    },
                    save_path=fea_dir / f"fea_{it:04d}.png",
                )
            except Exception as exc:
                _log(f"  [topo FEA frame error iter {it}: {exc}]")

        # ── early convergence check ───────────────────────────────────────
        if it > 5 and max_delta < 5e-4:
            _log(f"  [topo] converged at iter {it} (Δρ_max={max_delta:.2e})")
            break

    # ── post-processing: extract final cutout polygons ────────────────────
    # Use adaptive thresholding: find τ such that the void elements account for
    # exactly (1 − V_f) of the total area.  This ensures the extracted binary
    # design has the same mass as the continuous SIMP solution.
    total_area = areas.sum()
    lo_t, hi_t = 0.0, 1.0
    for _ in range(50):
        tau_mid  = 0.5 * (lo_t + hi_t)
        void_frac = areas[densities < tau_mid].sum() / max(total_area, 1.0)
        if void_frac < (1.0 - v_target):
            lo_t = tau_mid   # too few voids → lower threshold → more voids
        else:
            hi_t = tau_mid
    adaptive_threshold = float(np.clip(0.5 * (lo_t + hi_t), 0.1, 0.9))
    _log(f"[topo] Adaptive threshold τ={adaptive_threshold:.3f} "
         f"(Vf={v_target:.3f})")

    _log("[topo] Extracting final void regions...")
    cutout_polys = densities_to_cutout_polygons(
        nodes, elements, densities, solid_polygon,
        threshold=adaptive_threshold, min_area_mm2=30.0, smooth_mm=2.0,
    )
    _log(f"[topo] Found {len(cutout_polys)} void region(s) after threshold")

    # ── assemble final weapon polygon ─────────────────────────────────────
    from .geometry import assemble_weapon
    from .constraints import validate_geometry
    final_weapon = assemble_weapon(solid_polygon, cfg.mounting, cutout_polys)
    final_weapon = validate_geometry(final_weapon)

    final_mass = (cfg.material.density_kg_m3 * 1e-9 * thickness * final_weapon.area)
    _log(f"[topo] Final: mass={final_mass:.3f} kg  "
         f"(budget={cfg.weight_budget_kg:.3f} kg  "
         f"error={abs(final_mass - cfg.weight_budget_kg)/cfg.weight_budget_kg*100:.1f}%)")

    # ── save final FEA stress frame ───────────────────────────────────────
    try:
        final_fea = fea_stress_analysis_with_mesh(
            final_weapon,
            rpm=rpm,
            density_kg_m3=cfg.material.density_kg_m3,
            thickness_mm=thickness,
            yield_strength_mpa=yield_mpa,
            bore_diameter_mm=cfg.mounting.bore_diameter_mm,
            mesh_spacing=cfg.optimization.fea_fine_spacing_mm,
        )
        from .objectives_enhanced import compute_metrics_enhanced
        final_metrics = compute_metrics_enhanced(
            final_weapon, cfg,
            fea_spacing=cfg.optimization.fea_fine_spacing_mm,
        )
        render_fea_frame(
            final_weapon, final_fea, cfg,
            step_label="Topo-FINAL",
            metrics=final_metrics,
            save_path=case_dir / "topo_final_stress.png",
            dpi=120,
        )
        _log(f"[topo] Final FEA: SF={final_fea['safety_factor']:.2f}  "
             f"peak={final_fea['peak_stress_mpa']:.0f} MPa")
    except Exception as exc:
        _log(f"[topo] Final FEA error: {exc}")
        final_metrics = {}

    # ── convergence plot ──────────────────────────────────────────────────
    conv_plot = save_convergence_plot(history, case_dir / "topo_convergence.png")

    # ── assemble GIFs ─────────────────────────────────────────────────────
    gif_topo   = export_gif(topo_dir,   case_dir / "convergence_topo_density.gif", fps=5)
    gif_binary = export_gif(binary_dir, case_dir / "convergence_topo_binary.gif",  fps=5)
    gif_fea    = export_gif(fea_dir,    case_dir / "convergence_topo_fea.gif",     fps=3) \
                 if fea_interval > 0 else None

    if gif_topo:
        _log(f"[topo] GIF: {gif_topo}")
    if gif_binary:
        _log(f"[topo] GIF: {gif_binary}")
    if gif_fea:
        _log(f"[topo] GIF: {gif_fea}")

    return {
        "weapon_polygon":   final_weapon,
        "cutout_polygons":  cutout_polys,
        "history":          history,
        "final_metrics":    final_metrics,
        "gif_topo":         gif_topo,
        "gif_binary":       gif_binary,
        "gif_fea":          gif_fea,
        "convergence_plot": conv_plot,
        "logs":             logs,
        "v_target":         v_target,
        "n_elements":       n_elem,
    }
