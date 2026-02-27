#!/usr/bin/env python3
"""fea_speed_sweep.py — 3-D stacked FEA heatmaps: centrifugal + spiral-impact vs. RPM.

Loads a weapon DXF file, sweeps through a range of spin speeds, and renders a
publication-quality 3-D figure:

    XY plane   = weapon cross-section (mm)
    Z axis     = spin speed (RPM)
    Colour     = σ_VM / σ_yield   blue=safe → red=critical → dark-red=failed
    Yellow     = Archimedean spiral contact path per RPM level
    Orange dot = spiral contact point (impact site)

Sidebar panels show:
    • Peak centrifugal stress vs RPM
    • Peak combined stress (centrifugal + impact) vs RPM
    • Safety factor vs RPM with operating-range bands

Force model
-----------
Two load cases are combined at each RPM:

  1. Centrifugal body force
        f_body = ρ · ω² · r   (N/mm³)
     Scales as ω².  Produces hoop tension around the bore and root stresses at
     each structural section.

  2. Spiral-impact force
        F_impact = ½ · m_weapon · ω² · r_contact   (N)
     Equal to the centripetal reaction at the contact radius — the force the
     shaft must supply to keep the tip on its circular path is exactly what the
     weapon delivers to the enemy on each hit.  Applied as a concentrated load
     opposing weapon rotation at the Archimedean-spiral first-contact angle.

     This also scales as ω², but is spatially concentrated at the outer edge,
     creating bending stresses different from the body-force pattern.  Because
     bite_mm ∝ 1/ω (enemy approaches a fixed distance per revolution), the
     energy-per-hit grows as ω³ — higher-RPM weapons hit proportionally harder.

Usage
-----
    python3 fea_speed_sweep.py [dxf] [stats_json] [options]

    python3 fea_speed_sweep.py output_disk.dxf output_disk_stats.json
    python3 fea_speed_sweep.py output_disk.dxf output_disk_stats.json \\
        --rpm-min 500 --rpm-max 12000 --rpm-steps 9 \\
        --mesh-spacing 5.0 --out speed_sweep.png

Defaults: dxf=output_disk.dxf, stats=output_disk_stats.json, rpm-min=500,
          rpm-max=12000, rpm-steps=9, mesh-spacing=5.0, drive-speed=3.0 m/s
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import matplotlib

# mpl_toolkits is a namespace package; on systems with both a system-package
# matplotlib and a pip-installed one the wrong mplot3d is picked up.  Fix the
# path NOW — before matplotlib.pyplot is imported — so that pyplot's early
# projection scan finds the correct pip-installed mplot3d and registers '3d'.
import mpl_toolkits as _mpl_tk
_pip_site = Path(matplotlib.__file__).parent.parent  # e.g. ~/.local/…/site-packages
_mpl3d_pip = _pip_site / "mpl_toolkits"
if _mpl3d_pip.exists():
    _mpl_tk.__path__ = [str(_mpl3d_pip)]

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy import sparse
from scipy.spatial import Delaunay
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
import ezdxf
from shapely.geometry import Polygon, MultiPolygon, Point


# ─────────────────────────────────────────────────────────────────────────────
# Material / config bundle
# ─────────────────────────────────────────────────────────────────────────────

class SweepParams(NamedTuple):
    density_kg_m3: float       # material density
    yield_mpa: float           # yield strength (MPa)
    E_mpa: float               # Young's modulus (MPa)
    nu: float                  # Poisson's ratio
    thickness_mm: float        # sheet thickness (mm)
    bore_mm: float             # bore diameter (mm)
    mass_kg: float             # weapon mass (kg)
    drive_speed_mps: float     # enemy approach speed (m/s)
    mesh_spacing: float        # FEA element target edge length (mm)


_DEFAULT_PARAMS = SweepParams(
    density_kg_m3=7850.0,
    yield_mpa=1400.0,
    E_mpa=200_000.0,
    nu=0.3,
    thickness_mm=10.0,
    bore_mm=25.4,
    mass_kg=1.0,
    drive_speed_mps=3.0,
    mesh_spacing=5.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# DXF → Shapely polygon
# ─────────────────────────────────────────────────────────────────────────────

def load_polygon_from_dxf(dxf_path: str | Path) -> Polygon | MultiPolygon:
    """Read a weapon DXF (as written by exporter.py) and return a Shapely polygon.

    Layer "WEAPON"       → exterior ring(s)
    Layer "WEAPON_HOLES" → hole rings
    """
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    weapon_polys: list[Polygon] = []
    hole_polys: list[Polygon] = []

    for entity in msp.query("LWPOLYLINE"):
        layer = entity.dxf.layer.upper()
        pts = [(v[0], v[1]) for v in entity.get_points()]
        if len(pts) < 3:
            continue
        p = Polygon(pts)
        if not p.is_valid:
            p = p.buffer(0)
        if layer == "WEAPON":
            weapon_polys.append(p)
        elif "HOLES" in layer:
            hole_polys.append(p)

    if not weapon_polys:
        raise ValueError(f"No 'WEAPON' layer polylines found in {dxf_path}")

    # Merge all exterior rings
    from shapely.ops import unary_union
    weapon = unary_union(weapon_polys)

    # Subtract holes
    for hole in hole_polys:
        weapon = weapon.difference(hole)

    if not weapon.is_valid:
        weapon = weapon.buffer(0)

    return weapon


# ─────────────────────────────────────────────────────────────────────────────
# Meshing helpers (adapted from fea.py)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_boundary(poly: Polygon | MultiPolygon, spacing: float = 3.0) -> np.ndarray:
    points = []

    def _sample_ring(coords: list, sp: float) -> np.ndarray:
        ring_pts = np.array(coords[:-1])
        if len(ring_pts) < 2:
            return ring_pts
        diffs = np.diff(ring_pts, axis=0)
        lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        total = lengths.sum()
        n_pts = max(int(total / sp), len(ring_pts))
        cum = np.concatenate([[0], np.cumsum(lengths)])
        targets = np.linspace(0, total, n_pts, endpoint=False)
        sampled = np.zeros((n_pts, 2))
        for i, t in enumerate(targets):
            idx = min(np.searchsorted(cum, t, side="right") - 1, len(ring_pts) - 2)
            frac = (t - cum[idx]) / max(lengths[idx], 1e-12)
            sampled[i] = ring_pts[idx] + frac * diffs[idx]
        return sampled

    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            points.append(_sample_boundary(p, spacing))
        return np.vstack(points) if points else np.zeros((0, 2))

    points.append(_sample_ring(list(poly.exterior.coords), spacing))
    for interior in poly.interiors:
        points.append(_sample_ring(list(interior.coords), spacing))

    return np.vstack(points)


def _interior_grid(poly: Polygon | MultiPolygon, spacing: float = 5.0) -> np.ndarray:
    if isinstance(poly, MultiPolygon):
        grids = [_interior_grid(p, spacing) for p in poly.geoms]
        return np.vstack(grids) if grids else np.zeros((0, 2))

    bounds = poly.bounds
    xs = np.arange(bounds[0] + spacing, bounds[2], spacing)
    ys = np.arange(bounds[1] + spacing, bounds[3], spacing)
    grid = np.array([(x, y) for x in xs for y in ys])
    if len(grid) == 0:
        return np.zeros((0, 2))
    mask = np.array([poly.contains(Point(p[0], p[1])) for p in grid])
    return grid[mask]


def triangulate_polygon(poly: Polygon | MultiPolygon, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Delaunay triangulation; returns (nodes N×2, elements M×3)."""
    bnd = _sample_boundary(poly, spacing=spacing)
    interior = _interior_grid(poly, spacing=spacing)
    all_pts = np.vstack([bnd, interior]) if len(interior) > 0 else bnd

    if len(all_pts) < 3:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=int)

    tri = Delaunay(all_pts)
    simplices = tri.simplices
    centroids = all_pts[simplices].mean(axis=1)
    mask = np.array([poly.contains(Point(c[0], c[1])) for c in centroids])
    elements = simplices[mask]
    return all_pts, elements


# ─────────────────────────────────────────────────────────────────────────────
# CST FEA (adapted from fea.py)
# ─────────────────────────────────────────────────────────────────────────────

def _cst_stiffness(nodes3: np.ndarray, E: float, nu: float, t: float) -> np.ndarray:
    x, y = nodes3[:, 0], nodes3[:, 1]
    A2 = abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    area = A2 / 2.0
    if area < 1e-12:
        return np.zeros((6, 6))
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2*i] = b[i]; B[1, 2*i+1] = c[i]
        B[2, 2*i] = c[i]; B[2, 2*i+1] = b[i]
    B /= A2
    D = (E / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    return t * area * (B.T @ D @ B)


def assemble_stiffness(nodes: np.ndarray, elements: np.ndarray,
                       E: float, nu: float, t: float) -> sparse.csr_matrix:
    n_dof = 2 * len(nodes)
    rows, cols, vals = [], [], []
    for elem in elements:
        Ke = _cst_stiffness(nodes[elem], E, nu, t)
        dofs = np.array([2*elem[0], 2*elem[0]+1,
                         2*elem[1], 2*elem[1]+1,
                         2*elem[2], 2*elem[2]+1])
        for i in range(6):
            for j in range(6):
                if abs(Ke[i, j]) > 1e-20:
                    rows.append(dofs[i]); cols.append(dofs[j]); vals.append(Ke[i, j])
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()


def centrifugal_load(nodes: np.ndarray, elements: np.ndarray,
                     omega: float, rho_tmm3: float, t: float) -> np.ndarray:
    """Body force vector for centrifugal loading (mm-N-tonne system)."""
    F = np.zeros(2 * len(nodes))
    for elem in elements:
        tri = nodes[elem]
        x, y = tri[:, 0], tri[:, 1]
        area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
        if area < 1e-12:
            continue
        cx, cy = tri.mean(axis=0)
        r = np.hypot(cx, cy)
        if r < 1e-6:
            continue
        f_mag = rho_tmm3 * omega**2 * r * area * t
        fx, fy = f_mag * cx / r, f_mag * cy / r
        for i in range(3):
            nid = elem[i]
            F[2*nid] += fx / 3.0
            F[2*nid+1] += fy / 3.0
    return F


def impact_load(nodes: np.ndarray,
                contacts: list[tuple[float, float]],
                total_force_N: float) -> np.ndarray:
    """Apply impact forces distributed equally across all spiral contact points.

    Parameters
    ----------
    contacts        : list of (angle_rad, r_mm) for every contact in one revolution.
    total_force_N   : total impact force for the revolution (N).
                      Divided equally among all contact points so that more
                      contacts = lower force per point (balanced loading).

    Force direction at each contact opposes weapon CCW rotation:
        d = (sin θ, -cos θ)

    Each contact's load is spread (inverse-distance weighted) to the 5 nearest
    outer-boundary nodes.
    """
    F = np.zeros(2 * len(nodes))
    if total_force_N < 1e-3 or not contacts:
        return F

    per_contact_N = total_force_N / len(contacts)

    # Outer-boundary node cache (r > 80 % of max)
    r_nodes = np.hypot(nodes[:, 0], nodes[:, 1])
    outer_idx = np.where(r_nodes > r_nodes.max() * 0.80)[0]
    if len(outer_idx) == 0:
        return F

    for angle, r_mm in contacts:
        cx = r_mm * np.cos(angle)
        cy = r_mm * np.sin(angle)
        fx_dir = np.sin(angle)    # opposes CCW rotation
        fy_dir = -np.cos(angle)

        dists = np.hypot(nodes[outer_idx, 0] - cx, nodes[outer_idx, 1] - cy)
        k = min(5, len(outer_idx))
        near = np.argpartition(dists, k-1)[:k]
        nids = outer_idx[near]
        w = 1.0 / (dists[near] + 1e-3)
        w /= w.sum()

        for nid, wi in zip(nids, w):
            F[2*nid]   += per_contact_N * fx_dir * wi
            F[2*nid+1] += per_contact_N * fy_dir * wi

    return F


def free_dofs_from_bc(nodes: np.ndarray, bore_r: float) -> np.ndarray:
    r = np.hypot(nodes[:, 0], nodes[:, 1])
    fixed = np.where(r <= bore_r * 1.5)[0]
    if len(fixed) == 0:
        fixed = np.argsort(r)[:3]
    fixed_dofs = np.sort(np.concatenate([2*fixed, 2*fixed+1]))
    return np.setdiff1d(np.arange(2 * len(nodes)), fixed_dofs)


def von_mises(nodes: np.ndarray, elements: np.ndarray,
              u: np.ndarray, E: float, nu: float) -> np.ndarray:
    D = (E / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    stresses = np.zeros(len(elements))
    for idx, elem in enumerate(elements):
        tri = nodes[elem]
        x, y = tri[:, 0], tri[:, 1]
        A2 = abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
        if A2 < 1e-12:
            continue
        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2*i] = b[i]; B[1, 2*i+1] = c[i]
            B[2, 2*i] = c[i]; B[2, 2*i+1] = b[i]
        B /= A2
        dofs = np.array([2*elem[0], 2*elem[0]+1,
                         2*elem[1], 2*elem[1]+1,
                         2*elem[2], 2*elem[2]+1])
        sx, sy, txy = D @ B @ u[dofs]
        stresses[idx] = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)
    return stresses


def solve_displacements(K: sparse.csr_matrix, F: np.ndarray,
                        free_dofs: np.ndarray) -> np.ndarray:
    n_dof = len(F)
    u = np.zeros(n_dof)
    if len(free_dofs) == 0:
        return u
    K_f = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u_f = spsolve(K_f, F_f)
        if np.isfinite(u_f).all():
            u[free_dofs] = u_f
    except Exception:
        pass
    return u


# ─────────────────────────────────────────────────────────────────────────────
# Spiral contact detection
# ─────────────────────────────────────────────────────────────────────────────

def _radial_profile(poly: Polygon, n: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta_grid, r_profile) for the outer boundary."""
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)
    cx, cy = poly.centroid.x, poly.centroid.y
    ext = np.array(poly.exterior.coords[:-1])
    angles = np.arctan2(ext[:, 1] - cy, ext[:, 0] - cx)
    radii = np.hypot(ext[:, 0] - cx, ext[:, 1] - cy)
    sort_i = np.argsort(angles)
    a_s, r_s = angles[sort_i], radii[sort_i]
    a_w = np.concatenate([a_s - 2*np.pi, a_s, a_s + 2*np.pi])
    r_w = np.tile(r_s, 3)
    a_u, uid = np.unique(a_w, return_index=True)
    r_u = r_w[uid]
    f = interp1d(a_u, r_u, kind="linear", fill_value="extrapolate")
    theta_grid = np.linspace(-np.pi, np.pi, n, endpoint=False)
    return theta_grid, f(theta_grid)


def spiral_contact(poly: Polygon | MultiPolygon,
                   omega: float, v_mm_s: float,
                   n: int = 720) -> dict:
    """Detect ALL Archimedean spiral contact points in one revolution.

    Returns
    -------
    contacts   : list of (angle_rad, r_mm) for every crossing in [-π, π]
    n_contacts : len(contacts), min 1
    bite_mm    : v_per_rad · 2π / n_contacts  (depth per hit)
    spiral_xy  : (x_arr, y_arr) full spiral path for plotting
    """
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)

    theta_grid, r_profile = _radial_profile(poly, n)
    v_per_rad = v_mm_s / omega
    r_start = r_profile.max() * 1.01
    r_spiral = r_start - v_per_rad * (theta_grid + np.pi)  # maps [-π,π] → [0, 2π] advance

    diff = r_spiral - r_profile
    crossing_idx = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0]

    contacts: list[tuple[float, float]] = []
    for idx in crossing_idx:
        frac = diff[idx] / (diff[idx] - diff[idx+1] + 1e-12)
        angle = float(theta_grid[idx] + frac * (theta_grid[1] - theta_grid[0]))
        r_mm  = float(r_profile[idx])
        contacts.append((angle, r_mm))

    if not contacts:
        # Smooth disk — single contact at the maximum-radius point
        peak_idx = np.argmax(r_profile)
        contacts = [(float(theta_grid[peak_idx]), float(r_profile[peak_idx]))]

    n_contacts = len(contacts)
    bite_mm = v_per_rad * 2 * np.pi / n_contacts

    # Full spiral path for plotting (weapon frame, CW as weapon turns CCW)
    cx, cy = poly.centroid.x, poly.centroid.y
    t_plot = np.linspace(0, 2*np.pi, 600)
    r_plot_min = 5.0
    r_plot = np.clip(r_start - v_per_rad * t_plot, r_plot_min, None)
    theta_plot = np.pi/2 - t_plot   # start at 12 o'clock
    x_sp = cx + r_plot * np.cos(theta_plot)
    y_sp = cy + r_plot * np.sin(theta_plot)
    spiral_xy = (x_sp[r_plot > r_plot_min + 0.1], y_sp[r_plot > r_plot_min + 0.1])

    return {
        "contacts":   contacts,
        "n_contacts": n_contacts,
        "bite_mm":    bite_mm,
        "spiral_xy":  spiral_xy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

class SliceResult(NamedTuple):
    rpm: float
    omega: float
    vm_centrifugal: np.ndarray    # per-element von Mises, centrifugal only (MPa)
    vm_combined: np.ndarray       # per-element von Mises, centrifugal + impact (MPa)
    peak_cent_mpa: float
    peak_combined_mpa: float
    safety_factor_cent: float
    safety_factor_combined: float
    contacts: list                # list of (angle_rad, r_mm) — all contact points
    n_contacts: int
    bite_mm: float
    spiral_xy: tuple              # (x_arr, y_arr) full spiral path
    impact_force_N: float         # total force for the revolution (N)


def run_speed_sweep(
    poly: Polygon | MultiPolygon,
    rpms: np.ndarray,
    params: SweepParams,
) -> tuple[np.ndarray, np.ndarray, list[SliceResult]]:
    """
    Returns (nodes, elements, list_of_SliceResult).

    The stiffness matrix K is assembled once; only load vectors change per RPM.
    """
    print(f"[sweep] Meshing at spacing={params.mesh_spacing:.1f} mm …", flush=True)
    nodes, elements = triangulate_polygon(poly, params.mesh_spacing)
    if len(elements) < 3:
        raise RuntimeError("Mesh degenerate — try smaller mesh_spacing.")
    print(f"[sweep] Mesh: {len(nodes)} nodes, {len(elements)} elements")

    # Assemble K once
    print("[sweep] Assembling stiffness matrix …", flush=True)
    K = assemble_stiffness(nodes, elements, params.E_mpa, params.nu, params.thickness_mm)
    free_dofs = free_dofs_from_bc(nodes, params.bore_mm / 2)
    rho_tmm3 = params.density_kg_m3 * 1e-12   # kg/m³ → tonne/mm³

    results: list[SliceResult] = []

    for rpm in rpms:
        omega = rpm * 2 * np.pi / 60.0
        v_mm_s = params.drive_speed_mps * 1000.0
        print(f"[sweep]   RPM={rpm:>7.0f}  ω={omega:6.1f} rad/s", end=" ", flush=True)

        # ── Centrifugal load ──────────────────────────────────────────────
        F_cent = centrifugal_load(nodes, elements, omega, rho_tmm3, params.thickness_mm)
        u_cent = solve_displacements(K, F_cent, free_dofs)
        vm_cent = von_mises(nodes, elements, u_cent, params.E_mpa, params.nu)
        peak_cent = float(vm_cent.max()) if len(vm_cent) > 0 else 0.0

        # ── Spiral contact (all points) ───────────────────────────────────
        sc = spiral_contact(poly, omega, v_mm_s)

        # ── Impact load ───────────────────────────────────────────────────
        # Total force F = ½ · m · ω² · r_mean  (centripetal reaction at mean
        # contact radius).  Divided equally across ALL contact points so that
        # a weapon with more teeth gets the same total force but balanced
        # across its perimeter — matching the physical reality that each tooth
        # carries only its share of the load.
        r_mean_m = np.mean([r for _, r in sc["contacts"]]) * 1e-3  # mm → m
        F_imp_N  = 0.5 * params.mass_kg * omega**2 * r_mean_m

        F_imp     = impact_load(nodes, sc["contacts"], F_imp_N)
        F_combined = F_cent + F_imp
        u_comb    = solve_displacements(K, F_combined, free_dofs)
        vm_comb   = von_mises(nodes, elements, u_comb, params.E_mpa, params.nu)
        peak_comb = float(vm_comb.max()) if len(vm_comb) > 0 else 0.0

        sf_cent = params.yield_mpa / peak_cent if peak_cent > 1e-6 else 999.0
        sf_comb = params.yield_mpa / peak_comb if peak_comb > 1e-6 else 999.0

        print(f"σ_cent={peak_cent:6.0f} MPa  σ_comb={peak_comb:6.0f} MPa  "
              f"SF_comb={sf_comb:.2f}  n_contacts={sc['n_contacts']}  "
              f"F_imp={F_imp_N/1000:.1f} kN")

        results.append(SliceResult(
            rpm=float(rpm),
            omega=omega,
            vm_centrifugal=vm_cent,
            vm_combined=vm_comb,
            peak_cent_mpa=peak_cent,
            peak_combined_mpa=peak_comb,
            safety_factor_cent=sf_cent,
            safety_factor_combined=sf_comb,
            contacts=sc["contacts"],
            n_contacts=sc["n_contacts"],
            bite_mm=sc["bite_mm"],
            spiral_xy=sc["spiral_xy"],
            impact_force_N=F_imp_N,
        ))

    return nodes, elements, results


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

# Custom colormap: dark-blue → cyan → green → yellow → red → dark-red
_CMAP = LinearSegmentedColormap.from_list(
    "stress_sweep",
    [(0.0,  "#0a1628"),   # dark navy (no stress)
     (0.25, "#1565C0"),   # blue
     (0.50, "#43A047"),   # green
     (0.72, "#FDD835"),   # yellow
     (0.88, "#E53935"),   # red (near yield)
     (1.00, "#4A0000")],  # dark red (failed)
)


def _z_for_rpm(rpm: float, rpm_min: float, rpm_max: float,
               z_min: float = 0.0, z_max: float = 1.0) -> float:
    return z_min + (rpm - rpm_min) / (rpm_max - rpm_min + 1e-9) * (z_max - z_min)


def plot_3d_sweep(
    poly: Polygon | MultiPolygon,
    nodes: np.ndarray,
    elements: np.ndarray,
    results: list[SliceResult],
    params: SweepParams,
    output_path: str | Path | None = None,
    show_centrifugal: bool = False,
) -> None:
    """Render the 3-D stacked heatmap figure.

    Parameters
    ----------
    show_centrifugal : if True, use centrifugal-only stresses for the heatmap;
                       otherwise use combined (centrifugal + impact).
    """
    matplotlib.use("Agg" if output_path else "TkAgg")

    rpms    = np.array([r.rpm for r in results])
    rpm_min = rpms.min()
    rpm_max = rpms.max()

    # Z scale: spread slices in 3-D space
    z_min, z_max = 0.0, 1.0
    z_gap = (z_max - z_min) / max(len(results) - 1, 1)

    # ── Figure layout ─────────────────────────────────────────────────────
    BG = "#0d1117"
    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[2.8, 1, 1],
        height_ratios=[1.6, 1],
        wspace=0.40, hspace=0.55,
    )

    ax3d  = fig.add_subplot(gs[:, 0], projection="3d")   # main 3-D stack
    ax_sf = fig.add_subplot(gs[0, 1])                     # safety factor vs RPM
    ax_st = fig.add_subplot(gs[0, 2])                     # peak stress vs RPM
    ax_xy = fig.add_subplot(gs[1, 1])                     # worst-case 2-D heatmap
    ax_tb = fig.add_subplot(gs[1, 2])                     # operating range table

    for ax in [ax_sf, ax_st, ax_xy, ax_tb]:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a3a5a")

    fig.patch.set_facecolor(BG)
    ax3d.set_facecolor(BG)

    norm  = Normalize(vmin=0.0, vmax=1.2)
    cmap  = _CMAP

    # ── 3-D stacked heatmaps ─────────────────────────────────────────────
    # Precompute element centroids for contact-point identification
    centroids = nodes[elements].mean(axis=1)  # (M, 2)

    for res in results:
        z_i = _z_for_rpm(res.rpm, rpm_min, rpm_max, z_min, z_max)

        # Choose stress data
        vm = res.vm_combined if not show_centrifugal else res.vm_centrifugal
        stress_ratio = (vm / max(params.yield_mpa, 1e-6)).clip(0.0, 1.2)

        # Build Poly3DCollection for this slice
        tris_3d = []
        face_colors = []
        for i, elem in enumerate(elements):
            v0 = (nodes[elem[0], 0], nodes[elem[0], 1], z_i)
            v1 = (nodes[elem[1], 0], nodes[elem[1], 1], z_i)
            v2 = (nodes[elem[2], 0], nodes[elem[2], 1], z_i)
            tris_3d.append([v0, v1, v2])
            face_colors.append(cmap(norm(stress_ratio[i])))

        pcoll = Poly3DCollection(tris_3d, alpha=0.55, linewidths=0)
        pcoll.set_facecolors(face_colors)
        ax3d.add_collection3d(pcoll)

        # ── Archimedean spiral at this Z level ────────────────────────────
        xs, ys = res.spiral_xy
        if len(xs) > 1:
            zs = np.full(len(xs), z_i)
            ax3d.plot(xs, ys, zs, color="#FFD54F", alpha=0.6, linewidth=0.7)

        # ── Contact point markers (all contacts this RPM level) ───────────
        for angle, r_mm in res.contacts:
            cx_c = r_mm * np.cos(angle)
            cy_c = r_mm * np.sin(angle)
            ax3d.scatter([cx_c], [cy_c], [z_i],
                         color="#FF6F00", s=22, zorder=10, depthshade=False)

    # ── Weapon outline at bottom and top ─────────────────────────────────
    def _add_outline_3d(poly_in, z_val, col="#4488cc", lw=0.7):
        if isinstance(poly_in, MultiPolygon):
            for p in poly_in.geoms:
                _add_outline_3d(p, z_val, col, lw)
            return
        x, y = poly_in.exterior.xy
        ax3d.plot(list(x), list(y), [z_val]*len(x), color=col, lw=lw, alpha=0.5)

    _add_outline_3d(poly, z_min, col="#334455")
    _add_outline_3d(poly, z_max, col="#4488cc")

    # ── Failure-RPM vertical plane (if any slice SF_comb < 1) ─────────────
    fail_slices = [r for r in results if r.safety_factor_combined < 1.0]
    if fail_slices:
        fail_rpm = fail_slices[0].rpm
        z_fail = _z_for_rpm(fail_rpm, rpm_min, rpm_max, z_min, z_max)
        # Draw a translucent red plane at z_fail
        # Approximate bounds from polygon
        bds = poly.bounds  # (minx, miny, maxx, maxy)
        xs_pl = [bds[0], bds[2], bds[2], bds[0]]
        ys_pl = [bds[1], bds[1], bds[3], bds[3]]
        zs_pl = [z_fail]*4
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _P3D
        fail_plane = _P3D([[list(zip(xs_pl, ys_pl, zs_pl))[0],
                            list(zip(xs_pl, ys_pl, zs_pl))[1],
                            list(zip(xs_pl, ys_pl, zs_pl))[2],
                            list(zip(xs_pl, ys_pl, zs_pl))[3]]],
                          alpha=0.15, facecolors=["red"], edgecolors=["red"],
                          linewidths=1.5)
        ax3d.add_collection3d(fail_plane)

    # ── 3-D axes decoration ───────────────────────────────────────────────
    ax3d.set_xlabel("X (mm)", color="#8899bb", fontsize=8, labelpad=6)
    ax3d.set_ylabel("Y (mm)", color="#8899bb", fontsize=8, labelpad=6)
    ax3d.set_zlabel("")
    ax3d.zaxis.set_ticks([_z_for_rpm(r, rpm_min, rpm_max) for r in rpms[::2]])
    ax3d.zaxis.set_ticklabels([f"{int(r)}" for r in rpms[::2]],
                               color="#8899bb", fontsize=7)
    ax3d.set_zticks([_z_for_rpm(r, rpm_min, rpm_max) for r in rpms])
    ax3d.set_zticklabels([f"{int(r)}" for r in rpms], color="#8899bb", fontsize=6)
    ax3d.tick_params(colors="#8899bb", labelsize=6)
    ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#1a2533"); ax3d.yaxis.pane.set_edgecolor("#1a2533")
    ax3d.zaxis.pane.set_edgecolor("#1a2533")
    ax3d.grid(True, color="#1a2533", linewidth=0.4)
    mode = "combined (centrifugal + impact)" if not show_centrifugal else "centrifugal only"
    ax3d.set_title(
        f"FEA stress field — {mode}\nσ_VM / σ_yield  |  RPM axis (Z)",
        color="white", fontsize=9, pad=8,
    )
    ax3d.view_init(elev=22, azim=-55)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax3d, shrink=0.55, pad=0.08, aspect=28,
                      orientation="vertical")
    cb.set_label("σ_VM / σ_yield", color="#8899bb", fontsize=8)
    cb.ax.tick_params(colors="#8899bb", labelsize=7)
    cb.ax.axhline(y=1.0, color="red", linewidth=1.5, linestyle="--")  # yield line

    # ── Right panel 1: Safety Factor vs RPM ──────────────────────────────
    sf_cent = [r.safety_factor_cent for r in results]
    sf_comb = [r.safety_factor_combined for r in results]
    ax_sf.plot(rpms, sf_cent, color="#42A5F5", linewidth=1.8,
               marker="o", markersize=4, label="SF centrifugal")
    ax_sf.plot(rpms, sf_comb, color="#EF5350", linewidth=1.8,
               marker="s", markersize=4, label="SF combined")
    ax_sf.axhline(1.0, color="#EF9A9A", linestyle="--", linewidth=1.2, label="Yield (SF=1)")
    ax_sf.axhline(2.0, color="#A5D6A7", linestyle=":", linewidth=1.0, label="SF = 2 (safe)")

    # Shade operating range
    sf_arr = np.array(sf_comb)
    safe_rpms  = rpms[sf_arr >= 2.0]
    marginal   = rpms[(sf_arr >= 1.0) & (sf_arr < 2.0)]
    if len(safe_rpms):
        ax_sf.axvspan(safe_rpms.min(), safe_rpms.max(), alpha=0.10,
                      color="#4CAF50", label="Safe zone")
    if len(marginal):
        ax_sf.axvspan(marginal.min(), marginal.max(), alpha=0.10,
                      color="#FF9800", label="Marginal zone")
    fail_r = rpms[sf_arr < 1.0]
    if len(fail_r):
        ax_sf.axvspan(fail_r.min(), fail_r.max(), alpha=0.10,
                      color="#F44336", label="Failure zone")

    ax_sf.set_ylim(0, min(max(sf_cent) * 1.15, 20))
    ax_sf.set_xlabel("RPM", color="#8899bb", fontsize=8)
    ax_sf.set_ylabel("Safety Factor", color="#8899bb", fontsize=8)
    ax_sf.set_title("Safety Factor vs RPM", color="white", fontsize=9)
    ax_sf.tick_params(colors="#8899bb", labelsize=7)
    ax_sf.legend(fontsize=6, facecolor="#0d1117", labelcolor="#ccd0d4",
                 loc="upper right", framealpha=0.7)

    # ── Right panel 2: Peak stress vs RPM ────────────────────────────────
    pk_cent = [r.peak_cent_mpa for r in results]
    pk_comb = [r.peak_combined_mpa for r in results]
    ax_st.plot(rpms, pk_cent, color="#42A5F5", linewidth=1.8,
               marker="o", markersize=4, label="σ_cent")
    ax_st.plot(rpms, pk_comb, color="#EF5350", linewidth=1.8,
               marker="s", markersize=4, label="σ_combined")
    ax_st.axhline(params.yield_mpa, color="#FFB300", linestyle="--",
                  linewidth=1.2, label=f"σ_yield={params.yield_mpa:.0f} MPa")
    ax_st.fill_between(rpms, pk_comb, params.yield_mpa,
                       where=np.array(pk_comb) >= params.yield_mpa,
                       alpha=0.15, color="red", label="Over yield")
    ax_st.set_xlabel("RPM", color="#8899bb", fontsize=8)
    ax_st.set_ylabel("Peak σ_VM (MPa)", color="#8899bb", fontsize=8)
    ax_st.set_title("Peak Stress vs RPM", color="white", fontsize=9)
    ax_st.tick_params(colors="#8899bb", labelsize=7)
    ax_st.legend(fontsize=6, facecolor="#0d1117", labelcolor="#ccd0d4",
                 loc="upper left", framealpha=0.7)

    # ── Bottom-left panel: worst-case 2-D stress heatmap ─────────────────
    # Show the combined stress at the highest RPM (worst case)
    worst = max(results, key=lambda r: r.peak_combined_mpa)
    vm_w  = worst.vm_combined
    sr_w  = (vm_w / max(params.yield_mpa, 1e-6)).clip(0.0, 1.2)

    from matplotlib.collections import PolyCollection
    tris_2d = nodes[elements]          # (M, 3, 2)
    pc2 = PolyCollection(tris_2d, array=sr_w, cmap=cmap, norm=norm,
                         edgecolors="none", antialiased=False)
    ax_xy.add_collection(pc2)

    # Draw outline
    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            x, y = p.exterior.xy
            ax_xy.plot(x, y, color="#4488cc", lw=0.6, alpha=0.7)
    else:
        x, y = poly.exterior.xy
        ax_xy.plot(x, y, color="#4488cc", lw=0.6, alpha=0.7)

    # Contact point at worst RPM
    for _ang, _r in worst.contacts:
        ax_xy.scatter([_r * np.cos(_ang)], [_r * np.sin(_ang)],
                      color="#FF6F00", s=55, zorder=10)
    ax_xy.plot(*worst.spiral_xy, color="#FFD54F", lw=1.0, alpha=0.8)

    ax_xy.set_aspect("equal")
    ax_xy.autoscale()
    ax_xy.set_title(
        f"Worst case: {int(worst.rpm)} RPM\nσ_peak={worst.peak_combined_mpa:.0f} MPa  "
        f"SF={worst.safety_factor_combined:.2f}",
        color="white", fontsize=8,
    )
    ax_xy.set_xlabel("X (mm)", color="#8899bb", fontsize=7)
    ax_xy.set_ylabel("Y (mm)", color="#8899bb", fontsize=7)
    ax_xy.tick_params(colors="#8899bb", labelsize=6)

    # ── Bottom-right panel: operating range summary table ─────────────────
    ax_tb.axis("off")
    ax_tb.set_title("Operating Range Summary", color="white", fontsize=8, pad=4)

    # Find breakpoints
    def _rpm_at_sf(sf_target: float) -> str:
        for i, r in enumerate(results):
            if r.safety_factor_combined < sf_target:
                return f"~{int(r.rpm)} RPM"
        return ">max"

    def _rpm_safe_max() -> str:
        for r in reversed(results):
            if r.safety_factor_combined >= 2.0:
                return f"{int(r.rpm)} RPM"
        return "none"

    rows = [
        ("Design RPM",   f"{int(results[0].rpm):,} RPM",),
        ("Max safe RPM", _rpm_safe_max(),),
        ("SF = 1.5 at",  _rpm_at_sf(1.5),),
        ("Yield at",     _rpm_at_sf(1.0),),
        ("Bite @ design", f"{results[0].bite_mm:.1f} mm",),
        ("Bite @ max safe", f"{next((r.bite_mm for r in results if r.safety_factor_combined >= 2.0), 0):.1f} mm",),
        ("F_impact @ max safe", f"{next((r.impact_force_N for r in results if r.safety_factor_combined >= 2.0), 0)/1000:.1f} kN",),
        ("σ_yield (material)", f"{params.yield_mpa:.0f} MPa",),
        ("Thickness", f"{params.thickness_mm:.0f} mm",),
        ("Mass", f"{params.mass_kg:.3f} kg",),
    ]

    y_pos = 0.96
    ax_tb.text(0.02, y_pos, "Parameter", transform=ax_tb.transAxes,
               color="#7ec8e3", fontsize=7, fontweight="bold", va="top")
    ax_tb.text(0.60, y_pos, "Value", transform=ax_tb.transAxes,
               color="#7ec8e3", fontsize=7, fontweight="bold", va="top")
    y_pos -= 0.05
    ax_tb.plot([0.02, 0.98], [y_pos - 0.005, y_pos - 0.005],
               transform=ax_tb.transAxes, color="#2a3a5a", linewidth=0.7,
               solid_capstyle="butt")

    row_colors = ["#c8e6c9", "#fff9c4", "#fff9c4", "#ffccbc",
                  "#e1f5fe", "#e1f5fe", "#e1f5fe",
                  "#f5f5f5", "#f5f5f5", "#f5f5f5"]
    for (label, val), rc in zip(rows, row_colors):
        y_pos -= 0.075
        ax_tb.text(0.02, y_pos, label, transform=ax_tb.transAxes,
                   color="#aac8d8", fontsize=6.5, va="top")
        ax_tb.text(0.60, y_pos, val, transform=ax_tb.transAxes,
                   color=rc, fontsize=6.5, va="top", fontweight="bold")

    # ── Master title ──────────────────────────────────────────────────────
    fig.suptitle(
        "Weapon FEA Speed Sweep — Centrifugal + Archimedean Spiral Impact Loading",
        color="white", fontsize=12, y=0.98,
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"\n[sweep] Figure saved → {out}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[SliceResult], params: SweepParams) -> None:
    W = 90
    print("\n" + "═" * W)
    print(f"  FEA Speed Sweep Summary   |  σ_yield = {params.yield_mpa:.0f} MPa  |  "
          f"thickness = {params.thickness_mm:.0f} mm  |  mass = {params.mass_kg:.3f} kg")
    print("─" * W)
    hdr = (f"{'RPM':>7}  {'ω rad/s':>8}  {'σ_cent MPa':>11}  {'SF_cent':>8}  "
           f"{'σ_comb MPa':>11}  {'SF_comb':>8}  {'F_imp kN':>9}  "
           f"{'bite mm':>8}  {'status':>10}")
    print(hdr)
    print("─" * W)
    for r in results:
        sf = r.safety_factor_combined
        status = "OK (SF≥2)" if sf >= 2.0 else ("marginal" if sf >= 1.0 else "FAILED")
        print(f"{int(r.rpm):>7}  {r.omega:>8.1f}  {r.peak_cent_mpa:>11.0f}  "
              f"{r.safety_factor_cent:>8.2f}  {r.peak_combined_mpa:>11.0f}  "
              f"{sf:>8.2f}  {r.impact_force_N/1000:>9.1f}  "
              f"{r.bite_mm:>8.2f}  {status:>10}")
    print("═" * W)

    # Identify operating range
    safe  = [r.rpm for r in results if r.safety_factor_combined >= 2.0]
    marg  = [r.rpm for r in results if 1.0 <= r.safety_factor_combined < 2.0]
    fail  = [r.rpm for r in results if r.safety_factor_combined < 1.0]
    print()
    if safe:
        print(f"  ✓ SAFE RANGE (SF≥2):    {int(min(safe)):,}–{int(max(safe)):,} RPM")
    if marg:
        print(f"  ⚠ MARGINAL (1≤SF<2):   {int(min(marg)):,}–{int(max(marg)):,} RPM")
    if fail:
        print(f"  ✗ FAILURE  (SF<1):     {int(min(fail)):,}–{int(max(fail)):,} RPM  ← STRUCTURAL LIMIT")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-D stacked FEA heatmap sweep — centrifugal + spiral-impact vs. RPM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dxf",   nargs="?", default="output_disk.dxf",
                   help="Weapon DXF file")
    p.add_argument("stats", nargs="?", default="output_disk_stats.json",
                   help="Weapon stats JSON (for material / geometry defaults)")
    p.add_argument("--rpm-min",       type=float, default=500)
    p.add_argument("--rpm-max",       type=float, default=12000)
    p.add_argument("--rpm-steps",     type=int,   default=9,
                   help="Number of RPM levels to evaluate")
    p.add_argument("--mesh-spacing",  type=float, default=5.0,
                   help="FEA element edge length (mm); smaller = finer = slower")
    p.add_argument("--drive-speed",   type=float, default=3.0,
                   help="Enemy approach speed (m/s)")
    p.add_argument("--out", default=None,
                   help="Output PNG path; if omitted, show interactive figure")
    p.add_argument("--centrifugal-only", action="store_true",
                   help="Show centrifugal stress only in 3-D view (skip impact)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load polygon ──────────────────────────────────────────────────────
    dxf_path = Path(args.dxf)
    if not dxf_path.exists():
        print(f"[error] DXF file not found: {dxf_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[sweep] Loading DXF: {dxf_path}")
    poly = load_polygon_from_dxf(dxf_path)
    print(f"[sweep] Polygon loaded: area={poly.area:.0f} mm²  "
          f"bounds={[round(x,1) for x in poly.bounds]}")

    # ── Load material / geometry from stats JSON (if present) ────────────
    params = _DEFAULT_PARAMS._asdict()
    params["mesh_spacing"] = args.mesh_spacing
    params["drive_speed_mps"] = args.drive_speed

    stats_path = Path(args.stats)
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        params["thickness_mm"] = stats.get("sheet_thickness_mm", params["thickness_mm"])
        params["mass_kg"]      = stats.get("metrics", {}).get("mass_kg", params["mass_kg"])
        mat = stats.get("material", "")
        if mat == "AR500":
            params["density_kg_m3"] = 7850.0
            params["yield_mpa"]     = 1400.0
            params["E_mpa"]         = 200_000.0
        print(f"[sweep] Stats loaded: thickness={params['thickness_mm']} mm  "
              f"mass={params['mass_kg']:.3f} kg  material={mat}")
    else:
        print(f"[sweep] Stats file not found; using defaults.")

    params = SweepParams(**params)

    # ── RPM sweep ─────────────────────────────────────────────────────────
    # Use non-uniform spacing: denser at low RPM where changes are fast
    rpms = np.unique(np.round(np.geomspace(
        args.rpm_min, args.rpm_max, args.rpm_steps
    ))).astype(float)

    print(f"[sweep] RPM range: {rpms.tolist()}")

    nodes, elements, results = run_speed_sweep(poly, rpms, params)
    print_summary(results, params)

    # ── Plot ──────────────────────────────────────────────────────────────
    out = args.out
    if out is None:
        out = "speed_sweep.png"
        print(f"[sweep] No --out specified; saving to {out}")

    plot_3d_sweep(
        poly, nodes, elements, results, params,
        output_path=out,
        show_centrifugal=args.centrifugal_only,
    )


if __name__ == "__main__":
    main()
