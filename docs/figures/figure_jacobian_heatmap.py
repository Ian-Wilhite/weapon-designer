#!/usr/bin/env python3
"""Figure: Jacobian influence heatmap and parameter correlation matrices.

For each of the five profile families the script computes the profile Jacobian
  J  =  dp/dx   ∈ R^{A × N}   (A = N_ANGLES = 360,  N = n_ctrl = 12)
at the circular midpoint using central finite differences at h = 1 % of range,
then produces two rows of panels:

  Top row    — J heat-strip:  colour = J[θ, i] / max|J[:,i]| (diverging)
               x = parameter index i,  y = angle θ.
               Local families show bright diagonal bands;  B-spline shows
               broad oscillatory columns; Fourier shows full-height stripes.

  Bottom row — Parameter correlation:  C = J^T J  (entry-wise normalised)
               N × N matrix,  C_ij = (J[:,i]·J[:,j]) / (||J[:,i]|| ||J[:,j]||)
               Reveals aliased parameters (off-diagonal bright cells).
               B-spline: characteristic anti-diagonal (antipodal pairs).
               Bézier / Catmull-Rom: near-identity (independent parameters).

Output: docs/figures/stab_J_heatmap.png

Usage
-----
    cd /path/to/weapon-designer
    python3 docs/figures/figure_jacobian_heatmap.py
    python3 docs/figures/figure_jacobian_heatmap.py --config configs/example_bar.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent          # docs/figures/
_ROOT    = _HERE.parent.parent                       # weapon-designer/
_SRC     = _ROOT / "src"
for _p in (_SRC,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Profile builders
# ---------------------------------------------------------------------------
from weapon_designer.bspline_profile import build_bspline_profile
from weapon_designer.profile_splines import build_bezier_profile, build_catmull_rom_profile

N_ANGLES = 360


def _fourier_radial_poly(radii_arr: np.ndarray, max_r: float, min_r: float,
                          n_eval: int = 1440):
    """Build a Shapely polygon from a Fourier radial parametrisation."""
    from shapely.geometry import Polygon
    r = np.clip(radii_arr, min_r, max_r)
    N_r = len(r)
    coeffs = np.fft.rfft(r) / N_r
    theta_out = np.linspace(0, 2 * np.pi, n_eval, endpoint=False)
    r_out = np.zeros(n_eval)
    for k, c in enumerate(coeffs):
        if k == 0:
            r_out += c.real
        else:
            r_out += 2.0 * (c.real * np.cos(k * theta_out)
                             - c.imag * np.sin(k * theta_out))
    x = r_out * np.cos(theta_out)
    y = r_out * np.sin(theta_out)
    return Polygon(np.column_stack([x, y]))


def _pwlinear_poly(radii_arr: np.ndarray, max_r: float, min_r: float,
                    n_eval: int = 1440):
    """Build a Shapely polygon from a piecewise-linear radial parametrisation."""
    from shapely.geometry import Polygon
    r = np.clip(radii_arr, min_r, max_r)
    theta_ctrl = np.linspace(0, 2 * np.pi, len(r), endpoint=False)
    theta_out = np.linspace(0, 2 * np.pi, n_eval, endpoint=False)
    r_ext = np.append(r, r[0])
    t_ext = np.append(theta_ctrl, 2 * np.pi)
    r_out = np.interp(theta_out, t_ext, r_ext)
    x = r_out * np.cos(theta_out)
    y = r_out * np.sin(theta_out)
    return Polygon(np.column_stack([x, y]))


def _eval_radii(profile_type: str, x: np.ndarray, cfg,
                n_eval: int = 1440) -> np.ndarray | None:
    """Evaluate profile, return radius at N_ANGLES equally-spaced angles."""
    max_r = cfg.envelope.max_radius_mm
    min_r = 20.0
    build = {
        "bspline":        lambda: build_bspline_profile(x, max_r, min_r, n_eval),
        "bezier":         lambda: build_bezier_profile(x, max_r, min_r, n_eval),
        "catmull_rom":    lambda: build_catmull_rom_profile(x, max_r, min_r, n_eval,
                                                             alpha=0.5),
        "fourier_radial": lambda: _fourier_radial_poly(x, max_r, min_r, n_eval),
        "pwlinear":       lambda: _pwlinear_poly(x, max_r, min_r, n_eval),
    }
    try:
        poly = build[profile_type]()
    except Exception:
        return None
    if poly is None or poly.is_empty:
        return None
    coords = np.array(poly.exterior.coords[:-1])
    p_angles = np.arctan2(coords[:, 1], coords[:, 0]) % (2 * np.pi)
    p_radii  = np.hypot(coords[:, 0], coords[:, 1])
    sort_idx = np.argsort(p_angles)
    p_angles = p_angles[sort_idx]
    p_radii  = p_radii[sort_idx]
    # Periodic interpolation to N_ANGLES uniform angles
    angles_out = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    p_ang_ext = np.concatenate([p_angles - 2*np.pi, p_angles, p_angles + 2*np.pi])
    p_rad_ext = np.tile(p_radii, 3)
    return np.interp(angles_out, p_ang_ext, p_rad_ext)


def _get_bounds(profile_type: str, cfg, n_ctrl: int) -> list[tuple[float, float]]:
    max_r = cfg.envelope.max_radius_mm
    min_r = 20.0
    return [(min_r, max_r)] * n_ctrl


def finite_diff_at_h(x0: np.ndarray, h: float,
                      profile_type: str, cfg) -> dict | None:
    """Central-difference Jacobian J (N_ANGLES × N_CTRL) of radius w.r.t. x."""
    if _eval_radii(profile_type, x0, cfg) is None:
        return None
    n_ctrl = len(x0)
    J = np.zeros((N_ANGLES, n_ctrl))
    for i in range(n_ctrl):
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        rp = _eval_radii(profile_type, xp, cfg)
        rm = _eval_radii(profile_type, xm, cfg)
        if rp is None or rm is None:
            J[:, i] = 0.0
        else:
            J[:, i] = (rp - rm) / (2.0 * h)
    return {"J": J}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROFILE_TYPES = ["bspline", "bezier", "catmull_rom", "fourier_radial", "pwlinear"]
LABELS = {
    "bspline":       "B-spline (C²)",
    "bezier":        "Bézier (C¹)",
    "catmull_rom":   "Catmull-Rom (C¹)",
    "fourier_radial":"Fourier radial",
    "pwlinear":      "Piecewise-linear (C⁰)",
}
COLOURS = {
    "bspline":       "#1f77b4",
    "bezier":        "#ff7f0e",
    "catmull_rom":   "#2ca02c",
    "fourier_radial":"#d62728",
    "pwlinear":      "#9467bd",
}

N_CTRL      = 12
STEP_FRAC   = 0.01    # 1 % of parameter range


# ---------------------------------------------------------------------------
# Compute Jacobian matrix for one profile type
# ---------------------------------------------------------------------------

def compute_jacobian(profile_type: str, cfg) -> np.ndarray | None:
    """Return J  (N_ANGLES × N_CTRL) at the circular midpoint, h = STEP_FRAC."""
    bounds   = _get_bounds(profile_type, cfg, N_CTRL)
    lo       = np.array([b[0] for b in bounds])
    hi       = np.array([b[1] for b in bounds])
    x0       = (lo + hi) / 2.0
    r_range  = float(hi[0] - lo[0])
    h        = STEP_FRAC * r_range

    res = finite_diff_at_h(x0, h, profile_type, cfg)
    if res is None:
        print(f"  Warning: {profile_type} degenerate at midpoint")
        return None
    return res["J"]   # (N_ANGLES, N_CTRL)


def correlation_matrix(J: np.ndarray) -> np.ndarray:
    """Column-normalised Gram matrix  C_ij = (J[:,i]·J[:,j]) / (||i||·||j||)."""
    norms = np.linalg.norm(J, axis=0)                 # (N,)
    norms = np.where(norms < 1e-12, 1.0, norms)
    J_n   = J / norms[np.newaxis, :]                  # unit-column normalised
    return J_n.T @ J_n                                 # (N, N)


def normalised_J(J: np.ndarray) -> np.ndarray:
    """Normalise each column to [-1, 1] by its absolute maximum."""
    col_max = np.max(np.abs(J), axis=0)                # (N,)
    col_max = np.where(col_max < 1e-12, 1.0, col_max)
    return J / col_max[np.newaxis, :]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(jacobians: dict[str, np.ndarray | None], out_path: Path) -> None:
    """5 profiles arranged 3+2 (centred).  Each profile gets a J-heatmap
    (top sub-row) and a correlation matrix (bottom sub-row)."""
    from matplotlib.gridspec import GridSpec

    # 4 subplot rows (J+C for top-3, J+C for bottom-2), 6 columns so the
    # bottom pair can be centred at columns 1-2 and 3-4.
    fig = plt.figure(figsize=(3.6 * 3, 7.2 * 2))
    fig.suptitle(
        "Profile Jacobian $J = \\partial p/\\partial x$  at circular midpoint"
        f"  ($N={N_CTRL}$, $h=1\\%$ range)",
        fontsize=12, fontweight="bold", y=0.99,
    )

    gs = GridSpec(4, 6, figure=fig,
                  left=0.07, right=0.97, top=0.95, bottom=0.05,
                  hspace=0.60, wspace=0.45)

    # Map each profile index to (J-axes spec, C-axes spec)
    ax_specs = [
        (gs[0, 0:2], gs[1, 0:2]),   # profile 0  — top row
        (gs[0, 2:4], gs[1, 2:4]),   # profile 1
        (gs[0, 4:6], gs[1, 4:6]),   # profile 2
        (gs[2, 1:3], gs[3, 1:3]),   # profile 3  — bottom row centred
        (gs[2, 3:5], gs[3, 3:5]),   # profile 4
    ]

    theta_deg = np.linspace(0, 360, N_ANGLES, endpoint=False)
    cmap_J    = "RdBu_r"
    cmap_corr = "coolwarm"

    for (spec_J, spec_C), pt in zip(ax_specs, PROFILE_TYPES):
        J      = jacobians.get(pt)
        ax_J   = fig.add_subplot(spec_J)
        ax_C   = fig.add_subplot(spec_C)
        colour = COLOURS[pt]

        # ── Top: J heatmap ──────────────────────────────────────────────
        if J is not None:
            Jn = normalised_J(J)
            im = ax_J.imshow(
                Jn,
                aspect="auto",
                cmap=cmap_J,
                vmin=-1.0, vmax=1.0,
                origin="lower",
                extent=[0.5, N_CTRL + 0.5, 0, 360],
                interpolation="nearest",
            )
            ax_J.set_xlabel("Param index $i$", fontsize=7.5)
            ax_J.set_ylabel("Angle $\\theta$ (°)", fontsize=7.5)
            ax_J.set_xticks(range(1, N_CTRL + 1, 2))
            ax_J.set_yticks([0, 90, 180, 270, 360])
            ax_J.tick_params(labelsize=6.5)
            cb = fig.colorbar(im, ax=ax_J, fraction=0.046, pad=0.04)
            cb.set_ticks([-1, 0, 1])
            cb.ax.tick_params(labelsize=6)
            cb.set_label("$J_{\\theta i} / \\max|J_{:i}|$", fontsize=6.5)
        else:
            ax_J.text(0.5, 0.5, "degenerate", ha="center", va="center",
                      transform=ax_J.transAxes, fontsize=9, color="grey")

        ax_J.set_title(LABELS[pt], fontsize=8.5, fontweight="bold",
                       color=colour, pad=4)

        # ── Bottom: correlation matrix ───────────────────────────────────
        if J is not None:
            C   = correlation_matrix(J)
            im2 = ax_C.imshow(
                C,
                aspect="equal",
                cmap=cmap_corr,
                vmin=-1.0, vmax=1.0,
                origin="upper",
                extent=[0.5, N_CTRL + 0.5, N_CTRL + 0.5, 0.5],
                interpolation="nearest",
            )
            ax_C.set_xlabel("Param $i$", fontsize=7.5)
            ax_C.set_ylabel("Param $j$", fontsize=7.5)
            ax_C.set_xticks(range(1, N_CTRL + 1, 2))
            ax_C.set_yticks(range(1, N_CTRL + 1, 2))
            ax_C.tick_params(labelsize=6.5)
            cb2 = fig.colorbar(im2, ax=ax_C, fraction=0.046, pad=0.04)
            cb2.set_ticks([-1, 0, 1])
            cb2.ax.tick_params(labelsize=6)
            cb2.set_label("correlation", fontsize=6.5)
            try:
                cond = float(np.linalg.cond(J))
                kappa_str = (f"$\\kappa={cond:.0f}$" if cond < 1e6
                             else f"$\\kappa=10^{{{int(np.log10(cond))}}}$")
            except Exception:
                kappa_str = ""
            ax_C.text(0.03, 0.97, kappa_str, transform=ax_C.transAxes,
                      va="top", ha="left", fontsize=7,
                      color=colour, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor=colour, alpha=0.85))
        else:
            ax_C.text(0.5, 0.5, "degenerate", ha="center", va="center",
                      transform=ax_C.transAxes, fontsize=9, color="grey")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path,
                        default=_ROOT / "configs" / "example_disk.json")
    parser.add_argument("--out", type=Path,
                        default=_HERE / "stab_J_heatmap.png")
    args = parser.parse_args()

    from weapon_designer.config import load_config
    cfg = load_config(args.config)
    cfg.optimization.n_bspline_points = N_CTRL

    print(f"Config : {args.config}")
    print(f"N_ctrl : {N_CTRL},  step = {STEP_FRAC*100:.0f}% of range")

    jacobians: dict[str, np.ndarray | None] = {}
    for pt in PROFILE_TYPES:
        cfg.optimization.profile_type = pt
        print(f"  Computing J for {pt}...", end=" ", flush=True)
        J = compute_jacobian(pt, cfg)
        jacobians[pt] = J
        if J is not None:
            _, S, _ = np.linalg.svd(J, full_matrices=False)
            cond = float(S[0] / max(S[-1], 1e-12))
            rank = int(np.sum(S > 0.01 * S[0]))
            print(f"κ={cond:.2g}  rank={rank}/{N_CTRL}")
        else:
            print("FAILED")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_figure(jacobians, args.out)


if __name__ == "__main__":
    main()
