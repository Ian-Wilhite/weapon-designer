"""Generate parametrisation comparison figure for research poster.

5 profile families on a 3-lobe eggbeater test case (N=12 control points).
Saves to poster/assets/profile_comparison.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.interpolate import splprep, splev

# ── Project imports ──────────────────────────────────────────────────────────
from weapon_designer.config import WeaponConfig
from weapon_designer.bspline_profile import build_bspline_profile
from weapon_designer.profile_splines import build_bezier_profile, build_catmull_rom_profile

# ── Control point definition ─────────────────────────────────────────────────
N = 12
theta_ctrl = np.linspace(0, 2 * np.pi, N, endpoint=False)
radii = 60.0 + 22.0 * np.cos(3.0 * theta_ctrl)   # 3-lobe eggbeater

# Clamp to realistic bounds (consistent with build_* functions)
MAX_R = 90.0
MIN_R = 20.0
radii_clamped = np.clip(radii, MIN_R, MAX_R)

# Cartesian control points
cx = radii_clamped * np.cos(theta_ctrl)
cy = radii_clamped * np.sin(theta_ctrl)

# ── Build WeaponConfig with desired envelope ─────────────────────────────────
cfg = WeaponConfig()
cfg.envelope.max_radius_mm = MAX_R
cfg.mounting.bore_diameter_mm = 25.4
cfg.optimization.n_bspline_points = N
cfg.optimization.min_wall_thickness_mm = 5.0

# ── Piecewise-linear helper ───────────────────────────────────────────────────
def build_piecewise_linear(radii_arr, n_eval=720):
    """Wrap control radii as a closed piecewise-linear polygon."""
    r = np.clip(radii_arr, MIN_R, MAX_R)
    theta = np.linspace(0, 2 * np.pi, len(r), endpoint=False)
    # Upsample by linear interpolation between control points
    theta_out = np.linspace(0, 2 * np.pi, n_eval, endpoint=False)
    # Periodic linear interp of radius
    r_periodic = np.append(r, r[0])
    theta_periodic = np.append(theta, 2 * np.pi)
    r_interp = np.interp(theta_out, theta_periodic, r_periodic)
    x = r_interp * np.cos(theta_out)
    y = r_interp * np.sin(theta_out)
    return x, y

# ── Fourier radial series ─────────────────────────────────────────────────────
def build_fourier_radial(radii_arr, n_eval=720):
    """Fit a Fourier series to the control radii and evaluate it.

    We fit N/2 harmonics to the N uniformly-spaced radii using the DFT,
    then reconstruct on n_eval points — the standard Fourier radial
    parametrisation used in the baseline optimizer.
    """
    r = np.clip(radii_arr, MIN_R, MAX_R)
    N_r = len(r)
    coeffs = np.fft.rfft(r) / N_r          # complex DFT coefficients
    theta_out = np.linspace(0, 2 * np.pi, n_eval, endpoint=False)
    # Reconstruct via IDFT at arbitrary points
    r_out = np.zeros(n_eval)
    for k, c in enumerate(coeffs):
        if k == 0:
            r_out += c.real
        else:
            r_out += 2.0 * (c.real * np.cos(k * theta_out)
                            - c.imag * np.sin(k * theta_out))
    x = r_out * np.cos(theta_out)
    y = r_out * np.sin(theta_out)
    return x, y

# ── Evaluate all five families ────────────────────────────────────────────────
N_EVAL = 720

# 1. Fourier radial
x_fourier, y_fourier = build_fourier_radial(radii_clamped, N_EVAL)

# 2. Periodic cubic B-spline (C²)
poly_bspline = build_bspline_profile(radii_clamped, MAX_R, MIN_R, N_EVAL)
coords_bs = np.array(poly_bspline.exterior.coords)
x_bspline, y_bspline = coords_bs[:, 0], coords_bs[:, 1]

# 3. Composite cubic Bézier (C¹)
poly_bezier = build_bezier_profile(radii_clamped, MAX_R, MIN_R, N_EVAL)
coords_bz = np.array(poly_bezier.exterior.coords)
x_bezier, y_bezier = coords_bz[:, 0], coords_bz[:, 1]

# 4. Centripetal Catmull-Rom (C¹)
poly_cr = build_catmull_rom_profile(radii_clamped, MAX_R, MIN_R, N_EVAL, alpha=0.5)
coords_cr = np.array(poly_cr.exterior.coords)
x_cr, y_cr = coords_cr[:, 0], coords_cr[:, 1]

# 5. Piecewise-linear (C⁰)
x_pwl, y_pwl = build_piecewise_linear(radii_clamped, N_EVAL)

# ── Jacobian / sensitivity metrics ───────────────────────────────────────────
# Numerical Jacobian: perturb each control radius by δ, measure displacement
# of n_eval curve points (x,y stacked → 2*n_eval output vector).
# We compute ||∂P/∂r_i||₂ for each i, then form the N×(2*n_eval) Jacobian
# and report condition number κ(J).

def numerical_jacobian(build_fn, radii_ref, delta=0.5):
    """Return (2*n_eval, N) Jacobian matrix via central differences."""
    N_c = len(radii_ref)
    x0, y0 = build_fn(radii_ref)
    n_out = 2 * len(x0)
    J = np.zeros((n_out, N_c))
    for i in range(N_c):
        rp = radii_ref.copy(); rp[i] += delta
        rm = radii_ref.copy(); rm[i] -= delta
        xp, yp = build_fn(rp)
        xm, ym = build_fn(rm)
        J[:len(x0), i] = (xp - xm) / (2 * delta)
        J[len(x0):, i] = (yp - ym) / (2 * delta)
    return J


def poly_to_xy(poly):
    coords = np.array(poly.exterior.coords[:-1])
    return coords[:, 0], coords[:, 1]


def spline_build_fn(radii_arr):
    p = build_bspline_profile(radii_arr, MAX_R, MIN_R, N_EVAL)
    return poly_to_xy(p) if p else (np.zeros(N_EVAL), np.zeros(N_EVAL))


def bezier_build_fn(radii_arr):
    p = build_bezier_profile(radii_arr, MAX_R, MIN_R, N_EVAL)
    return poly_to_xy(p) if p else (np.zeros(N_EVAL), np.zeros(N_EVAL))


def cr_build_fn(radii_arr):
    p = build_catmull_rom_profile(radii_arr, MAX_R, MIN_R, N_EVAL, alpha=0.5)
    return poly_to_xy(p) if p else (np.zeros(N_EVAL), np.zeros(N_EVAL))


print("Computing Jacobians (this may take ~30 s)...")
J_fourier = numerical_jacobian(build_fourier_radial, radii_clamped)
J_bspline = numerical_jacobian(spline_build_fn, radii_clamped)
J_bezier  = numerical_jacobian(bezier_build_fn,  radii_clamped)
J_cr      = numerical_jacobian(cr_build_fn,      radii_clamped)
J_pwl     = numerical_jacobian(build_piecewise_linear, radii_clamped)

def jac_metrics(J):
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * 1e-8))
    kappa = sv[0] / sv[rank - 1] if rank > 0 and sv[rank - 1] > 0 else np.inf
    return kappa, rank

kappa_f, rank_f = jac_metrics(J_fourier)
kappa_b, rank_b = jac_metrics(J_bspline)
kappa_z, rank_z = jac_metrics(J_bezier)
kappa_c, rank_c = jac_metrics(J_cr)
kappa_p, rank_p = jac_metrics(J_pwl)

print(f"Fourier      κ={kappa_f:.1f}  rank={rank_f}")
print(f"B-spline     κ={kappa_b:.1f}  rank={rank_b}")
print(f"Bézier       κ={kappa_z:.1f}  rank={rank_z}")
print(f"Catmull-Rom  κ={kappa_c:.1f}  rank={rank_c}")
print(f"Piecewise-L  κ={kappa_p:.1f}  rank={rank_p}")

# ── Figure setup ──────────────────────────────────────────────────────────────
FONT_FAMILY = 'DejaVu Sans'
plt.rcParams.update({
    'font.family': FONT_FAMILY,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

CTRL_COLOR = '#e07b00'   # amber / orange
GRID_COLOR = '#e8e8e8'

# 2 rows: top 3, bottom 2 (centred).  Use a 6-column grid so bottom plots
# sit at columns 1-2 and 3-4, giving equal flanking white-space.
fig = plt.figure(figsize=(18, 11), dpi=150)
gs = GridSpec(2, 6, figure=fig,
              left=0.05, right=0.97, top=0.95, bottom=0.04,
              wspace=0.15, hspace=0.38)

# subplot specs: row 0 → 3 plots; row 1 → 2 centred plots
plot_specs = [
    gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],
    gs[1, 1:3], gs[1, 3:5],
]

# ── Panel data ────────────────────────────────────────────────────────────────
panels = [
    {
        'label': 'Fourier Radial',
        'cont':  'C\u221e',
        'local': 'Global support',
        'x': x_fourier, 'y': y_fourier,
        'kappa': kappa_f, 'rank': rank_f,
        'color': '#c84b6e',
    },
    {
        'label': 'Periodic B-spline',
        'cont':  'C\u00b2',
        'local': 'Local support',
        'x': x_bspline, 'y': y_bspline,
        'kappa': kappa_b, 'rank': rank_b,
        'color': '#1a6fa8',
    },
    {
        'label': 'Composite B\u00e9zier',
        'cont':  'C\u00b9',
        'local': 'Local support',
        'x': x_bezier, 'y': y_bezier,
        'kappa': kappa_z, 'rank': rank_z,
        'color': '#1a8c5e',
    },
    {
        'label': 'Catmull-Rom',
        'cont':  'C\u00b9  (centripetal)',
        'local': 'Interpolating',
        'x': x_cr, 'y': y_cr,
        'kappa': kappa_c, 'rank': rank_c,
        'color': '#7950b8',
    },
    {
        'label': 'Piecewise Linear',
        'cont':  'C\u2070',
        'local': 'Interpolating',
        'x': x_pwl, 'y': y_pwl,
        'kappa': kappa_p, 'rank': rank_p,
        'color': '#8a7a60',
    },
]

bore_r = cfg.mounting.bore_diameter_mm / 2.0
theta_ring = np.linspace(0, 2 * np.pi, 300)

# leftmost panel in each row gets y-axis labels
LEFT_COL_INDICES = {0, 3}

for idx, (p, spec) in enumerate(zip(panels, plot_specs)):
    ax = fig.add_subplot(spec)
    color = p['color']

    # ── Shaded fill ──────────────────────────────────────────────────────────
    ax.fill(p['x'], p['y'], color=color, alpha=0.12, zorder=1)

    # ── Profile outline ──────────────────────────────────────────────────────
    xc = np.append(p['x'], p['x'][0])
    yc = np.append(p['y'], p['y'][0])
    ax.plot(xc, yc, color=color, lw=2.0, zorder=3)

    # ── Control points ───────────────────────────────────────────────────────
    ax.scatter(cx, cy, s=38, color=CTRL_COLOR, zorder=5,
               edgecolors='#7a4000', linewidths=0.6)

    # ── Control polygon (dashed) ─────────────────────────────────────────────
    cx_closed = np.append(cx, cx[0])
    cy_closed = np.append(cy, cy[0])
    ax.plot(cx_closed, cy_closed, '--', color=CTRL_COLOR, lw=0.8,
            alpha=0.5, zorder=2)

    # ── Bore circle ──────────────────────────────────────────────────────────
    bx = bore_r * np.cos(theta_ring)
    by = bore_r * np.sin(theta_ring)
    ax.fill(bx, by, color='#d0d0d0', zorder=4)
    ax.plot(bx, by, color='#888888', lw=0.8, zorder=4)

    # ── Reference circles (max_r, mid) ───────────────────────────────────────
    for r_ref, ls in [(MAX_R, '--'), (60.0, ':')]:
        rx = r_ref * np.cos(theta_ring)
        ry = r_ref * np.sin(theta_ring)
        ax.plot(rx, ry, ls, color='#cccccc', lw=0.7, zorder=1)

    # ── Axes formatting ──────────────────────────────────────────────────────
    lim = 102
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.tick_params(labelsize=7, length=3, pad=2)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_xlabel('x  [mm]', fontsize=8, labelpad=2)
    if idx in LEFT_COL_INDICES:
        ax.set_ylabel('y  [mm]', fontsize=8, labelpad=2)
    else:
        ax.set_yticklabels([])

    # ── Title ────────────────────────────────────────────────────────────────
    title_str = f"{p['label']}\n{p['cont']}  |  {p['local']}"
    ax.set_title(title_str, fontsize=11, color=color, fontweight='bold',
                 pad=5, linespacing=1.4)

    # ── κ annotation ─────────────────────────────────────────────────────────
    kstr = f"κ(J) = {p['kappa']:.1f}"
    ax.text(0.97, 0.03, kstr, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8.5,
            color='#444444',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8f8f8',
                      edgecolor='#cccccc', linewidth=0.6))

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), 'assets', 'profile_comparison.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out_path}")
plt.close(fig)
