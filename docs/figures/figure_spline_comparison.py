#!/usr/bin/env python3
"""Generate a publication-quality comparison figure of all profile spline families.

Illustrates three key properties that distinguish the families:
  1. Interpolation vs approximation (does the curve pass through control points?)
  2. Continuity at joins (C¹ vs C²)
  3. Local support (perturbing one control point — what region is affected?)
  4. Fourier ringing vs spline smoothness on sharp features

Output: docs/spline_comparison.png

Usage:
    cd /path/to/weapon-designer
    .venv/bin/python docs/figure_spline_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

from weapon_designer.bspline_profile import build_bspline_profile
from weapon_designer.profile_splines import build_bezier_profile, build_catmull_rom_profile


# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

COLORS = {
    "bspline":    "#2196F3",   # blue
    "bezier":     "#FF5722",   # orange-red
    "catmull_rom":"#4CAF50",   # green
    "fourier":    "#9C27B0",   # purple
    "pwlinear":   "#795548",   # brown — piecewise-linear C⁰
    "ctrl_pts":   "#333333",
    "ctrl_poly":  "#AAAAAA",
    "perturbed":  "#E91E63",   # pink — perturbed control point
    "diff_fill":  "#FFD54F",   # yellow — region of curve change
}

LABELS = {
    "bspline":    "Periodic B-Spline (C²)",
    "bezier":     "Composite Bézier (C¹)",
    "catmull_rom":"Catmull-Rom (C¹, interpolating)",
    "fourier":    "Fourier Series",
    "pwlinear":   "Piecewise-Linear (C⁰)",
}

BG     = "white"
GRID_A = "#F5F5F5"


# ---------------------------------------------------------------------------
# Helpers: control point layout
# ---------------------------------------------------------------------------

def ctrl_xy(radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian control points for N equally-spaced radii."""
    N = len(radii)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return radii * np.cos(theta), radii * np.sin(theta)


def poly_xy(poly) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract exterior coordinates from a Shapely Polygon."""
    if poly is None or poly.is_empty:
        return None
    x, y = poly.exterior.xy
    return np.array(x), np.array(y)


# ---------------------------------------------------------------------------
# Fourier profile  (fit DFT to radii, evaluate R(θ))
# ---------------------------------------------------------------------------

def build_fourier_profile_xy(
    radii: np.ndarray,
    n_terms: int,
    n_eval: int = 360,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Fourier series to radii control points and evaluate.

    Uses the DFT of the radii to extract Fourier coefficients,
    then evaluates R(θ) = a0 + Σ aₖcos(kθ) + bₖsin(kθ).
    With n_terms << N this is a low-pass approximation; with
    n_terms = N/2 it exactly interpolates.
    """
    N = len(radii)
    # DFT of radii → Fourier coefficients
    fft = np.fft.rfft(radii) / N
    theta = np.linspace(0, 2 * np.pi, n_eval, endpoint=False)
    R = np.full(n_eval, fft[0].real * 2)  # DC term (= mean radius × 2 / 2)
    R = np.full(n_eval, fft[0].real)
    for k in range(1, min(n_terms + 1, len(fft))):
        ak = 2 * fft[k].real
        bk = -2 * fft[k].imag
        R = R + ak * np.cos(k * theta) + bk * np.sin(theta * k)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    return x, y


# ---------------------------------------------------------------------------
# Panel drawing helpers
# ---------------------------------------------------------------------------

def _draw_ctrl(ax, radii, highlight_idx=None, color=COLORS["ctrl_pts"],
               poly_color=COLORS["ctrl_poly"]):
    """Draw control polygon and control points."""
    cx, cy = ctrl_xy(radii)
    # Close the polygon
    cx_c = np.append(cx, cx[0])
    cy_c = np.append(cy, cy[0])
    ax.plot(cx_c, cy_c, color=poly_color, ls="--", lw=0.9, zorder=2, alpha=0.7)
    ax.scatter(cx, cy, s=28, color=color, zorder=5, edgecolors="white", linewidths=0.5)
    if highlight_idx is not None:
        ax.scatter(cx[highlight_idx], cy[highlight_idx],
                   s=70, color=COLORS["perturbed"], zorder=6,
                   edgecolors="white", linewidths=0.8, marker="*")


def _style_ax(ax, title="", subtitle="", r_max=90):
    """Apply consistent axis styling."""
    lim = r_max * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_facecolor(GRID_A)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=4)
    if subtitle:
        ax.text(0.5, -0.04, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color="#555555",
                style="italic")
    # Light reference circle
    theta_ref = np.linspace(0, 2*np.pi, 200)
    for r in [30, 60]:
        ax.plot(r*np.cos(theta_ref), r*np.sin(theta_ref),
                color="#CCCCCC", lw=0.4, zorder=0)


def _draw_profile(ax, poly, color, lw=1.8, alpha=1.0, label=None, fill=False):
    """Draw a profile polygon on an axis."""
    xy = poly_xy(poly)
    if xy is None:
        return
    kw = dict(color=color, lw=lw, alpha=alpha, zorder=3)
    if label:
        kw["label"] = label
    ax.plot(xy[0], xy[1], **kw)
    if fill:
        ax.fill(xy[0], xy[1], color=color, alpha=0.08, zorder=1)


def _annotate_interpolation(ax, radii, poly, color, only_mismatch=True):
    """Draw small arrows from each control point to the nearest curve point,
    showing whether the curve interpolates (zero gap) or approximates (gap)."""
    if poly is None:
        return
    cx, cy = ctrl_xy(radii)
    curve_xy = np.column_stack(poly_xy(poly))
    for i, (px, py) in enumerate(zip(cx, cy)):
        dists = np.hypot(curve_xy[:, 0] - px, curve_xy[:, 1] - py)
        min_d = dists.min()
        if only_mismatch and min_d < 1.5:   # close enough — curve interpolates
            ax.plot(px, py, "o", color=color, ms=4, zorder=7, alpha=0.6)
        else:
            nearest = curve_xy[dists.argmin()]
            ax.annotate("", xy=nearest, xytext=(px, py),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=0.8, mutation_scale=6),
                        zorder=8)


# ---------------------------------------------------------------------------
# Build all profiles for a given set of radii
# ---------------------------------------------------------------------------

R_MAX = 85.0
R_MIN = 5.0


def _build_pwlinear(radii, n_eval=360):
    """Piecewise-linear profile: straight chords between N radial knots."""
    from scipy.interpolate import interp1d
    N = len(radii)
    r = np.clip(radii, R_MIN, R_MAX)
    theta_ctrl = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta_eval = np.linspace(0, 2 * np.pi, n_eval, endpoint=False)
    theta_w = np.concatenate([theta_ctrl - 2*np.pi, theta_ctrl, theta_ctrl + 2*np.pi])
    radii_w = np.tile(r, 3)
    f = interp1d(theta_w, radii_w, kind="linear")
    r_eval = f(theta_eval)
    x = r_eval * np.cos(theta_eval)
    y = r_eval * np.sin(theta_eval)
    coords = list(zip(x, y)); coords.append(coords[0])
    from shapely.geometry import Polygon
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if (not poly.is_empty and poly.area > 1.0) else None


def _build_all(radii):
    poly_bs = build_bspline_profile(radii, R_MAX, R_MIN, n_eval=360)
    poly_bz = build_bezier_profile(radii, R_MAX, R_MIN, n_eval=360)
    poly_cr = build_catmull_rom_profile(radii, R_MAX, R_MIN, n_eval=360)
    poly_pw = _build_pwlinear(radii, n_eval=360)
    return poly_bs, poly_bz, poly_cr, poly_pw


# ---------------------------------------------------------------------------
# Control point configurations
# ---------------------------------------------------------------------------

N = 10

# Config A: sharp alternating — maximises visual difference between methods
#   even indices → outer, odd → inner
radii_A = np.array([72, 30, 72, 30, 72, 30, 72, 30, 72, 30], dtype=float)

# Config B: smooth 3-lobe weapon shape
theta_ctrl = np.linspace(0, 2*np.pi, N, endpoint=False)
radii_B = 52 + 22 * np.cos(3 * theta_ctrl + np.pi/6)

# Config C: asymmetric — single dominant lobe
radii_C = np.array([75, 55, 35, 28, 32, 38, 55, 68, 72, 68], dtype=float)

# Perturbed version of C: move control point 2 outward for local-support demo
PERTURB_IDX = 2
radii_C_pert = radii_C.copy()
radii_C_pert[PERTURB_IDX] += 30


# ---------------------------------------------------------------------------
# Build profiles
# ---------------------------------------------------------------------------

bs_A, bz_A, cr_A, pw_A = _build_all(radii_A)
bs_B, bz_B, cr_B, pw_B = _build_all(radii_B)
bs_C, bz_C, cr_C, pw_C = _build_all(radii_C)
bs_Cp, bz_Cp, cr_Cp, pw_Cp = _build_all(radii_C_pert)

fx_A_lo, fy_A_lo = build_fourier_profile_xy(radii_A, n_terms=3,  n_eval=360)
fx_A_hi, fy_A_hi = build_fourier_profile_xy(radii_A, n_terms=N//2, n_eval=360)
fx_B,    fy_B    = build_fourier_profile_xy(radii_B, n_terms=4,  n_eval=360)


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
#
#   Row 0 — "Portrait" panels: one per method on Config A (sharp alternating)
#   ┌──────────┬──────────┬──────────┬──────────┬──────────┐
#   │ B-Spline │  Bézier  │ Catmull  │ Fourier  │ PW-Lin.  │
#   │  (C²)    │  (C¹)    │ (C¹,int) │ (global) │  (C⁰)    │
#   └──────────┴──────────┴──────────┴──────────┴──────────┘
#
#   Row 1 — wider panels spanning cols 0-2 and cols 2-4
#   ┌─────────────────────────┬────────────────────────────┐
#   │  Overlay (Config B)     │  Local-support demo (C)    │
#   └─────────────────────────┴────────────────────────────┘

fig = plt.figure(figsize=(18.5, 9.5), facecolor=BG)
gs = gridspec.GridSpec(
    2, 5,
    figure=fig,
    hspace=0.38, wspace=0.18,
    left=0.03, right=0.97, top=0.93, bottom=0.06,
)

# ── Row 0: per-method portraits on Config A ──────────────────────────────

ax_bs = fig.add_subplot(gs[0, 0])
ax_bz = fig.add_subplot(gs[0, 1])
ax_cr = fig.add_subplot(gs[0, 2])
ax_fo = fig.add_subplot(gs[0, 3])
ax_pw = fig.add_subplot(gs[0, 4])

for ax, poly, color, key in [
    (ax_bs, bs_A, COLORS["bspline"],     "bspline"),
    (ax_bz, bz_A, COLORS["bezier"],      "bezier"),
    (ax_cr, cr_A, COLORS["catmull_rom"], "catmull_rom"),
]:
    _style_ax(ax, title=LABELS[key])
    _draw_ctrl(ax, radii_A)
    _draw_profile(ax, poly, color, lw=2.2, fill=True)
    _annotate_interpolation(ax, radii_A, poly, color)

# Fourier panel: low vs high term count to show Gibbs
_style_ax(ax_fo, title=LABELS["fourier"])
_draw_ctrl(ax_fo, radii_A)
ax_fo.plot(fx_A_lo, fy_A_lo, color=COLORS["fourier"], lw=2.2,
           label=f"k≤3 ({3*2+1} params)", zorder=3)
ax_fo.plot(fx_A_hi, fy_A_hi, color=COLORS["fourier"], lw=1.4,
           ls="--", alpha=0.65, label=f"k≤{N//2} ({N//2*2+1} params)", zorder=3)
ax_fo.legend(fontsize=7, loc="lower right", framealpha=0.8)

# Piecewise-linear panel
_style_ax(ax_pw, title=LABELS["pwlinear"])
_draw_ctrl(ax_pw, radii_A)
_draw_profile(ax_pw, pw_A, COLORS["pwlinear"], lw=2.2, fill=True)
# Annotate kinks at every control point
cx_pw, cy_pw = ctrl_xy(radii_A)
ax_pw.scatter(cx_pw, cy_pw, s=45, color=COLORS["pwlinear"],
              zorder=6, edgecolors="white", linewidths=0.6, marker="D",
              label="kink at each ctrl pt")
ax_pw.legend(fontsize=7, loc="lower right", framealpha=0.8)

# Subtitle annotations
_SUBS = {
    ax_bs: "C² — smooth, does not pass through control pts",
    ax_bz: "C¹ — tangent-continuous, does not interpolate",
    ax_cr: "C¹ — centripetal (α=0.5), interpolates each control pt",
    ax_fo: "Global support — all harmonics affected by any feature",
    ax_pw: "C⁰ — maximum locality, kink at every control point",
}
for ax, sub in _SUBS.items():
    ax.text(0.5, -0.04, sub, transform=ax.transAxes,
            ha="center", va="top", fontsize=7.0, color="#555555", style="italic")

# Highlight: interpolation arrows for CR (green arrow = 0 gap)
ax_cr.text(0.02, 0.97, "✓ curve passes\nthrough each\ncontrol point",
           transform=ax_cr.transAxes, va="top", fontsize=7,
           color=COLORS["catmull_rom"], fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor=COLORS["catmull_rom"], alpha=0.85))

# Highlight B-spline: "approximation gap" arrows for a couple of outer points
cx_A, cy_A = ctrl_xy(radii_A)
curve_bs_xy = np.column_stack(poly_xy(bs_A)) if bs_A else None
for i in [0, 2, 4]:
    if curve_bs_xy is not None:
        dists = np.hypot(curve_bs_xy[:, 0] - cx_A[i], curve_bs_xy[:, 1] - cy_A[i])
        nearest = curve_bs_xy[dists.argmin()]
        ax_bs.annotate("",
            xy=nearest, xytext=(cx_A[i], cy_A[i]),
            arrowprops=dict(arrowstyle="-|>", color="#888888",
                            lw=0.7, mutation_scale=5),
            zorder=8)
ax_bs.text(0.02, 0.97, "≠ curve does not\npass through\ncontrol points",
           transform=ax_bs.transAxes, va="top", fontsize=7,
           color="#888888",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor="#BBBBBB", alpha=0.85))


# ── Row 1 left (cols 0-2): Overlay on Config B ───────────────────────────

ax_ov = fig.add_subplot(gs[1, 0:3])
_style_ax(ax_ov, title="Overlay — same control points, all five profile families\n"
          "(3-lobe weapon shape, N=10 control radii)", r_max=90)

_draw_ctrl(ax_ov, radii_B, color=COLORS["ctrl_pts"])

for poly, color, key in [
    (bs_B, COLORS["bspline"],     "bspline"),
    (bz_B, COLORS["bezier"],      "bezier"),
    (cr_B, COLORS["catmull_rom"], "catmull_rom"),
    (pw_B, COLORS["pwlinear"],    "pwlinear"),
]:
    _draw_profile(ax_ov, poly, color, lw=2.2, label=LABELS[key])

# Fourier overlay on Config B
ax_ov.plot(fx_B, fy_B, color=COLORS["fourier"], lw=2.2, label=LABELS["fourier"], zorder=3)

# Mark where CR actually hits control points
cx_B, cy_B = ctrl_xy(radii_B)
ax_ov.scatter(cx_B, cy_B, s=35, color=COLORS["ctrl_pts"], zorder=6,
              edgecolors="white", linewidths=0.5)

# CR interpolation markers (tick marks where CR passes through ctrl pts)
if cr_B:
    for px, py in zip(cx_B, cy_B):
        ax_ov.plot(px, py, "o", ms=7, color=COLORS["catmull_rom"],
                   alpha=0.5, zorder=4, markerfacecolor="none",
                   markeredgewidth=1.4)

legend_handles = [
    Line2D([0], [0], color=COLORS[k], lw=2.2, label=LABELS[k])
    for k in ("bspline", "bezier", "catmull_rom", "fourier", "pwlinear")
] + [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
           markeredgecolor=COLORS["catmull_rom"], markersize=9,
           markeredgewidth=1.4, label="CR interpolates ctrl pts"),
]
ax_ov.legend(handles=legend_handles, loc="lower right",
             fontsize=7.5, framealpha=0.9)

# Annotate a region where the curves diverge
ax_ov.annotate("curves diverge\nnear sharp valleys",
               xy=(-30, -70), xytext=(-75, -55),
               fontsize=7.5, color="#333333",
               arrowprops=dict(arrowstyle="-|>", color="#555555",
                               lw=0.8, mutation_scale=7),
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="#CCCCCC", alpha=0.9))


# ── Row 1 right (cols 3-4): Local-support perturbation demo ─────────────

ax_ls = fig.add_subplot(gs[1, 3:5])
_style_ax(ax_ls, title="Local support — effect of moving one control point\n"
          "(highlighted ★ = perturbed, shaded region = curve change)", r_max=90)

# Draw original and perturbed profiles for each method
for poly_orig, poly_pert, color, key in [
    (bs_C,  bs_Cp,  COLORS["bspline"],     "bspline"),
    (bz_C,  bz_Cp,  COLORS["bezier"],      "bezier"),
    (cr_C,  cr_Cp,  COLORS["catmull_rom"], "catmull_rom"),
    (pw_C,  pw_Cp,  COLORS["pwlinear"],    "pwlinear"),
]:
    # Original: thin dashed
    _draw_profile(ax_ls, poly_orig, color, lw=1.1, alpha=0.4)
    # Perturbed: solid
    _draw_profile(ax_ls, poly_pert, color, lw=2.2, label=LABELS[key])

# Shade the "changed" region between original and perturbed for B-spline
if bs_C and bs_Cp:
    # Build a fill-between using both curves
    ox, oy = poly_xy(bs_C)
    px, py = poly_xy(bs_Cp)
    # Only fill where curves differ significantly
    n = min(len(ox), len(px))
    diff = np.hypot(ox[:n] - px[:n], oy[:n] - py[:n])
    mask = diff > 2.0
    if mask.any():
        ax_ls.fill(
            np.concatenate([ox[:n][mask], px[:n][mask][::-1]]),
            np.concatenate([oy[:n][mask], py[:n][mask][::-1]]),
            color=COLORS["diff_fill"], alpha=0.25, zorder=1,
            label="Region of B-spline change",
        )

# Control points (original + highlighted perturbed point)
_draw_ctrl(ax_ls, radii_C, highlight_idx=PERTURB_IDX)

# Annotation arrow pointing to the perturbed control point
cx_C, cy_C = ctrl_xy(radii_C)
cx_Cp, cy_Cp = ctrl_xy(radii_C_pert)
ax_ls.annotate(
    f"ctrl pt {PERTURB_IDX}\nmoved outward\n+30 mm",
    xy=(cx_Cp[PERTURB_IDX], cy_Cp[PERTURB_IDX]),
    xytext=(cx_Cp[PERTURB_IDX] - 28, cy_Cp[PERTURB_IDX] - 22),
    fontsize=7.5, color=COLORS["perturbed"],
    arrowprops=dict(arrowstyle="-|>", color=COLORS["perturbed"],
                    lw=1.0, mutation_scale=7),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor=COLORS["perturbed"], alpha=0.9),
)

ls_legend = [
    Line2D([0], [0], color=COLORS[k], lw=2.2, label=LABELS[k])
    for k in ("bspline", "bezier", "catmull_rom", "pwlinear")
] + [
    Line2D([0], [0], color="#888888", lw=1.1, alpha=0.6, ls="-",
           label="original profile (dashed)"),
]
ax_ls.legend(handles=ls_legend, loc="lower right", fontsize=7.5, framealpha=0.9)


# ── Row labels ───────────────────────────────────────────────────────────

fig.text(0.005, 0.74, "A", fontsize=14, fontweight="bold", color="#333333",
         va="center", ha="left")
fig.text(0.005, 0.22, "B", fontsize=14, fontweight="bold", color="#333333",
         va="center", ha="left")
fig.text(0.025, 0.74,
         "Sharp alternating control radii (r ∈ {30, 72} mm, N=10)",
         fontsize=8, color="#555555", va="center")
fig.text(0.025, 0.22,
         "Left: smooth 3-lobe shape (all 5 families).  Right: asymmetric shape with one perturbed control point",
         fontsize=8, color="#555555", va="center")


# ── Overall title ─────────────────────────────────────────────────────────

fig.suptitle(
    "Profile Parametrisation Families — Visual Comparison",
    fontsize=13.5, fontweight="bold", y=0.97, x=0.5,
)

# ── Save ──────────────────────────────────────────────────────────────────

out = Path(__file__).parent / "spline_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved: {out}")
