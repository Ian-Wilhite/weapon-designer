"""B-spline outer profile parameterisation for the enhanced optimizer.

Replaces the Fourier radial profile
    R(θ) = R_base + Σ(aₖcos(kθ) + bₖsin(kθ))
with a much smoother representation: a periodic cubic B-spline through
N uniformly-spaced radial control points.

Why B-splines over Fourier
───────────────────────────
• Local support — moving one control radius r_i affects only the profile
  near angle θ_i (within roughly ±2π/N), not the entire curve.
  With Fourier, every coefficient ripples around the whole profile.

• No Gibbs phenomenon — abrupt changes between optimizer steps cannot
  excite high-frequency oscillations because the spline enforces C²
  continuity by construction.

• Interpretable parameters — r_i is literally the weapon radius at angle
  θ_i.  An engineer can read the parameter vector and immediately
  understand the shape.

• Compact search space — N ≈ 12 variables vs. 2k+1 Fourier coefficients
  for the same visual complexity.

Parameter layout
─────────────────
    [r_0, r_1, ..., r_{N-1}]

    θ_i = 2πi / N   (equally spaced, starting at 0)
    Control point i sits at (r_i·cos θ_i,  r_i·sin θ_i).

A periodic cubic B-spline (scipy.interpolate.splprep with per=True, k=3)
is fitted through these N Cartesian control points and evaluated at
n_eval uniformly-spaced parameter values to produce the final polygon.

All units in mm.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon

from .config import WeaponConfig

# Default number of B-spline control points.
N_BSPLINE_DEFAULT: int = 12


def build_bspline_profile(
    radii: np.ndarray,
    max_radius_mm: float,
    min_radius_mm: float = 5.0,
    n_eval: int = 360,
) -> Polygon | None:
    """Build a closed outer-profile polygon from B-spline radial control points.

    Parameters
    ----------
    radii         : 1-D array of N radial values in mm (at equally-spaced angles)
    max_radius_mm : upper clamp for all radii
    min_radius_mm : lower clamp (should be ≥ bore radius + wall)
    n_eval        : polygon vertex count (higher = smoother boundary)

    Returns None on degenerate input.
    """
    N = len(radii)
    if N < 4:
        return None

    radii = np.clip(radii, min_radius_mm, max_radius_mm)

    theta = np.linspace(0, 2.0 * np.pi, N, endpoint=False)
    x_ctrl = radii * np.cos(theta)
    y_ctrl = radii * np.sin(theta)

    try:
        # splprep with per=True fits a periodic (closed) parametric spline.
        # s=0 → interpolating (curve passes through all control points).
        # k=3 → cubic; requires N >= 4 for per=True.
        tck, _ = splprep([x_ctrl, y_ctrl], s=0, k=3, per=True)
    except Exception:
        return None

    u = np.linspace(0, 1, n_eval, endpoint=False)
    x_eval, y_eval = splev(u, tck)

    coords = list(zip(x_eval.tolist(), y_eval.tolist()))
    coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area < 1.0:
        return None
    return poly


def get_bspline_bounds(cfg: WeaponConfig) -> list[tuple[float, float]]:
    """Return DE bounds for the N B-spline control radii.

    All control points share the same (r_min, r_max) range:

    r_min  — bore radius + minimum-wall clearance (with absolute floor).
    r_max  — outer envelope limit.

    For bar-style weapons the bounds are set to the half-diagonal of the
    envelope rectangle; the optimizer discovers the elongated shape
    naturally, and the envelope-penalty in the objective prevents
    violations.
    """
    N = getattr(cfg.optimization, "n_bspline_points", N_BSPLINE_DEFAULT)

    bore_r   = cfg.mounting.bore_diameter_mm / 2.0
    min_wall = cfg.optimization.min_wall_thickness_mm
    r_min    = max(bore_r + min_wall + 5.0, 15.0)   # absolute floor

    if cfg.weapon_style == "bar":
        max_l = cfg.envelope.max_length_mm
        max_w = cfg.envelope.max_width_mm
        r_max = float(np.hypot(max_l / 2.0, max_w / 2.0))
    else:
        r_max = float(cfg.envelope.max_radius_mm)

    r_max = max(r_max, r_min + 10.0)   # ensure nonzero range

    return [(r_min, r_max)] * N
