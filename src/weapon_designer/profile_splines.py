"""Alternative outer-profile parametrisations: composite Bézier and Catmull-Rom.

Both functions share the same interface as ``build_bspline_profile`` in
``bspline_profile.py`` so they are interchangeable in the optimizer:

    build_bezier_profile(radii, max_radius_mm, min_radius_mm, n_eval)
    build_catmull_rom_profile(radii, max_radius_mm, min_radius_mm, n_eval, alpha)

All units in mm.  Returns a closed Shapely ``Polygon | None``.

Why additional parametrisations?
─────────────────────────────────
The B-spline (``bspline_profile.py``) uses a global basis (all control points
influence the whole curve) and C² continuity.  For certain weapon shapes:

• **Bézier (composite cubic)** — C¹ joins, slightly sharper corners, faster
  to evaluate analytically.  Each control point pair defines exactly one cubic
  segment, giving the optimizer more local control.

• **Catmull-Rom (centripetal)** — C¹ joins, guaranteed to interpolate every
  control point (unlike B-spline which approximates), makes the parameter
  space more interpretable: r_i is the exact radius the curve passes through
  at angle θ_i.  The centripetal variant (α=0.5) avoids cusps and loops even
  for non-uniform control point spacing.

Both families avoid Gibbs oscillations (finite degree, no global frequency
content) and have local support at C¹ level.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _polar_to_cartesian(radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert uniformly-spaced radii to Cartesian control points."""
    N = len(radii)
    theta = np.linspace(0, 2.0 * np.pi, N, endpoint=False)
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    return x, y


def _make_polygon(x_eval: np.ndarray, y_eval: np.ndarray) -> Polygon | None:
    """Wrap sampled curve coordinates in a Shapely Polygon."""
    coords = list(zip(x_eval.tolist(), y_eval.tolist()))
    coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area < 1.0:
        return None
    return poly


# ---------------------------------------------------------------------------
# Composite cubic Bézier profile
# ---------------------------------------------------------------------------

def build_bezier_profile(
    radii: np.ndarray,
    max_radius_mm: float,
    min_radius_mm: float = 5.0,
    n_eval: int = 360,
) -> Polygon | None:
    """Build a closed outer-profile polygon from composite cubic Bézier curves.

    Parameters
    ----------
    radii         : 1-D array of N radial values in mm (at equally-spaced angles)
    max_radius_mm : upper clamp for all radii
    min_radius_mm : lower clamp (should be ≥ bore radius + wall)
    n_eval        : total polygon vertex count across all segments

    Returns None on degenerate input.

    Implementation notes
    ────────────────────
    N control points → N cubic Bézier segments (periodic / closed).
    Tangents at each control point are derived from the chord between the
    preceding and following control points (Catmull-Rom-style finite
    differences), scaled so the Bézier handles sit at 1/3 of the chord
    length.  This gives C¹ joins with local support.

    For segment i (from P_i to P_{i+1}):
        C1_i = P_i + (1/3) * T_i
        C2_i = P_{i+1} - (1/3) * T_{i+1}
    where T_i = 0.5 * (P_{i+1} - P_{i-1})  (central-difference tangent).
    """
    N = len(radii)
    if N < 3:
        return None

    radii = np.clip(radii, min_radius_mm, max_radius_mm)
    x, y = _polar_to_cartesian(radii)

    # Compute Catmull-Rom-style tangents for all control points (periodic)
    # T_i = 0.5 * (P_{i+1} - P_{i-1})
    x_prev = np.roll(x, 1)
    x_next = np.roll(x, -1)
    y_prev = np.roll(y, 1)
    y_next = np.roll(y, -1)
    Tx = 0.5 * (x_next - x_prev)
    Ty = 0.5 * (y_next - y_prev)

    # Evaluate all N segments
    pts_per_seg = max(2, n_eval // N)
    t_vals = np.linspace(0.0, 1.0, pts_per_seg, endpoint=False)

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for i in range(N):
        j = (i + 1) % N  # next control point index

        P0x, P0y = x[i], y[i]
        P3x, P3y = x[j], y[j]
        P1x = P0x + Tx[i] / 3.0   # first handle
        P1y = P0y + Ty[i] / 3.0
        P2x = P3x - Tx[j] / 3.0   # second handle
        P2y = P3y - Ty[j] / 3.0

        # Cubic Bézier: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        t  = t_vals
        t2 = t * t
        t3 = t2 * t
        mt  = 1.0 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        bx = mt3 * P0x + 3.0 * mt2 * t * P1x + 3.0 * mt * t2 * P2x + t3 * P3x
        by = mt3 * P0y + 3.0 * mt2 * t * P1y + 3.0 * mt * t2 * P2y + t3 * P3y

        all_x.append(bx)
        all_y.append(by)

    x_eval = np.concatenate(all_x)
    y_eval = np.concatenate(all_y)

    try:
        return _make_polygon(x_eval, y_eval)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Closed centripetal Catmull-Rom profile
# ---------------------------------------------------------------------------

def build_catmull_rom_profile(
    radii: np.ndarray,
    max_radius_mm: float,
    min_radius_mm: float = 5.0,
    n_eval: int = 360,
    alpha: float = 0.5,
) -> Polygon | None:
    """Build a closed outer-profile polygon from centripetal Catmull-Rom spline.

    Parameters
    ----------
    radii         : 1-D array of N radial values in mm (at equally-spaced angles)
    max_radius_mm : upper clamp for all radii
    min_radius_mm : lower clamp (should be ≥ bore radius + wall)
    n_eval        : total polygon vertex count across all segments
    alpha         : parameterisation exponent
                    0.0 = uniform, 0.5 = centripetal (default), 1.0 = chordal

    Returns None on degenerate input.

    Implementation notes
    ────────────────────
    Closed Catmull-Rom uses a rolling window of four control points for each
    segment (P_{i-1}, P_i, P_{i+1}, P_{i+2}).  Wrapping with modular indexing
    makes the curve periodic.

    The centripetal variant (α=0.5) spaces knots proportional to √(chord
    length), which prevents cusps and loops that the uniform variant can
    produce for widely-spaced control points.
    """
    N = len(radii)
    if N < 4:
        return None

    radii = np.clip(radii, min_radius_mm, max_radius_mm)
    x, y = _polar_to_cartesian(radii)

    def _knot_distance(p0: tuple, p1: tuple) -> float:
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        d = (dx * dx + dy * dy) ** 0.5
        return max(d ** alpha, 1e-10)

    pts_per_seg = max(2, n_eval // N)
    t_local = np.linspace(0.0, 1.0, pts_per_seg, endpoint=False)

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for i in range(N):
        # Four control points with periodic wrapping
        i0 = (i - 1) % N
        i1 = i
        i2 = (i + 1) % N
        i3 = (i + 2) % N

        P = [(x[i0], y[i0]), (x[i1], y[i1]), (x[i2], y[i2]), (x[i3], y[i3])]

        # Centripetal knot sequence
        t0 = 0.0
        t1 = t0 + _knot_distance(P[0], P[1])
        t2 = t1 + _knot_distance(P[1], P[2])
        t3 = t2 + _knot_distance(P[2], P[3])

        # Remap t_local [0,1] → [t1, t2]
        t = t1 + t_local * (t2 - t1)

        bx = np.zeros(len(t))
        by = np.zeros(len(t))

        for k, tk in enumerate(t):
            # Barry-Goldman recursive algorithm for Catmull-Rom
            def _lerp(a, b, ta, tb, tv):
                if abs(tb - ta) < 1e-12:
                    return a
                return a + (b - a) * (tv - ta) / (tb - ta)

            A1x = _lerp(P[0][0], P[1][0], t0, t1, tk)
            A1y = _lerp(P[0][1], P[1][1], t0, t1, tk)
            A2x = _lerp(P[1][0], P[2][0], t1, t2, tk)
            A2y = _lerp(P[1][1], P[2][1], t1, t2, tk)
            A3x = _lerp(P[2][0], P[3][0], t2, t3, tk)
            A3y = _lerp(P[2][1], P[3][1], t2, t3, tk)

            B1x = _lerp(A1x, A2x, t0, t2, tk)
            B1y = _lerp(A1y, A2y, t0, t2, tk)
            B2x = _lerp(A2x, A3x, t1, t3, tk)
            B2y = _lerp(A2y, A3y, t1, t3, tk)

            bx[k] = _lerp(B1x, B2x, t1, t2, tk)
            by[k] = _lerp(B1y, B2y, t1, t2, tk)

        all_x.append(bx)
        all_y.append(by)

    x_eval = np.concatenate(all_x)
    y_eval = np.concatenate(all_y)

    try:
        return _make_polygon(x_eval, y_eval)
    except Exception:
        return None
