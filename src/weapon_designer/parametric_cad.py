"""Backward-compatibility shim — canonical module is cutouts_superellipse.py.

Original docstring preserved below for reference.

CAD-style cutout primitives: superellipse weight-reduction pockets.

Replaces Fourier cutouts  (cx, cy, r_base, c1, s1, c2, s2, …)
with a 6-parameter engineering primitive:

    (cx, cy, a, b, n, angle)

    cx, cy  – pocket centre in mm (relative to spin axis)
    a, b    – semi-axes in mm (half-width, half-height)  — directly readable
    n       – shape exponent:  2 = ellipse, 4 = squircle, 6 ≈ rectangle
    angle   – rotation in degrees

Design intent
─────────────
• Every parameter maps to a physical feature an engineer would specify in CAD.
• Changing 'a' widens the hole, changing 'n' rounds the corners — no
  nonlinear interactions between coefficients.
• Interior fillets are implicit: n=4 gives well-rounded corners; n=6 gives
  nearly rectangular holes with tight fillets.  The optimizer can trade
  corner sharpness against hole area by tuning n.
• The shape is always valid (no self-intersection, no Gibbs ripples).

Superellipse (Lamé curve) parametric form
──────────────────────────────────────────
    x(t) = a · sgn(cos t) · |cos t|^(2/n)
    y(t) = b · sgn(sin t) · |sin t|^(2/n)

Satisfies  |x/a|^n + |y/b|^n = 1.

All units in mm.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, translate

from .config import WeaponConfig

# Number of optimizer parameters per CAD cutout.
CUTOUT_STRIDE_CAD: int = 6   # (cx, cy, a, b, n, angle)

# Number of optimizer parameters per polar cutout.
CUTOUT_STRIDE_POLAR: int = 5  # (r, phi_deg, a, b, n)
# NOTE: the superellipse orientation angle is NOT a free parameter in polar mode.
# It is analytically fixed to phi_deg + 90° (tangential to the radius vector),
# which is the structurally optimal orientation for a rotating-disk stress field.


# ---------------------------------------------------------------------------
# Primitive generators
# ---------------------------------------------------------------------------

def make_superellipse_cutout(
    cx: float,
    cy: float,
    a: float,
    b: float,
    n: float,
    angle_deg: float,
    n_pts: int = 80,
) -> Polygon | None:
    """Generate a superellipse pocket centred at (cx, cy).

    Parameters
    ----------
    a, b    : semi-axes in mm (must be > 1 mm)
    n       : shape exponent in [2, ∞).  2 = ellipse, 4 = squircle.
    angle_deg : rotation applied after generation (degrees, CCW)
    n_pts   : boundary resolution (more → smoother polygon)

    Returns None if the shape is degenerate.
    """
    if a < 1.0 or b < 1.0:
        return None

    n = max(n, 0.5)   # guard against degenerate exponent
    exp = 2.0 / n

    t = np.linspace(0, 2.0 * np.pi, n_pts, endpoint=False)
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    x = np.sign(cos_t) * np.abs(cos_t) ** exp * a
    y = np.sign(sin_t) * np.abs(sin_t) ** exp * b

    coords = list(zip(x.tolist(), y.tolist()))
    coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area < 1.0:
        return None

    if abs(angle_deg) > 0.05:
        poly = rotate(poly, angle_deg, origin=(0, 0))
    poly = translate(poly, cx, cy)
    return poly


# ---------------------------------------------------------------------------
# Multi-cutout factory (mirrors the Fourier make_cutouts() interface)
# ---------------------------------------------------------------------------

def make_cutouts_cad(
    params_flat: np.ndarray,
    num_cutouts: int,
    symmetry: int = 1,
) -> list[Polygon]:
    """Generate CAD cutout pockets from a flat parameter vector.

    Layout: CUTOUT_STRIDE_CAD (=6) values per cutout:
        [cx, cy, a, b, n, angle_deg]

    symmetry: number of rotational copies (1=none, 2=180° mirror, N=N-fold).
    """
    cutouts: list[Polygon] = []
    if num_cutouts == 0 or params_flat.size == 0:
        return cutouts

    params = params_flat.reshape(num_cutouts, CUTOUT_STRIDE_CAD)

    for row in params:
        cx, cy, a, b, n, angle_deg = row
        pocket = make_superellipse_cutout(cx, cy, a, b, n, angle_deg)
        if pocket is None:
            continue
        cutouts.append(pocket)

        if symmetry > 1:
            for s in range(1, symmetry):
                ang = 360.0 * s / symmetry
                cutouts.append(rotate(pocket, ang, origin=(0, 0)))

    return cutouts


# ---------------------------------------------------------------------------
# Parameter decoding (mirrors Fourier decode_params_* interface)
# ---------------------------------------------------------------------------

def decode_cutout_params_cad(
    x_cutout: np.ndarray,
    num_cutouts: int,
) -> np.ndarray:
    """Reshape flat cutout vector into (num_cutouts, 6) matrix."""
    if num_cutouts == 0 or x_cutout.size == 0:
        return np.zeros((0, CUTOUT_STRIDE_CAD))
    return x_cutout.reshape(num_cutouts, CUTOUT_STRIDE_CAD)


# ---------------------------------------------------------------------------
# Optimizer bounds for CAD cutouts
# ---------------------------------------------------------------------------

def get_cutout_bounds_cad(cfg: WeaponConfig) -> list[tuple[float, float]]:
    """Return DE bounds for all CAD cutout parameters.

    Each cutout: (cx, cy, a, b, n, angle)
    """
    C = cfg.optimization.num_cutout_pairs
    bounds: list[tuple[float, float]] = []

    if cfg.weapon_style == "bar":
        max_l = cfg.envelope.max_length_mm
        max_w = cfg.envelope.max_width_mm
        a_max = min(max_l, max_w) * 0.28
        b_max = min(max_l, max_w) * 0.28
        for _ in range(C):
            bounds.append((-max_l * 0.40,  max_l * 0.40))   # cx
            bounds.append((-max_w * 0.30,  max_w * 0.30))   # cy
            bounds.append((3.0, a_max))                       # a (semi-width)
            bounds.append((3.0, b_max))                       # b (semi-height)
            bounds.append((2.0, 5.5))                         # n (shape exponent)
            bounds.append((0.0, 180.0))                       # angle_deg
    else:
        max_r = cfg.envelope.max_radius_mm
        axis_max = max_r * 0.28
        for _ in range(C):
            bounds.append((-max_r * 0.65,  max_r * 0.65))   # cx
            bounds.append((-max_r * 0.65,  max_r * 0.65))   # cy
            bounds.append((3.0, axis_max))                    # a
            bounds.append((3.0, axis_max))                    # b
            bounds.append((2.0, 5.5))                         # n
            bounds.append((0.0, 180.0))                       # angle_deg

    return bounds


# ---------------------------------------------------------------------------
# Polar cutout parameterisation  (r, phi_deg, a, b, n)
# ---------------------------------------------------------------------------

def make_polar_cutout(
    r: float,
    phi_deg: float,
    a: float,
    b: float,
    n: float,
    n_pts: int = 80,
) -> Polygon | None:
    """Generate a superellipse pocket at polar position (r, φ).

    The hole centre is at  (r·cos φ, r·sin φ).
    The long axis (a) is oriented tangentially — φ + 90° — which minimises
    stress concentration in the hoop-dominated rotating-disk stress field.

    Parameters
    ----------
    r       : radial distance from spin axis in mm
    phi_deg : angular position in degrees (CCW from +x axis)
    a, b    : semi-axes in mm.  a is tangential, b is radial.
    n       : shape exponent  (2 = ellipse, 4 = squircle)

    Returns None for degenerate inputs.
    """
    if a < 1.0 or b < 1.0 or r < 0.0:
        return None

    n = max(n, 0.5)
    exp = 2.0 / n

    t = np.linspace(0, 2.0 * np.pi, n_pts, endpoint=False)
    # Build superellipse in local frame: 'a' along x, 'b' along y
    x = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** exp * a
    y = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** exp * b

    coords = list(zip(x.tolist(), y.tolist()))
    coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area < 1.0:
        return None

    # Rotate so a-axis is tangential: rotate by (phi_deg + 90°)
    poly = rotate(poly, phi_deg + 90.0, origin=(0, 0))

    # Translate to polar position
    cx = r * np.cos(np.radians(phi_deg))
    cy = r * np.sin(np.radians(phi_deg))
    poly = translate(poly, cx, cy)
    return poly


def make_cutouts_polar(
    params_flat: np.ndarray,
    num_cutouts: int,
) -> list[Polygon]:
    """Generate polar superellipse pockets from a flat parameter vector.

    Layout: CUTOUT_STRIDE_POLAR (=5) values per cutout:
        [r, phi_deg, a, b, n]

    No symmetry is assumed — each hole sits at its own (r, φ).
    Phase angles are fully free for the optimizer; radial distances
    are later normalized analytically to hit the mass budget.
    """
    cutouts: list[Polygon] = []
    if num_cutouts == 0 or params_flat.size == 0:
        return cutouts

    params = params_flat.reshape(num_cutouts, CUTOUT_STRIDE_POLAR)
    for row in params:
        r, phi_deg, a, b, n = row
        pocket = make_polar_cutout(r, phi_deg, a, b, n)
        if pocket is not None:
            cutouts.append(pocket)
    return cutouts


def get_cutout_bounds_polar(cfg: WeaponConfig) -> list[tuple[float, float]]:
    """Return DE bounds for polar cutout parameters: (r, phi_deg, a, b, n).

    r is bounded away from the bore/bolt-circle to guarantee minimum wall
    thickness analytically rather than relying on a penalty term.
    phi_deg spans [0°, 360°] — fully free, no symmetry assumed.
    """
    C = cfg.optimization.num_cutout_pairs
    bounds: list[tuple[float, float]] = []

    min_wall = cfg.optimization.min_wall_thickness_mm
    # Inner radial clearance: clear the outer bolt-circle with min wall
    bolt_outer_r = (cfg.mounting.bolt_circle_diameter_mm / 2.0
                    + cfg.mounting.bolt_hole_diameter_mm / 2.0)
    r_min = bolt_outer_r + min_wall

    if cfg.weapon_style == "bar":
        max_l  = cfg.envelope.max_length_mm
        max_w  = cfg.envelope.max_width_mm
        r_max  = np.hypot(max_l / 2.0, max_w / 2.0) * 0.70
        a_max  = min(max_l, max_w) * 0.28
        b_max  = a_max
    else:
        max_r  = cfg.envelope.max_radius_mm
        r_max  = max_r * 0.72          # leave ~28% for outer wall + min_wall
        a_max  = max_r * 0.28
        b_max  = a_max

    r_min = max(r_min, 5.0)            # absolute floor
    r_max = max(r_max, r_min + 5.0)    # ensure range is nonzero

    for _ in range(C):
        bounds.append((r_min,   r_max))   # r
        bounds.append((0.0,   360.0))     # phi_deg  — full circle, no symmetry
        bounds.append((3.0,   a_max))     # a (tangential semi-axis)
        bounds.append((3.0,   b_max))     # b (radial semi-axis)
        bounds.append((2.0,     5.5))     # n (shape exponent)

    return bounds


# ---------------------------------------------------------------------------
# Analytical mass normalisation
# ---------------------------------------------------------------------------

def mass_normalize_cutouts(
    cutout_params_flat: np.ndarray,
    num_cutouts: int,
    weapon_area_mm2: float,
    no_cutout_area_mm2: float,
    cfg: WeaponConfig,
    max_scale: float = 2.8,
    min_scale: float = 0.15,
) -> tuple[np.ndarray, float]:
    """Scale hole semi-axes (a, b) analytically to hit the mass budget.

    After Phase 2, the optimizer has found good hole *positions* and *shapes*
    but the total material removed is only approximately correct.  This
    function computes a single uniform scale factor  s  and applies it to
    every hole's  a  and  b  dimensions so that the resulting weapon mass
    equals cfg.weight_budget_kg exactly (to first order).

    Works for both polar  (stride=5: r, phi, a, b, n)
            and Cartesian (stride=6: cx, cy, a, b, n, angle) layouts.

    The positions (r/φ or cx/cy), shape exponent (n), and orientation
    (angle) are all left untouched — only sizes change.

    Parameters
    ----------
    cutout_params_flat   : flat Phase-2 optimizer parameter vector
    num_cutouts          : number of holes  (C)
    weapon_area_mm2      : Shapely area of the assembled weapon WITH holes (mm²)
    no_cutout_area_mm2   : Shapely area of the weapon WITHOUT weight-reduction
                           holes (bore + bolt holes only) (mm²)
    cfg                  : weapon configuration (supplies ρ, t, budget)

    Returns
    -------
    (scaled_params_flat, scale_factor s)
    s > 1 → holes were grown (weapon was too heavy)
    s < 1 → holes were shrunk (weapon was too light)
    s = 1 → mass was already on target
    """
    if num_cutouts == 0 or cutout_params_flat.size == 0:
        return cutout_params_flat, 1.0

    stride = cutout_params_flat.size // num_cutouts

    # Effective area removed by weight-reduction holes (accounts for
    # overlaps and partial boundary clips via Shapely computation).
    A_holes_eff = no_cutout_area_mm2 - weapon_area_mm2

    if A_holes_eff <= 0.0:
        # No material is being removed yet — can't divide by zero.
        return cutout_params_flat, 1.0

    # Target solid area from mass budget.
    # ρ [kg/m³] × 1e-9 → [kg/mm³];  × t [mm] → [kg/mm²];  / budget [kg] → [mm²]
    density_kg_mm3 = cfg.material.density_kg_m3 * 1e-9
    A_target_solid = cfg.weight_budget_kg / (density_kg_mm3 * cfg.sheet_thickness_mm)
    A_holes_target = no_cutout_area_mm2 - A_target_solid

    if A_holes_target <= 0.0:
        # Budget exceeds the mass of a solid weapon — no holes needed.
        return cutout_params_flat, 0.0

    # Clamp target to a physically realizable fraction.
    A_holes_target = min(A_holes_target, no_cutout_area_mm2 * 0.92)

    # Each hole's area ∝ a·b·f(n), so scaling both a and b by s scales
    # each hole's area by s².  Total effective area scales approximately as s²
    # (first-order; overlap geometry changes slightly but the effect is small).
    s = float(np.clip(np.sqrt(A_holes_target / A_holes_eff), min_scale, max_scale))

    params = cutout_params_flat.reshape(num_cutouts, stride).copy()

    if stride == CUTOUT_STRIDE_POLAR:      # (r, phi, a, b, n)
        params[:, 2] *= s   # a
        params[:, 3] *= s   # b
    elif stride == CUTOUT_STRIDE_CAD:      # (cx, cy, a, b, n, angle)
        params[:, 2] *= s   # a
        params[:, 3] *= s   # b
    else:
        # Unknown stride — return unchanged
        return cutout_params_flat, 1.0

    return params.reshape(-1), s
