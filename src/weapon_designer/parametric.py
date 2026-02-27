"""Parametric weapon profile generation.

Fourier radial profiles for disk/eggbeater, rectangle + sculpted tips for bars,
and elliptical cutout parameterization.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import rotate, scale, translate
from shapely.ops import unary_union
from shapely import Point

from .config import WeaponConfig

# Angular resolution for profile discretisation
N_THETA = 360


def fourier_radius(
    theta: np.ndarray,
    r_base: float,
    coeffs_cos: np.ndarray,
    coeffs_sin: np.ndarray,
) -> np.ndarray:
    """Evaluate R(θ) = R_base + Σ(aₖcos(kθ) + bₖsin(kθ)) for k=1..N."""
    r = np.full_like(theta, r_base)
    for k in range(len(coeffs_cos)):
        r = r + coeffs_cos[k] * np.cos((k + 1) * theta)
        r = r + coeffs_sin[k] * np.sin((k + 1) * theta)
    return r


def make_disk_profile(
    r_base: float,
    coeffs_cos: np.ndarray,
    coeffs_sin: np.ndarray,
    max_radius: float,
) -> Polygon:
    """Generate a disk outer profile from Fourier coefficients.

    Radii are clamped to [0, max_radius].
    """
    theta = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    r = fourier_radius(theta, r_base, coeffs_cos, coeffs_sin)
    r = np.clip(r, 1.0, max_radius)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    coords = list(zip(x, y))
    coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def make_bar_profile(
    length: float,
    width: float,
    tip_coeffs: np.ndarray,
    max_length: float,
    max_width: float,
) -> Polygon:
    """Generate a bar profile: rectangle body with Fourier-sculpted tips.

    The bar is centred at the origin, extending ±length/2 in X, ±width/2 in Y.
    Tip sculpting applies a radial perturbation to the short ends.
    """
    length = min(length, max_length)
    width = min(width, max_width)
    half_l = length / 2
    half_w = width / 2

    # Build the basic rectangle
    n_side = 40
    n_tip = 60

    coords = []

    # Top edge (left to right)
    for x in np.linspace(-half_l, half_l, n_side):
        coords.append((x, half_w))

    # Right tip (top to bottom, sculpted)
    tip_angles = np.linspace(np.pi / 2, -np.pi / 2, n_tip)
    for i, a in enumerate(tip_angles):
        # Base radius is half_w so tip is semicircular
        r = half_w
        for k in range(len(tip_coeffs)):
            r += tip_coeffs[k] * np.cos((k + 1) * a)
        r = max(r, 2.0)
        coords.append((half_l + r * np.cos(a) - half_w * np.cos(a), r * np.sin(a)))

    # Bottom edge (right to left)
    for x in np.linspace(half_l, -half_l, n_side):
        coords.append((x, -half_w))

    # Left tip (bottom to top, sculpted)
    tip_angles = np.linspace(-np.pi / 2, np.pi / 2, n_tip)
    for i, a in enumerate(tip_angles):
        r = half_w
        for k in range(len(tip_coeffs)):
            r += tip_coeffs[k] * np.cos((k + 1) * a)
        r = max(r, 2.0)
        coords.append((-half_l + r * np.cos(a) + half_w * np.cos(a), r * np.sin(a)))

    coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def make_eggbeater_profile(
    r_base: float,
    coeffs_cos: np.ndarray,
    coeffs_sin: np.ndarray,
    num_beaters: int,
    max_radius: float,
) -> Polygon:
    """Generate an eggbeater profile: N symmetric blades from a Fourier profile.

    Each blade is the Fourier profile intersected with a sector, then unioned.
    Enforces N-fold rotational symmetry.
    """
    # Generate one blade sector
    sector_angle = 2 * np.pi / num_beaters
    blade_half_angle = sector_angle * 0.4  # 80% of sector for blade

    theta = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    r = fourier_radius(theta, r_base, coeffs_cos, coeffs_sin)
    r = np.clip(r, 1.0, max_radius)

    # Create the full profile
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    full_coords = list(zip(x, y))
    full_coords.append(full_coords[0])
    full_profile = Polygon(full_coords)
    if not full_profile.is_valid:
        full_profile = full_profile.buffer(0)

    # Create sector masks and intersect
    blades = []
    big_r = max_radius * 2
    for i in range(num_beaters):
        base_angle = sector_angle * i
        a0 = base_angle - blade_half_angle
        a1 = base_angle + blade_half_angle
        # Build a sector polygon (triangle fan)
        sector_pts = [(0, 0)]
        for a in np.linspace(a0, a1, 40):
            sector_pts.append((big_r * np.cos(a), big_r * np.sin(a)))
        sector_pts.append((0, 0))
        sector = Polygon(sector_pts)
        blade = full_profile.intersection(sector)
        if not blade.is_empty:
            blades.append(blade)

    # Add a central hub
    hub_radius = max(r_base * 0.3, 20.0)
    hub = Point(0, 0).buffer(hub_radius, resolution=64)
    blades.append(hub)

    result = unary_union(blades)
    if not result.is_valid:
        result = result.buffer(0)
    return result


def make_fourier_cutout(
    cx: float,
    cy: float,
    r_base: float,
    coeffs_cos: np.ndarray,
    coeffs_sin: np.ndarray,
    n_pts: int = 64,
) -> Polygon | None:
    """Generate a smooth closed cutout curve from Fourier terms.

    Parameters
    ----------
    cx, cy : centre of the cutout
    r_base : base radius
    coeffs_cos, coeffs_sin : Fourier amplitude coefficients (one per term)
    n_pts : angular resolution

    Returns None if the cutout is too small.
    """
    if r_base < 1.0:
        return None
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r = np.full_like(theta, r_base)
    for k in range(len(coeffs_cos)):
        r = r + coeffs_cos[k] * np.cos((k + 1) * theta)
        r = r + coeffs_sin[k] * np.sin((k + 1) * theta)
    # Clamp radius to prevent self-intersection or collapse
    r = np.clip(r, 1.0, r_base * 3.0)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    coords = list(zip(x, y))
    coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area < 1.0:
        return None
    return poly


def make_cutouts(
    params: np.ndarray,
    num_cutouts: int,
    symmetry: int = 1,
) -> list[Polygon]:
    """Generate mirrored Fourier cutout pockets.

    Each cutout is parameterized by (cx, cy, r_base, c1, s1, c2, s2, c3, s3, ...).
    params shape: (num_cutouts, 1 + 2*num_fourier_terms + 2) i.e. (num_cutouts, 7) for 3 terms
    with layout: (cx, cy, r_base, c1, s1, c2, s2).
    symmetry: number of rotational copies (1=none, 2=180° mirror, N=N-fold).
    """
    cutouts = []
    # Determine number of Fourier terms from param width: width = 3 + 2*T => T = (width-3)/2
    if num_cutouts == 0 or params.size == 0:
        return cutouts
    param_width = params.shape[1]
    n_terms = (param_width - 3) // 2

    for i in range(num_cutouts):
        row = params[i]
        cx, cy, r_base = row[0], row[1], row[2]
        coeffs_cos = row[3: 3 + n_terms]
        coeffs_sin = row[3 + n_terms: 3 + 2 * n_terms]

        cutout = make_fourier_cutout(cx, cy, r_base, coeffs_cos, coeffs_sin)
        if cutout is None:
            continue
        cutouts.append(cutout)

        # Rotational copies for symmetry
        if symmetry > 1:
            for s in range(1, symmetry):
                angle = 360.0 * s / symmetry
                rotated = rotate(cutout, angle, origin=(0, 0))
                cutouts.append(rotated)

    return cutouts


def _cutout_stride(cfg: WeaponConfig) -> int:
    """Number of parameters per cutout: cx, cy, r_base + 2*fourier_terms."""
    return 3 + 2 * cfg.optimization.num_cutout_fourier_terms


def decode_params_disk(x: np.ndarray, cfg: WeaponConfig) -> dict:
    """Decode optimizer parameter vector for disk style.

    Parameters layout:
        [0]                         : r_base
        [1 : 1+N]                   : cos coefficients
        [1+N : 1+2N]                : sin coefficients
        [1+2N : 1+2N + C*S]         : cutout params (C cutouts × S params each)
    """
    N = cfg.optimization.num_fourier_terms
    C = cfg.optimization.num_cutout_pairs
    S = _cutout_stride(cfg)

    r_base = x[0]
    cos_c = x[1 : 1 + N]
    sin_c = x[1 + N : 1 + 2 * N]
    cutout_raw = x[1 + 2 * N : 1 + 2 * N + C * S]
    cutout_params = cutout_raw.reshape(C, S) if C > 0 else np.zeros((0, S))

    return {
        "r_base": r_base,
        "coeffs_cos": cos_c,
        "coeffs_sin": sin_c,
        "cutout_params": cutout_params,
    }


def decode_params_bar(x: np.ndarray, cfg: WeaponConfig) -> dict:
    """Decode optimizer parameter vector for bar style.

    Parameters layout:
        [0]                  : length
        [1]                  : width
        [2 : 2+T]           : tip Fourier coefficients
        [2+T : 2+T + C*S]   : cutout params
    """
    N = cfg.optimization.num_fourier_terms
    C = cfg.optimization.num_cutout_pairs
    S = _cutout_stride(cfg)

    length = x[0]
    width = x[1]
    tip_c = x[2 : 2 + N]
    cutout_raw = x[2 + N : 2 + N + C * S]
    cutout_params = cutout_raw.reshape(C, S) if C > 0 else np.zeros((0, S))

    return {
        "length": length,
        "width": width,
        "tip_coeffs": tip_c,
        "cutout_params": cutout_params,
    }


def decode_params_eggbeater(x: np.ndarray, cfg: WeaponConfig) -> dict:
    """Decode optimizer parameter vector for eggbeater style.

    Parameters layout:
        [0]                          : r_base
        [1 : 1+N]                    : cos coefficients
        [1+N : 1+2N]                 : sin coefficients
        [1+2N]                       : num_beaters (rounded to int, 2-4)
        [1+2N+1 : 1+2N+1 + C*S]     : cutout params
    """
    N = cfg.optimization.num_fourier_terms
    C = cfg.optimization.num_cutout_pairs
    S = _cutout_stride(cfg)

    r_base = x[0]
    cos_c = x[1 : 1 + N]
    sin_c = x[1 + N : 1 + 2 * N]
    num_beaters = int(np.clip(np.round(x[1 + 2 * N]), 2, 4))
    cutout_raw = x[1 + 2 * N + 1 : 1 + 2 * N + 1 + C * S]
    cutout_params = cutout_raw.reshape(C, S) if C > 0 else np.zeros((0, S))

    return {
        "r_base": r_base,
        "coeffs_cos": cos_c,
        "coeffs_sin": sin_c,
        "num_beaters": num_beaters,
        "cutout_params": cutout_params,
    }


def build_weapon_polygon(x: np.ndarray, cfg: WeaponConfig) -> tuple[Polygon, dict]:
    """Build the outer profile + cutouts from a parameter vector.

    Returns (outer_profile, decoded_params).
    """
    style = cfg.weapon_style

    if style == "disk":
        params = decode_params_disk(x, cfg)
        outer = make_disk_profile(
            params["r_base"],
            params["coeffs_cos"],
            params["coeffs_sin"],
            cfg.envelope.max_radius_mm,
        )
        symmetry = 1  # disk uses CoM penalty instead
    elif style == "bar":
        params = decode_params_bar(x, cfg)
        outer = make_bar_profile(
            params["length"],
            params["width"],
            params["tip_coeffs"],
            cfg.envelope.max_length_mm,
            cfg.envelope.max_width_mm,
        )
        symmetry = 2  # 180° symmetry for bar
    elif style == "eggbeater":
        params = decode_params_eggbeater(x, cfg)
        outer = make_eggbeater_profile(
            params["r_base"],
            params["coeffs_cos"],
            params["coeffs_sin"],
            params["num_beaters"],
            cfg.envelope.max_radius_mm,
        )
        symmetry = params["num_beaters"]
    else:
        raise ValueError(f"Unknown weapon style: {style}")

    cutout_polys = make_cutouts(
        params["cutout_params"],
        cfg.optimization.num_cutout_pairs,
        symmetry=symmetry,
    )

    return outer, params, cutout_polys
