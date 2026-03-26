"""Functional / lobed profile parameterisation.

A 6-parameter analytic profile family based on radial harmonic series:

    r(θ) = R0 + A1·cos(n·(θ−φ)) + A2·cos(2n·(θ−φ)) + A3·cos(3n·(θ−φ))

where n is the (integer) tooth count, R0 is the base radius, A1/A2/A3 are
harmonic amplitudes, and φ is the angular phase.

Advantages over free B-spline radii:
- Only 6 free parameters vs. 12+ → much smoother DE landscape
- Physically meaningful: n_teeth and amplitude directly control bite
- Can be analytically seeded from spiral bite targets
- Local structure: phase sweep (24 values) pre-screens best orientations

Usage (as a parallel profile_type="functional"):
    from .functional_profiles import build_functional_profile, get_functional_bounds
    poly = build_functional_profile(params, max_r, min_r, N_ctrl)

Or for pure seed generation:
    from .functional_profiles import functional_seed_bank
    seeds = functional_seed_bank(cfg, n_range=(1, 5), k_per_n=10)
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from .config import WeaponConfig


# ---------------------------------------------------------------------------
# Core harmonic profile
# ---------------------------------------------------------------------------

def lobed_radii(
    n_teeth: int,
    R0: float,
    A1: float,
    A2: float,
    A3: float,
    phi: float,
    N_ctrl: int,
    min_r: float = 5.0,
    max_r: float = 300.0,
) -> np.ndarray:
    """Compute a radial control-point vector for a lobed profile.

    r(θ_i) = R0
           + A1·cos(n·(θ_i − φ))
           + A2·cos(2n·(θ_i − φ))
           + A3·cos(3n·(θ_i − φ))

    Control points are at θ_i = 2πi/N_ctrl, clamped to [min_r, max_r].

    Parameters
    ----------
    n_teeth : number of lobes/teeth (positive integer, ≥1)
    R0      : base radius (mm)
    A1      : fundamental amplitude (mm); positive = outward at φ
    A2      : 2nd harmonic amplitude (mm); sharpens tooth tips
    A3      : 3rd harmonic amplitude (mm); further sharpening
    phi     : angular phase offset (radians)
    N_ctrl  : number of output control points

    Returns
    -------
    radii : (N_ctrl,) float array in mm
    """
    n = max(1, int(n_teeth))
    theta = np.linspace(0.0, 2.0 * np.pi, N_ctrl, endpoint=False)
    t = n * (theta - phi)
    r = R0 + A1 * np.cos(t) + A2 * np.cos(2.0 * t) + A3 * np.cos(3.0 * t)
    return np.clip(r, min_r, max_r)


def build_functional_profile(
    params: np.ndarray,
    max_radius_mm: float,
    min_radius_mm: float,
    N_ctrl: int,
    n_eval: int = 360,
) -> Polygon | None:
    """Build a polygon from the 6-parameter functional profile vector.

    Parameter layout:
        params[0] : n_teeth_continuous  — rounded to nearest int in [1, 8]
        params[1] : R0  (mm)
        params[2] : A1  (mm)
        params[3] : A2  (mm)
        params[4] : A3  (mm)
        params[5] : phi (radians)

    The lobed_radii output is rendered via build_bspline_profile for a
    smooth, C²-continuous polygon — same pipeline as direct B-spline.

    Returns None on degenerate input (coincident points, self-intersection).
    """
    from .bspline_profile import build_bspline_profile

    if len(params) < 6:
        return None

    n_t = max(1, min(8, round(float(params[0]))))
    R0  = float(np.clip(params[1], min_radius_mm, max_radius_mm * 0.9))
    A1  = float(params[2])
    A2  = float(params[3])
    A3  = float(params[4])
    phi = float(params[5])

    radii = lobed_radii(n_t, R0, A1, A2, A3, phi, N_ctrl,
                        min_r=min_radius_mm, max_r=max_radius_mm)
    return build_bspline_profile(radii, max_radius_mm, min_radius_mm, n_eval)


# ---------------------------------------------------------------------------
# DE bounds for the 6-parameter space
# ---------------------------------------------------------------------------

def get_functional_bounds(cfg: WeaponConfig) -> list[tuple[float, float]]:
    """Differential-evolution bounds for the 6-parameter functional profile.

    Returns 6 (lo, hi) tuples: [n_teeth, R0, A1, A2, A3, phi]
    """
    if cfg.weapon_style == "bar":
        max_r = float(np.hypot(cfg.envelope.max_length_mm / 2.0,
                               cfg.envelope.max_width_mm / 2.0))
    else:
        max_r = float(cfg.envelope.max_radius_mm)

    bore_r = cfg.mounting.bore_diameter_mm / 2.0
    min_r  = max(bore_r + cfg.optimization.min_wall_thickness_mm + 5.0, 15.0)
    max_r  = max(max_r, min_r + 10.0)

    max_amp = (max_r - min_r) * 0.40   # amplitude ≤ 40 % of available range

    return [
        (1.0, 8.0),                           # n_teeth (continuous, rounded internally)
        (min_r + 5.0, max_r * 0.80),           # R0  base radius
        (-max_amp,       max_amp),             # A1  fundamental
        (-max_amp * 0.5, max_amp * 0.5),       # A2  2nd harmonic
        (-max_amp * 0.3, max_amp * 0.3),       # A3  3rd harmonic
        (0.0, 2.0 * np.pi),                    # phi angular phase
    ]


# ---------------------------------------------------------------------------
# Analytic seeding helpers
# ---------------------------------------------------------------------------

def n_for_target_bite(
    rpm: float,
    drive_speed_mps: float,
    target_bite_mm: float,
) -> float:
    """Compute the ideal tooth count to hit a target bite depth.

    From the spiral model:
        bite_mm ≈ v_per_rad × 2π / n_contacts
        n_contacts ≈ n_teeth  (one contact per tooth per revolution)

    So: n_teeth ≈ v_per_rad × 2π / target_bite
    """
    omega      = 2.0 * np.pi * max(rpm, 1.0) / 60.0
    v_per_rad  = (drive_speed_mps * 1000.0) / omega   # mm/radian
    return (v_per_rad * 2.0 * np.pi) / max(target_bite_mm, 0.1)


def functional_seed_bank(
    cfg: WeaponConfig,
    n_range: tuple[int, int] = (1, 5),
    k_per_n: int = 10,
) -> list[np.ndarray]:
    """Generate a bank of lobed radii vectors as population seeds.

    For each n in n_range (inclusive), generate k_per_n variants by:
    - Sweeping phi uniformly in [0, 2π/n]
    - Setting R0 = 70 % of max_r (generous base radius)
    - Setting A1 = 15 % of R0 (noticeable lobes), A2 = A3 = 0

    Returns a list of (N_ctrl,) radii arrays.
    """
    if cfg.weapon_style == "bar":
        max_r = float(np.hypot(cfg.envelope.max_length_mm / 2.0,
                               cfg.envelope.max_width_mm / 2.0))
    else:
        max_r = float(cfg.envelope.max_radius_mm)

    bore_r = cfg.mounting.bore_diameter_mm / 2.0
    min_r  = max(bore_r + cfg.optimization.min_wall_thickness_mm + 5.0, 15.0)
    max_r  = max(max_r, min_r + 10.0)

    N_ctrl = getattr(cfg.optimization, "n_bspline_points", 12)
    seeds: list[np.ndarray] = []

    n_lo, n_hi = int(n_range[0]), int(n_range[1])
    for n in range(n_lo, n_hi + 1):
        R0 = 0.70 * max_r
        A1 = 0.15 * R0
        phi_values = np.linspace(0.0, 2.0 * np.pi / max(n, 1), k_per_n, endpoint=False)
        for phi in phi_values:
            r = lobed_radii(n, R0, A1, 0.0, 0.0, phi, N_ctrl,
                            min_r=min_r, max_r=max_r)
            seeds.append(r)

    return seeds


def phase_sweep_seeds(
    cfg: WeaponConfig,
    n: int,
    n_phases: int = 24,
) -> np.ndarray:
    """Sweep phi for a given n_teeth, return all phase variants as a 2D array.

    Creates n_phases lobed radii variants at uniformly-spaced phi values.
    The caller can sort these by a cheap score and keep the top-k.

    Returns
    -------
    seeds : (n_phases, N_ctrl) array of radii vectors
    """
    if cfg.weapon_style == "bar":
        max_r = float(np.hypot(cfg.envelope.max_length_mm / 2.0,
                               cfg.envelope.max_width_mm / 2.0))
    else:
        max_r = float(cfg.envelope.max_radius_mm)

    bore_r = cfg.mounting.bore_diameter_mm / 2.0
    min_r  = max(bore_r + cfg.optimization.min_wall_thickness_mm + 5.0, 15.0)
    max_r  = max(max_r, min_r + 10.0)

    N_ctrl = getattr(cfg.optimization, "n_bspline_points", 12)
    R0 = 0.70 * max_r
    A1 = 0.15 * R0

    phi_values = np.linspace(0.0, 2.0 * np.pi / max(n, 1), n_phases, endpoint=False)
    seeds = np.stack([
        lobed_radii(n, R0, A1, 0.0, 0.0, phi, N_ctrl, min_r=min_r, max_r=max_r)
        for phi in phi_values
    ])
    return seeds
