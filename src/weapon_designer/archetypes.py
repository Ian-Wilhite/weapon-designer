"""Scale-agnostic weapon archetype library.

Each archetype stores normalized Fourier coefficients (unit radius) that
represent a classic weapon shape.  Before optimization the coefficients are
scaled to the target envelope so the optimizer starts near a proven design.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .config import WeaponConfig


@dataclass(frozen=True)
class Archetype:
    """A single normalized weapon template."""

    name: str
    style: str  # "disk", "bar", "eggbeater"
    description: str
    # Normalized to unit radius / unit length
    r_base: float = 0.7
    coeffs_cos: tuple[float, ...] = ()
    coeffs_sin: tuple[float, ...] = ()
    # Bar-specific
    length: float = 0.0
    width: float = 0.0
    tip_coeffs: tuple[float, ...] = ()
    # Eggbeater-specific
    num_beaters: int = 3


# ---------------------------------------------------------------------------
# Disk archetypes  (normalised to unit radius)
# ---------------------------------------------------------------------------

DISK_SOLID = Archetype(
    name="solid_disk",
    style="disk",
    description="Plain circular disk, maximum MOI for a given radius",
    r_base=0.90,
    coeffs_cos=(0.0, 0.0, 0.0, 0.0, 0.0),
    coeffs_sin=(0.0, 0.0, 0.0, 0.0, 0.0),
)

DISK_2TOOTH = Archetype(
    name="2tooth_disk",
    style="disk",
    description="Two-pronged disk with opposing impact teeth",
    r_base=0.60,
    coeffs_cos=(0.25, 0.0, 0.0, 0.0, 0.0),
    coeffs_sin=(0.0, 0.0, 0.0, 0.0, 0.0),
)

DISK_3TOOTH = Archetype(
    name="3tooth_disk",
    style="disk",
    description="Three-pronged disk for higher bite rate",
    r_base=0.55,
    coeffs_cos=(0.0, 0.0, 0.25, 0.0, 0.0),
    coeffs_sin=(0.0, 0.0, 0.0, 0.0, 0.0),
)

DISK_4TOOTH = Archetype(
    name="4tooth_disk",
    style="disk",
    description="Four-pronged disk, balanced bite and MOI",
    r_base=0.55,
    coeffs_cos=(0.0, 0.0, 0.0, 0.20, 0.0),
    coeffs_sin=(0.0, 0.0, 0.0, 0.0, 0.0),
)

DISK_ASYMMETRIC = Archetype(
    name="asymmetric_disk",
    style="disk",
    description="Single heavy tooth for maximum single-hit energy",
    r_base=0.55,
    coeffs_cos=(0.30, 0.10, 0.0, 0.0, 0.0),
    coeffs_sin=(0.05, 0.0, 0.0, 0.0, 0.0),
)

DISK_SHELL = Archetype(
    name="shell_disk",
    style="disk",
    description="Full-diameter shell spinner, high MOI ring shape",
    r_base=0.92,
    coeffs_cos=(0.0, 0.02, 0.0, 0.0, 0.0),
    coeffs_sin=(0.0, 0.0, 0.0, 0.0, 0.0),
)

# ---------------------------------------------------------------------------
# Bar archetypes  (normalised to unit length / unit width)
# ---------------------------------------------------------------------------

BAR_STANDARD = Archetype(
    name="standard_bar",
    style="bar",
    description="Classic rectangular bar spinner",
    length=0.85,
    width=0.70,
    tip_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
)

BAR_WIDE = Archetype(
    name="wide_bar",
    style="bar",
    description="Wide bar for increased MOI",
    length=0.80,
    width=0.95,
    tip_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
)

BAR_TAPERED = Archetype(
    name="tapered_bar",
    style="bar",
    description="Bar with tapered impact tips for concentrated hits",
    length=0.90,
    width=0.60,
    tip_coeffs=(0.15, -0.05, 0.0, 0.0, 0.0),
)

BAR_TOMBSTONE = Archetype(
    name="tombstone_bar",
    style="bar",
    description="Tombstone-style bar with bulbous striking ends",
    length=0.85,
    width=0.55,
    tip_coeffs=(0.25, 0.0, 0.05, 0.0, 0.0),
)

# ---------------------------------------------------------------------------
# Eggbeater archetypes  (normalised to unit radius)
# ---------------------------------------------------------------------------

EGGBEATER_2BLADE = Archetype(
    name="2blade_eggbeater",
    style="eggbeater",
    description="Two-blade eggbeater, high bite with two impacts per revolution",
    r_base=0.75,
    coeffs_cos=(0.15, 0.0, 0.0, 0.0, 0.0),
    coeffs_sin=(0.0, 0.0, 0.0, 0.0, 0.0),
    num_beaters=2,
)

EGGBEATER_3BLADE = Archetype(
    name="3blade_eggbeater",
    style="eggbeater",
    description="Three-blade eggbeater, balanced bite rate and MOI",
    r_base=0.70,
    coeffs_cos=(0.0, 0.0, 0.20, 0.0, 0.0),
    coeffs_sin=(0.0, 0.0, 0.0, 0.0, 0.0),
    num_beaters=3,
)

EGGBEATER_4BLADE = Archetype(
    name="4blade_eggbeater",
    style="eggbeater",
    description="Four-blade eggbeater, maximum bite rate",
    r_base=0.65,
    coeffs_cos=(0.0, 0.0, 0.0, 0.18, 0.0),
    coeffs_sin=(0.0, 0.0, 0.0, 0.0, 0.0),
    num_beaters=4,
)

EGGBEATER_WIDE = Archetype(
    name="wide_eggbeater",
    style="eggbeater",
    description="Wide-blade eggbeater for higher MOI per blade",
    r_base=0.80,
    coeffs_cos=(0.10, 0.0, 0.0, 0.0, 0.0),
    coeffs_sin=(0.05, 0.0, 0.0, 0.0, 0.0),
    num_beaters=3,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_ARCHETYPES: dict[str, list[Archetype]] = {
    "disk": [DISK_SOLID, DISK_2TOOTH, DISK_3TOOTH, DISK_4TOOTH,
             DISK_ASYMMETRIC, DISK_SHELL],
    "bar": [BAR_STANDARD, BAR_WIDE, BAR_TAPERED, BAR_TOMBSTONE],
    "eggbeater": [EGGBEATER_2BLADE, EGGBEATER_3BLADE, EGGBEATER_4BLADE,
                  EGGBEATER_WIDE],
}


def get_archetypes(style: str) -> list[Archetype]:
    """Return all archetypes for a given weapon style."""
    return ALL_ARCHETYPES.get(style, [])


def _pad_or_trim(coeffs: tuple[float, ...], n: int) -> np.ndarray:
    """Pad with zeros or trim to length *n*."""
    arr = np.array(coeffs, dtype=float)
    if len(arr) >= n:
        return arr[:n]
    return np.pad(arr, (0, n - len(arr)))


def archetype_to_params(
    arch: Archetype,
    cfg: WeaponConfig,
    profile_only: bool = False,
) -> np.ndarray:
    """Scale an archetype to the target config and return a parameter vector.

    Parameters
    ----------
    arch : the archetype template
    cfg : weapon configuration for scaling
    profile_only : if True, return only profile params (no cutout zeros).
                   Used for Phase 1 seeding.
    """
    N = cfg.optimization.num_fourier_terms
    C = cfg.optimization.num_cutout_pairs
    T = cfg.optimization.num_cutout_fourier_terms
    S = 3 + 2 * T  # params per cutout: cx, cy, r_base + 2*T fourier coeffs
    style = cfg.weapon_style
    cutout_part = [] if profile_only else [np.zeros(C * S)]

    if style == "disk":
        max_r = cfg.envelope.max_radius_mm
        r_base = arch.r_base * max_r
        cos_c = _pad_or_trim(arch.coeffs_cos, N) * max_r
        sin_c = _pad_or_trim(arch.coeffs_sin, N) * max_r
        parts = [[r_base], cos_c, sin_c] + cutout_part

    elif style == "bar":
        max_l = cfg.envelope.max_length_mm
        max_w = cfg.envelope.max_width_mm
        length = arch.length * max_l
        width = arch.width * max_w
        tip_c = _pad_or_trim(arch.tip_coeffs, N) * max_w
        parts = [[length, width], tip_c] + cutout_part

    elif style == "eggbeater":
        max_r = cfg.envelope.max_radius_mm
        r_base = arch.r_base * max_r
        cos_c = _pad_or_trim(arch.coeffs_cos, N) * max_r
        sin_c = _pad_or_trim(arch.coeffs_sin, N) * max_r
        parts = [[r_base], cos_c, sin_c, [float(arch.num_beaters)]] + cutout_part

    else:
        raise ValueError(f"Unknown weapon style: {style}")

    return np.concatenate(parts)


def seed_population_from_archetypes(
    cfg: WeaponConfig,
    bounds: list[tuple[float, float]],
    pop_size: int,
    rng: np.random.Generator | None = None,
    profile_only: bool = False,
) -> np.ndarray:
    """Build an initial population seeded from archetypes.

    For each archetype matching the weapon style, the base parameter vector
    is included along with perturbed copies.  Remaining population slots are
    filled with Latin-hypercube-style random samples.

    Parameters
    ----------
    profile_only : if True, generate profile-only param vectors (for Phase 1).

    Returns array of shape ``(pop_size, ndim)``.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ndim = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    span = upper - lower

    archetypes = get_archetypes(cfg.weapon_style)
    seeds: list[np.ndarray] = []

    for arch in archetypes:
        base = archetype_to_params(arch, cfg, profile_only=profile_only)
        base = np.clip(base, lower, upper)
        seeds.append(base)
        # Add perturbed copies
        for _ in range(2):
            noise = rng.normal(scale=0.03, size=ndim) * span
            perturbed = np.clip(base + noise, lower, upper)
            seeds.append(perturbed)

    # Fill remaining slots with random samples
    n_remaining = max(0, pop_size - len(seeds))
    if n_remaining > 0:
        random_pop = lower + rng.random((n_remaining, ndim)) * span
        seeds.extend(list(random_pop))

    pop = np.array(seeds[:pop_size])
    return pop
