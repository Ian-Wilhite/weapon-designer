"""Population seeding for the enhanced optimizer.

Provides structured initial populations that mix analytically-motivated
functional seeds with random Latin Hypercube Sampling (LHS) fill.

Setting structured_seed_frac=0.0 (the default) reproduces the current
pure-LHS behaviour exactly — no behavioural change unless opted in.

Typical usage in optimize_enhanced():
    from .seeding import mixed_init_population
    init = mixed_init_population(cfg, pop_size * n_dims, bounds,
                                 structured_frac=cfg.optimization.structured_seed_frac)
    result = differential_evolution(..., init=init)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .config import WeaponConfig


# ---------------------------------------------------------------------------
# Individual seed generators
# ---------------------------------------------------------------------------

def functional_seeds(
    cfg: WeaponConfig,
    n_range: tuple[int, int] = (1, 5),
    k_per_n: int = 10,
) -> np.ndarray:
    """Return a (N_seeds, N_ctrl) array of lobed radii vectors.

    For profile types other than "functional", the lobed radii are still
    valid as B-spline / Bézier / Catmull-Rom control points — they simply
    initialise the population near structured, physically-meaningful shapes.
    """
    from .functional_profiles import functional_seed_bank
    seeds = functional_seed_bank(cfg, n_range=n_range, k_per_n=k_per_n)
    if not seeds:
        return np.empty((0, 0))
    return np.stack(seeds)


def archetype_seeds(cfg: WeaponConfig) -> np.ndarray:
    """Resample Fourier archetypes to radii vectors for spline families.

    Evaluates each archetype's Fourier radial function at N_ctrl equally-
    spaced angles and returns the resulting radii as potential seeds.

    Returns (N_archetypes, N_ctrl) array, or empty (0, N_ctrl) if none found.
    """
    from .archetypes import seed_population_from_archetypes
    from .parametric import build_weapon_polygon

    if cfg.weapon_style == "bar":
        max_r = float(np.hypot(cfg.envelope.max_length_mm / 2.0,
                               cfg.envelope.max_width_mm / 2.0))
    else:
        max_r = float(cfg.envelope.max_radius_mm)

    bore_r = cfg.mounting.bore_diameter_mm / 2.0
    min_r  = max(bore_r + cfg.optimization.min_wall_thickness_mm + 5.0, 15.0)
    N_ctrl = getattr(cfg.optimization, "n_bspline_points", 12)

    try:
        archetype_params = seed_population_from_archetypes(cfg, n_seeds=8)
    except Exception:
        return np.empty((0, N_ctrl))

    theta = np.linspace(0.0, 2.0 * np.pi, N_ctrl, endpoint=False)
    result_rows: list[np.ndarray] = []

    for params in archetype_params:
        try:
            # Build the archetype polygon and sample its boundary radius at N_ctrl angles
            poly = build_weapon_polygon(params, cfg)
            if poly is None or poly.is_empty:
                continue
            cx, cy = poly.centroid.x, poly.centroid.y
            ext = np.array(poly.exterior.coords[:-1])
            angles = np.arctan2(ext[:, 1] - cy, ext[:, 0] - cx)
            radii  = np.hypot(ext[:, 0] - cx, ext[:, 1] - cy)

            # Sort by angle and interpolate to uniform grid
            sort_idx = np.argsort(angles)
            a_s = angles[sort_idx]
            r_s = radii[sort_idx]
            a_w = np.concatenate([a_s - 2*np.pi, a_s, a_s + 2*np.pi])
            r_w = np.tile(r_s, 3)
            a_u, uid = np.unique(a_w, return_index=True)
            r_u = r_w[uid]
            if len(a_u) < 4:
                continue
            from scipy.interpolate import interp1d
            f = interp1d(a_u, r_u, kind="linear", fill_value="extrapolate")
            r_ctrl = np.clip(f(theta - np.pi), min_r, max_r)
            result_rows.append(r_ctrl)
        except Exception:
            continue

    if not result_rows:
        return np.empty((0, N_ctrl))
    return np.stack(result_rows)


def perturb_seed(
    r: np.ndarray,
    sigma_mm: float,
    k: int,
    min_r: float = 5.0,
    max_r: float = 300.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return k Gaussian perturbations around a radii vector.

    Each perturbation is clipped to [min_r, max_r].

    Returns (k, len(r)) array.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma_mm, size=(k, len(r)))
    return np.clip(r[None, :] + noise, min_r, max_r)


# ---------------------------------------------------------------------------
# Mixed initial population
# ---------------------------------------------------------------------------

def mixed_init_population(
    cfg: WeaponConfig,
    total_pop: int,
    bounds: list[tuple[float, float]],
    structured_frac: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build an initial population mixing structured seeds and LHS fill.

    Parameters
    ----------
    cfg             : weapon config (reads n_bspline_points, n_range, etc.)
    total_pop       : total population size (= popsize × n_dims in scipy DE)
    bounds          : list of (lo, hi) per parameter
    structured_frac : fraction of total_pop to fill with structured seeds.
                      0.0 (default) = pure LHS — identical to current behaviour.
    rng             : optional random generator for reproducibility

    Returns
    -------
    init_pop : (total_pop, n_dims) array clipped to bounds, ready for
               ``differential_evolution(init=init_pop)``
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    n_dims = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # ── 1. LHS fill (always generated; structured seeds added on top) ──────
    sampler = qmc.LatinHypercube(d=n_dims, seed=rng)
    lhs = sampler.random(n=total_pop)
    lhs_scaled = qmc.scale(lhs, lo, hi)

    if structured_frac <= 0.0:
        return np.clip(lhs_scaled, lo, hi)

    # ── 2. Collect structured seeds ────────────────────────────────────────
    n_range = getattr(cfg.optimization, "structured_seed_n_range", (1, 5))
    k_per_n = max(2, int(np.ceil(total_pop * structured_frac / max(n_range[1] - n_range[0] + 1, 1))))

    fn_seeds = functional_seeds(cfg, n_range=n_range, k_per_n=k_per_n)
    ar_seeds = archetype_seeds(cfg)

    all_seeds: list[np.ndarray] = []

    for arr in [fn_seeds, ar_seeds]:
        if arr.size == 0:
            continue
        rows, cols = arr.shape
        if cols == n_dims:
            # Exact match — use directly
            all_seeds.append(np.clip(arr, lo, hi))
        elif cols < n_dims:
            # Pad with midpoints
            pad = (lo[cols:] + hi[cols:]) / 2.0
            padded = np.hstack([np.clip(arr, lo[:cols], hi[:cols]),
                                np.tile(pad, (rows, 1))])
            all_seeds.append(padded)
        else:
            # Truncate
            all_seeds.append(np.clip(arr[:, :n_dims], lo, hi))

    if not all_seeds:
        return np.clip(lhs_scaled, lo, hi)

    seeds = np.vstack(all_seeds)

    # Optionally add small perturbations around best seeds to diversify
    if len(seeds) < int(total_pop * structured_frac):
        n_extra = int(total_pop * structured_frac) - len(seeds)
        sigma = float(np.mean(hi - lo)) * 0.05   # 5 % of average range
        base = seeds[rng.integers(0, len(seeds), n_extra)]
        noise = rng.normal(0.0, sigma, size=(n_extra, n_dims))
        extras = np.clip(base + noise, lo, hi)
        seeds = np.vstack([seeds, extras])

    # ── 3. Splice seeds into the LHS population ────────────────────────────
    n_structured = min(len(seeds), int(total_pop * structured_frac))
    init_pop = lhs_scaled.copy()
    init_pop[:n_structured] = seeds[:n_structured]

    return np.clip(init_pop, lo, hi)
