"""Two-phase differential evolution optimizer for weapon profiles.

Phase 1: Optimize the outer profile (Fourier coefficients only, no cutouts).
Phase 2: Fix the outer profile and optimize cutout placement to hit mass budget.

Both phases use process-based parallelism via multiprocessing.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from .config import WeaponConfig
from .parametric import (
    build_weapon_polygon,
    make_disk_profile,
    make_bar_profile,
    make_eggbeater_profile,
    make_cutouts,
    decode_params_disk,
    decode_params_bar,
    decode_params_eggbeater,
    _cutout_stride,
)
from .geometry import assemble_weapon
from .objectives import compute_metrics, weighted_score, impact_zone_score
from .constraints import validate_geometry, constraint_penalty
from .archetypes import seed_population_from_archetypes


# ---------------------------------------------------------------------------
# Bounds helpers
# ---------------------------------------------------------------------------

def _get_profile_bounds(cfg: WeaponConfig) -> list[tuple[float, float]]:
    """Parameter bounds for Phase 1 (outer profile only, no cutout params)."""
    N = cfg.optimization.num_fourier_terms
    max_r = cfg.envelope.max_radius_mm
    fourier_amp = max_r * 0.3

    bounds: list[tuple[float, float]] = []

    if cfg.weapon_style == "disk":
        bounds.append((max_r * 0.3, max_r * 0.95))
        for _ in range(N):
            bounds.append((-fourier_amp, fourier_amp))
        for _ in range(N):
            bounds.append((-fourier_amp, fourier_amp))

    elif cfg.weapon_style == "bar":
        max_l = cfg.envelope.max_length_mm
        max_w = cfg.envelope.max_width_mm
        tip_amp = max_w * 0.3
        bounds.append((max_l * 0.4, max_l))
        bounds.append((max_w * 0.3, max_w))
        for _ in range(N):
            bounds.append((-tip_amp, tip_amp))

    elif cfg.weapon_style == "eggbeater":
        bounds.append((max_r * 0.3, max_r * 0.95))
        for _ in range(N):
            bounds.append((-fourier_amp, fourier_amp))
        for _ in range(N):
            bounds.append((-fourier_amp, fourier_amp))
        bounds.append((2.0, 4.0))

    return bounds


def _get_cutout_bounds(cfg: WeaponConfig) -> list[tuple[float, float]]:
    """Parameter bounds for Phase 2 (cutout placement only).

    Each cutout: (cx, cy, r_base, c1, s1, c2, s2, ...) — 3 + 2*T params.
    """
    C = cfg.optimization.num_cutout_pairs
    T = cfg.optimization.num_cutout_fourier_terms
    max_r = cfg.envelope.max_radius_mm
    bounds: list[tuple[float, float]] = []

    if cfg.weapon_style == "bar":
        max_l = cfg.envelope.max_length_mm
        max_w = cfg.envelope.max_width_mm
        r_base_max = min(max_l, max_w) * 0.25
        for _ in range(C):
            bounds.append((-max_l * 0.4, max_l * 0.4))     # cx
            bounds.append((-max_w * 0.3, max_w * 0.3))     # cy
            bounds.append((3.0, r_base_max))                 # r_base
            for _ in range(T):
                bounds.append((-r_base_max * 0.5, r_base_max * 0.5))  # cos coeff
                bounds.append((-r_base_max * 0.5, r_base_max * 0.5))  # sin coeff
    else:
        r_base_max = max_r * 0.25
        for _ in range(C):
            bounds.append((-max_r * 0.7, max_r * 0.7))     # cx
            bounds.append((-max_r * 0.7, max_r * 0.7))     # cy
            bounds.append((3.0, r_base_max))                 # r_base
            for _ in range(T):
                bounds.append((-r_base_max * 0.5, r_base_max * 0.5))  # cos coeff
                bounds.append((-r_base_max * 0.5, r_base_max * 0.5))  # sin coeff

    return bounds


def _get_full_bounds(cfg: WeaponConfig) -> list[tuple[float, float]]:
    """Full parameter bounds (profile + cutouts) for legacy single-phase."""
    return _get_profile_bounds(cfg) + _get_cutout_bounds(cfg)


# ---------------------------------------------------------------------------
# Phase 1: Profile objective
# ---------------------------------------------------------------------------

def _profile_objective(x: np.ndarray, cfg: WeaponConfig) -> float:
    """Objective for Phase 1: optimize outer profile shape only.

    Builds the profile without cutouts, assembles with mounting holes,
    and scores MOI, bite, balance, and envelope compliance.
    Mass is not penalised hard here — cutouts handle that in Phase 2.
    """
    try:
        # Build full param vector with zero cutouts
        C = cfg.optimization.num_cutout_pairs
        S = _cutout_stride(cfg)
        x_full = np.concatenate([x, np.zeros(C * S)])

        outer, params, cutout_polys = build_weapon_polygon(x_full, cfg)
        weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
        weapon = validate_geometry(weapon)

        if weapon.is_empty or weapon.area < 1.0:
            return 1.0

        metrics = compute_metrics(weapon, cfg)

        # Phase 1 scoring: emphasise MOI, bite, balance, impact; soft mass penalty
        w = cfg.optimization.weights
        max_r = cfg.envelope.max_radius_mm
        max_moi = 0.5 * cfg.weight_budget_kg * (max_r ** 2)
        moi_score = min(metrics["moi_kg_mm2"] / max(max_moi, 1e-6), 1.0)

        ideal_bite = 20.0
        bite_score = 1.0 - min(abs(metrics["bite_mm"] - ideal_bite) / ideal_bite, 1.0)

        balance_score = max(0.0, 1.0 - metrics["com_offset_mm"] / max(max_r * 0.1, 1.0))

        struct_score = metrics["structural_integrity"]

        iz_score = metrics.get("impact_zone", 0.0)

        # Soft mass penalty: prefer designs near or above budget (cutouts will trim)
        mu = metrics["mass_utilization"]
        if mu < 0.5:
            mass_score = mu  # too light even before cutouts is bad
        elif mu > 2.0:
            mass_score = max(0.0, 1.0 - (mu - 2.0))  # way too heavy
        else:
            mass_score = 1.0  # fine range for pre-cutout

        score = (
            w.moment_of_inertia * moi_score
            + w.bite * bite_score
            + w.structural_integrity * struct_score
            + w.mass_utilization * mass_score * 0.5  # down-weight mass in phase 1
            + w.balance * balance_score
            + w.impact_zone * iz_score
        )

        # Envelope penalty
        from .constraints import check_envelope
        if not check_envelope(weapon, cfg):
            score *= 0.3

        return -score

    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Phase 2: Cutout objective
# ---------------------------------------------------------------------------

def _cutout_objective(
    x_cutout: np.ndarray,
    profile_params: np.ndarray,
    cfg: WeaponConfig,
) -> float:
    """Objective for Phase 2: optimize cutout placement with fixed profile.

    Focuses on hitting mass budget and structural integrity.
    """
    try:
        x_full = np.concatenate([profile_params, x_cutout])
        outer, params, cutout_polys = build_weapon_polygon(x_full, cfg)
        weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
        weapon = validate_geometry(weapon)

        if weapon.is_empty or weapon.area < 1.0:
            return 1.0

        metrics = compute_metrics(weapon, cfg)
        score = weighted_score(metrics, cfg)
        penalty = constraint_penalty(weapon, cfg, cutout_polys=cutout_polys)

        return -(score * penalty)

    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Population seeding
# ---------------------------------------------------------------------------

def _build_init_population(resume_params, bounds, popsize):
    """Create an initial population for DE given previous best params."""
    if resume_params is None:
        return "latinhypercube"

    seed = np.asarray(resume_params, dtype=float)
    ndim = len(bounds)
    if seed.shape != (ndim,):
        raise ValueError("resume_params must be 1D with length equal to bounds")

    n_pop = popsize * ndim
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.02, size=(n_pop, ndim))
    span = np.array([b[1] - b[0] for b in bounds])
    pop = seed + noise * (span / 2)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    pop = np.clip(pop, lower, upper)
    pop[0] = seed
    return pop


def _build_archetype_population(cfg, bounds, popsize):
    """Build initial population seeded from archetype library."""
    return seed_population_from_archetypes(
        cfg, bounds, popsize, rng=np.random.default_rng(42),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize(cfg: WeaponConfig, verbose: bool = True, resume_params=None) -> dict:
    """Run two-phase optimization of the weapon profile.

    Phase 1: Optimize outer profile shape (archetype-seeded).
    Phase 2: Optimize cutout placement with fixed profile.

    Returns a dict with best params, polygon, metrics, and score.
    """
    profile_bounds = _get_profile_bounds(cfg)
    cutout_bounds = _get_cutout_bounds(cfg)
    n_profile = len(profile_bounds)

    # Determine worker count: use multiprocessing for parallelism
    import os
    workers = max(1, os.cpu_count() or 1)

    # ---- Phase 1: Outer profile ----
    if verbose:
        print(f"=== Phase 1: Profile Optimization ===")
        print(f"  Style: {cfg.weapon_style}")
        print(f"  Parameters: {n_profile}")
        print(f"  Workers: {workers}")

    # Seed from archetypes (profile-only params)
    profile_pop_size = cfg.optimization.population_size
    archetype_pop = seed_population_from_archetypes(
        cfg, profile_bounds, profile_pop_size * n_profile,
        rng=np.random.default_rng(42), profile_only=True,
    )

    # Phase 1 iterations: ~70% of budget (profile is the critical shape)
    phase1_iters = max(10, int(cfg.optimization.max_iterations * 0.7))

    result1 = differential_evolution(
        _profile_objective,
        bounds=profile_bounds,
        args=(cfg,),
        maxiter=phase1_iters,
        popsize=profile_pop_size,
        seed=42,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.8,
        workers=workers,
        updating="deferred",
        disp=verbose,
        init=archetype_pop,
    )

    best_profile = result1.x

    if verbose:
        print(f"\n  Phase 1 complete. Best score: {-result1.fun:.4f}")

    # ---- Phase 2: Cutout placement ----
    if cfg.optimization.num_cutout_pairs > 0 and len(cutout_bounds) > 0:
        if verbose:
            print(f"\n=== Phase 2: Cutout Optimization ===")
            print(f"  Parameters: {len(cutout_bounds)}")
            print(f"  Workers: {workers}")

        # Phase 2 iterations: ~30% of budget
        phase2_iters = max(10, int(cfg.optimization.max_iterations * 0.3))

        result2 = differential_evolution(
            _cutout_objective,
            bounds=cutout_bounds,
            args=(best_profile, cfg),
            maxiter=phase2_iters,
            popsize=cfg.optimization.population_size,
            seed=42,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.8,
            workers=workers,
            updating="deferred",
            disp=verbose,
        )

        best_cutouts = result2.x
    else:
        C = cfg.optimization.num_cutout_pairs
        S = _cutout_stride(cfg)
        best_cutouts = np.zeros(C * S)

    # ---- Reconstruct best solution (with FEA on final result) ----
    x_best = np.concatenate([best_profile, best_cutouts])
    outer, params, cutout_polys = build_weapon_polygon(x_best, cfg)
    weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
    weapon = validate_geometry(weapon)
    metrics = compute_metrics(weapon, cfg, use_fea=True)
    score = weighted_score(metrics, cfg)
    penalty = constraint_penalty(weapon, cfg, cutout_polys=cutout_polys)

    if verbose:
        print(f"\nOptimisation complete.")
        print(f"  Score: {score:.4f} (penalty: {penalty:.4f})")
        print(f"  Mass: {metrics['mass_kg']:.3f} kg / {cfg.weight_budget_kg:.3f} kg budget")
        print(f"  MOI: {metrics['moi_kg_mm2']:.1f} kg·mm²")
        print(f"  Energy: {metrics['energy_joules']:.1f} J @ {cfg.rpm} RPM")
        print(f"  CoM offset: {metrics['com_offset_mm']:.2f} mm")
        print(f"  Structural: {metrics['structural_integrity']:.3f}")
        print(f"  Impact zone: {metrics.get('impact_zone', 0):.3f}")
        if "fea_peak_stress_mpa" in metrics:
            print(f"  FEA peak stress: {metrics['fea_peak_stress_mpa']:.1f} MPa")
            print(f"  FEA safety factor: {metrics['fea_safety_factor']:.2f}")
            print(f"  FEA mesh elements: {metrics['fea_n_elements']}")

    return {
        "result_phase1": result1,
        "weapon_polygon": weapon,
        "outer_profile": outer,
        "cutout_polys": cutout_polys,
        "metrics": metrics,
        "score": score,
        "penalty": penalty,
        "params": params,
    }
