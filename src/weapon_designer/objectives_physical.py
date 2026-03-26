"""Physical objective function for weapon design optimisation.

Replaces the subjective weighted-sum approach with a single physical quantity:
energy transferred to the target per contact event.

    E_transfer = ½·I·ω² × cos²(θ_contact) × (bite / max_bite)
    E_score    = E_transfer / E_ref

where:
    ½·I·ω²          — stored kinetic energy  (J)
    cos²(θ_contact)  — normal energy-transfer fraction (contact_quality²)
    bite / max_bite  — bite efficiency (fraction of theoretical max)
    E_ref = ½ · mass_budget · max_r² · ω²   — solid-disk upper bound (J)

All design-space variables influence E_score through physical pathways:
  • Outer profile shape → I (mass at large radii) and bite / contact geometry
  • Cutout layout      → I (remove material intelligently) and bite geometry
  • Balance (CoM)      → enters as a hard constraint, not a weighted term

Hard constraints (kill-switch — return 0.0, not a soft penalty):
    mass        ≤ weight_budget_kg
    safety_factor ≥ safety_factor_min  (default 1.5, from FEA)
    com_offset  ≤ max_r × com_fraction (default 0.03)
    polygon_valid = True

Usage
-----
    from weapon_designer.objectives_physical import compute_physical_score

    score = compute_physical_score(weapon_polygon, cfg, metrics=None)

    # Or with pre-computed enhanced metrics dict (avoids double FEA):
    metrics = compute_metrics_enhanced(poly, cfg)
    score   = compute_physical_score(poly, cfg, metrics=metrics)
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from .config import WeaponConfig
from .physics import (
    polygon_mass_kg,
    mass_moi_kg_mm2,
    stored_energy_joules,
    com_offset_mm,
)


# Default hard-constraint thresholds (can be overridden via cfg or call kwargs)
_DEFAULT_SAFETY_FACTOR_MIN: float = 1.5
_DEFAULT_COM_FRACTION: float = 0.03   # com_offset ≤ max_r × this


def compute_physical_score(
    poly: Polygon | MultiPolygon,
    cfg: WeaponConfig,
    metrics: dict | None = None,
    safety_factor_min: float = _DEFAULT_SAFETY_FACTOR_MIN,
    com_fraction: float = _DEFAULT_COM_FRACTION,
    fea_spacing: float | None = None,
) -> float:
    """Single-value physical score in [0, 1].

    Parameters
    ----------
    poly              : weapon polygon (holes already punched)
    cfg               : weapon configuration
    metrics           : pre-computed metrics dict from compute_metrics_enhanced().
                        When provided, FEA is NOT re-run.  When None, runs FEA
                        internally at fea_spacing (or cfg.optimization.fea_coarse_spacing_mm).
    safety_factor_min : FEA safety factor hard floor (default 1.5)
    com_fraction      : CoM offset hard limit = max_r × com_fraction (default 0.03)
    fea_spacing       : mesh spacing for internal FEA (if metrics is None)

    Returns
    -------
    float in [0, 1]  — 0 means constraint-violating or degenerate geometry
    """
    if poly is None or poly.is_empty:
        return 0.0

    max_r = cfg.envelope.max_radius_mm

    # ── Hard constraint 1: mass budget ───────────────────────────────────────
    mass = polygon_mass_kg(poly, cfg.sheet_thickness_mm, cfg.material.density_kg_m3)
    if mass > cfg.weight_budget_kg:
        return 0.0

    # ── Hard constraint 2: CoM offset ────────────────────────────────────────
    com = com_offset_mm(poly)
    if com > max_r * com_fraction:
        return 0.0

    # ── Compute or reuse enhanced metrics ────────────────────────────────────
    if metrics is None:
        from .objectives_enhanced import compute_metrics_enhanced
        spacing = fea_spacing if fea_spacing is not None else cfg.optimization.fea_coarse_spacing_mm
        metrics = compute_metrics_enhanced(poly, cfg, fea_spacing=spacing)

    # ── Hard constraint 3: structural safety factor ───────────────────────────
    sf = float(metrics.get("fea_safety_factor", metrics.get("structural_integrity", 0.0)))
    if sf < safety_factor_min:
        return 0.0

    # ── Physical energy-transfer score ────────────────────────────────────────
    moi    = mass_moi_kg_mm2(poly, cfg.sheet_thickness_mm, cfg.material.density_kg_m3)
    omega  = 2.0 * math.pi * max(cfg.rpm, 1) / 60.0   # rad/s
    E_k    = 0.5 * moi * 1e-6 * omega ** 2             # J  (moi mm² → m²)

    # Reference energy: solid disk at mass budget, all mass at max_r
    E_ref  = 0.5 * cfg.weight_budget_kg * (max_r * 1e-3) ** 2 * omega ** 2  # J
    if E_ref < 1e-12:
        return 0.0

    # Contact quality (cos θ at contact points) and bite efficiency
    cq     = float(metrics.get("contact_quality", 0.5))
    bite   = float(metrics.get("bite_mm", 0.0))
    mb     = float(metrics.get("max_bite_mm", 1.0))
    bite_frac = min(bite / max(mb, 1e-6), 1.0)

    # Physical energy transfer: E_k × cos²θ × (bite/max_bite)
    E_transfer = E_k * (cq ** 2) * bite_frac
    E_score    = min(E_transfer / E_ref, 1.0)

    return float(E_score)


def physical_objective(
    poly: Polygon | MultiPolygon,
    cfg: WeaponConfig,
    metrics: dict | None = None,
    fea_spacing: float | None = None,
) -> float:
    """Return 1 − E_score so that minimisers (like scipy DE) work correctly.

    Returns 1.0 for invalid/constrained designs (worst possible).
    """
    score = compute_physical_score(poly, cfg, metrics=metrics, fea_spacing=fea_spacing)
    return 1.0 - score
