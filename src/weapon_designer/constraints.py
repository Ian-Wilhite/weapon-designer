"""Manufacturability constraints for weapon profiles."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

from .config import WeaponConfig
from .physics import polygon_mass_kg


def is_connected(poly: Polygon | MultiPolygon) -> bool:
    """Check that the weapon is a single connected piece."""
    if isinstance(poly, MultiPolygon):
        # Allow MultiPolygon only if all pieces are tiny slivers except one
        areas = [p.area for p in poly.geoms]
        main_area = max(areas)
        sliver_threshold = main_area * 0.01
        significant = [a for a in areas if a > sliver_threshold]
        return len(significant) == 1
    return True


def check_mass_budget(
    poly: Polygon | MultiPolygon, cfg: WeaponConfig
) -> tuple[bool, float]:
    """Check if mass is within budget. Returns (ok, mass_kg)."""
    mass = polygon_mass_kg(poly, cfg.sheet_thickness_mm, cfg.material.density_kg_m3)
    return mass <= cfg.weight_budget_kg * 1.01, mass  # 1% tolerance


def check_envelope(poly: Polygon | MultiPolygon, cfg: WeaponConfig) -> bool:
    """Check that the profile fits within the envelope."""
    bounds = poly.bounds  # (minx, miny, maxx, maxy)
    if cfg.weapon_style == "bar":
        length = bounds[2] - bounds[0]
        width = bounds[3] - bounds[1]
        return length <= cfg.envelope.max_length_mm * 1.01 and \
               width <= cfg.envelope.max_width_mm * 1.01
    else:
        # Disk / eggbeater: check max radius
        max_extent = max(abs(bounds[0]), abs(bounds[1]), abs(bounds[2]), abs(bounds[3]))
        return max_extent <= cfg.envelope.max_radius_mm * 1.01


def check_min_feature_size(
    poly: Polygon | MultiPolygon, min_size_mm: float
) -> bool:
    """Check that no feature is too small to cut.

    Uses negative buffer (erosion) — if the eroded polygon is empty,
    there is a feature thinner than min_size_mm/2.
    """
    eroded = poly.buffer(-min_size_mm / 2)
    if eroded.is_empty:
        return False
    return True


def validate_geometry(poly: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    """Fix invalid geometry and return a valid polygon."""
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty:
        return poly
    return poly


def constraint_penalty(
    poly: Polygon | MultiPolygon,
    cfg: WeaponConfig,
    cutout_polys: list | None = None,
) -> float:
    """Compute a [0, 1] penalty multiplier for constraint violations.

    1.0 = all constraints satisfied, 0.0 = fatal violations.
    """
    if poly.is_empty:
        return 0.0

    # Hard kill: disconnected geometry
    if not is_connected(poly):
        return 0.0

    # Mass checks with tighter window [0.975, 1.0) of budget
    mass = polygon_mass_kg(poly, cfg.sheet_thickness_mm, cfg.material.density_kg_m3)
    mass_util = mass / cfg.weight_budget_kg if cfg.weight_budget_kg > 0 else 0.0

    # Hard kill: meaningfully overweight (> 1% above budget).
    # Weapons at exactly the budget (mass_util = 1.0) are valid designs.
    if mass_util > 1.01:
        return 0.0

    # Hard kill: cutout-mounting overlap
    if cutout_polys is not None:
        from .geometry import check_mounting_clearance
        if not check_mounting_clearance(cfg.mounting, cutout_polys):
            return 0.0

    penalty = 1.0

    # Underweight penalty: scale down if below 97.5% of budget
    if mass_util < 0.975:
        penalty *= max(0.1, mass_util / 0.975)

    if not check_envelope(poly, cfg):
        penalty *= 0.3

    if not check_min_feature_size(poly, cfg.optimization.min_feature_size_mm):
        penalty *= 0.5

    return np.clip(penalty, 0.0, 1.0)
