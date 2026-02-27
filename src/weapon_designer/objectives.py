"""Multi-objective weighted scoring."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point

from .config import WeaponConfig
from .physics import (
    polygon_mass_kg,
    mass_moi_kg_mm2,
    stored_energy_joules,
    bite_mm,
    com_offset_mm,
)
from .structural import structural_score


def impact_zone_score(poly: Polygon | MultiPolygon, min_arc_degrees: float = 30.0) -> float:
    """Score how much of the outer perimeter has solid material for striking.

    Samples the boundary at 1-degree intervals, checks for solid material
    in the outer striking zone (80-100% of max radius), and finds contiguous
    arcs of solid material >= min_arc_degrees.

    Returns a score in [0, 1] based on qualifying impact arcs.
    """
    if isinstance(poly, MultiPolygon):
        # Use the largest piece
        areas = [(p.area, p) for p in poly.geoms]
        poly = max(areas, key=lambda x: x[0])[1]

    centroid = poly.centroid
    cx, cy = centroid.x, centroid.y

    # Find max radius from centroid
    bounds = poly.bounds
    max_extent = max(
        abs(bounds[0] - cx), abs(bounds[2] - cx),
        abs(bounds[1] - cy), abs(bounds[3] - cy),
    )
    if max_extent < 1.0:
        return 0.0

    # Sample at 1-degree intervals
    n_samples = 360
    solid_at_angle = np.zeros(n_samples, dtype=bool)
    strike_zone_start = 0.80  # check from 80% to 100% of max radius

    for i in range(n_samples):
        angle = 2 * np.pi * i / n_samples
        # Check if there's solid material in the striking zone
        has_solid = True
        for frac in [0.80, 0.85, 0.90, 0.95, 1.0]:
            r = max_extent * frac
            px = cx + r * np.cos(angle)
            py = cy + r * np.sin(angle)
            pt = Point(px, py)
            if not poly.contains(pt):
                has_solid = False
                break
        solid_at_angle[i] = has_solid

    # Find contiguous arcs of solid material
    # Wrap around: duplicate the array
    extended = np.concatenate([solid_at_angle, solid_at_angle])
    qualifying_arcs = 0
    arc_length = 0
    counted_starts = set()

    for i in range(len(extended)):
        if extended[i]:
            arc_length += 1
        else:
            if arc_length >= min_arc_degrees:
                # Avoid double-counting wrap-around arcs
                start = (i - arc_length) % n_samples
                if start not in counted_starts:
                    qualifying_arcs += 1
                    counted_starts.add(start)
            arc_length = 0

    # Check final arc
    if arc_length >= min_arc_degrees:
        start = (len(extended) - arc_length) % n_samples
        if start not in counted_starts:
            qualifying_arcs += 1

    score = (qualifying_arcs * min_arc_degrees) / 360.0
    return min(score, 1.0)


def _num_teeth(cfg: WeaponConfig) -> int:
    """Infer number of impact teeth from weapon style."""
    if cfg.weapon_style == "bar":
        return 2
    elif cfg.weapon_style == "eggbeater":
        return 3  # typical
    else:
        return 1  # disk spinner


def compute_metrics(
    poly: Polygon | MultiPolygon,
    cfg: WeaponConfig,
    use_fea: bool = False,
) -> dict:
    """Compute all objective metrics for a weapon polygon.

    Parameters
    ----------
    poly : weapon polygon
    cfg : weapon configuration
    use_fea : if True, run lightweight 2D FEA for structural scoring
    """
    t = cfg.sheet_thickness_mm
    rho = cfg.material.density_kg_m3

    mass = polygon_mass_kg(poly, t, rho)
    moi = mass_moi_kg_mm2(poly, t, rho)
    energy = stored_energy_joules(moi, cfg.rpm)
    teeth = _num_teeth(cfg)
    b = bite_mm(teeth, cfg.rpm)
    com = com_offset_mm(poly)
    struct = structural_score(
        poly,
        cfg.optimization.min_feature_size_mm,
        cfg.optimization.min_wall_thickness_mm,
    )
    mass_util = mass / cfg.weight_budget_kg if cfg.weight_budget_kg > 0 else 0.0
    impact = impact_zone_score(poly)

    result = {
        "mass_kg": mass,
        "moi_kg_mm2": moi,
        "energy_joules": energy,
        "bite_mm": b,
        "com_offset_mm": com,
        "structural_integrity": struct,
        "mass_utilization": mass_util,
        "num_teeth": teeth,
        "impact_zone": impact,
    }

    if use_fea:
        from .fea import fea_stress_analysis
        fea = fea_stress_analysis(
            poly,
            rpm=cfg.rpm,
            density_kg_m3=rho,
            thickness_mm=t,
            yield_strength_mpa=cfg.material.yield_strength_mpa,
            bore_diameter_mm=cfg.mounting.bore_diameter_mm,
        )
        result["fea_peak_stress_mpa"] = fea["peak_stress_mpa"]
        result["fea_safety_factor"] = fea["safety_factor"]
        result["fea_score"] = fea["fea_score"]
        result["fea_n_elements"] = fea["n_elements"]
        # Blend geometric and FEA structural scores
        result["structural_integrity"] = 0.4 * struct + 0.6 * fea["fea_score"]

    return result


def weighted_score(metrics: dict, cfg: WeaponConfig) -> float:
    """Compute weighted multi-objective score in [0, 1].

    Higher is better. Each sub-score is normalized to [0, 1].
    """
    w = cfg.optimization.weights
    max_r = cfg.envelope.max_radius_mm

    # MOI score: normalize by theoretical max (solid disk at max radius)
    # I_disk = 0.5 * m * r², use mass budget for m
    max_moi = 0.5 * cfg.weight_budget_kg * (max_r**2)
    moi_score = min(metrics["moi_kg_mm2"] / max(max_moi, 1e-6), 1.0)

    # Bite score: ideal bite is 10-30mm for most combat robots
    ideal_bite = 20.0  # mm
    bite_val = metrics["bite_mm"]
    bite_score = 1.0 - min(abs(bite_val - ideal_bite) / ideal_bite, 1.0)

    # Structural integrity: already in [0, 1]
    struct_score = metrics["structural_integrity"]

    # Mass utilization: closer to 1.0 is better, penalise over-budget
    mu = metrics["mass_utilization"]
    if mu > 1.0:
        mass_score = max(0.0, 1.0 - (mu - 1.0) * 5.0)  # heavy penalty for overweight
    else:
        mass_score = mu  # higher utilization = better

    # Balance: lower CoM offset is better
    balance_score = max(0.0, 1.0 - metrics["com_offset_mm"] / max(max_r * 0.1, 1.0))

    # Impact zone: already in [0, 1]
    iz_score = metrics.get("impact_zone", 0.0)

    total = (
        w.moment_of_inertia * moi_score
        + w.bite * bite_score
        + w.structural_integrity * struct_score
        + w.mass_utilization * mass_score
        + w.balance * balance_score
        + w.impact_zone * iz_score
    )

    return total
