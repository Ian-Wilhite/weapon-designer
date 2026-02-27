"""Enhanced objective metrics for the research-comparison pipeline.

Baseline objectives  (objectives.py) are left completely unchanged.
This module provides parallel implementations that fix two known weaknesses:

1. Kinematic spiral bite simulation
   The baseline bite_mm() returns a constant for a given style/RPM —
   geometry cannot influence it.  Here we simulate the actual contact
   mechanics: the approaching enemy robot's face traces an Archimedean
   spiral in the weapon's rotating frame.  Each time the spiral crosses
   the weapon's radial profile, a tooth contact occurs.  Bite depth is
   the enemy advance between consecutive contacts:

       bite_mm = v_approach / (n_contacts × f_rotation)

   This correctly incentivises fewer, more pronounced protrusions:
   • 1 large tooth  → n_contacts=1 → maximum bite (single-hit per revolution)
   • 12 small teeth → n_contacts=12 → 1/12 of maximum bite (like a saw-blade)

2. FEA structural scoring in the objective loop
   The baseline uses a geometric proxy (min wall thickness, section width)
   during optimisation and only runs FEA at the very end.  Here we run a
   coarse-mesh FEA every objective evaluation so the optimizer gets real
   stress information.

All public functions mirror the signatures of their baseline counterparts
so they can be swapped in trivially.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from .config import WeaponConfig
from .physics import (
    polygon_mass_kg,
    mass_moi_kg_mm2,
    stored_energy_joules,
    com_offset_mm,
)
from .objectives import impact_zone_score   # reuse — unchanged baseline helper
from .spiral_contact import analyse_contacts as _spiral_analyse


# ---------------------------------------------------------------------------
# Tooth detection from radial profile
# ---------------------------------------------------------------------------

def detect_teeth(
    poly: Polygon | MultiPolygon,
    min_prominence_mm: float = 3.0,
    n_samples: int = 720,
) -> dict:
    """Detect tooth-like protrusions in the outer boundary radial profile.

    Samples r(θ) — outermost radius at each angle — then finds local maxima
    with scipy.signal.find_peaks.

    Returns
    -------
    dict with keys:
        n_teeth          : int   — number of detected teeth
        mean_height_mm   : float — mean prominence above local baseline (mm)
        mean_sharpness   : float — mean normalised negative 2nd-derivative at peaks
        peak_angles_rad  : array — angles of detected teeth (may be empty)
    """
    from scipy.signal import find_peaks
    from scipy.interpolate import interp1d

    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)

    cx, cy = poly.centroid.x, poly.centroid.y

    # Sample outer boundary coordinates
    ext = np.array(poly.exterior.coords[:-1])
    ext_angles = np.arctan2(ext[:, 1] - cy, ext[:, 0] - cx)
    ext_radii  = np.hypot(ext[:, 0] - cx, ext[:, 1] - cy)

    sort_idx  = np.argsort(ext_angles)
    angles_s  = ext_angles[sort_idx]
    radii_s   = ext_radii[sort_idx]

    # Wrap ±2π copies to allow interpolation at boundaries
    angles_w = np.concatenate([angles_s - 2 * np.pi, angles_s, angles_s + 2 * np.pi])
    radii_w  = np.tile(radii_s, 3)

    # Remove numerical duplicates introduced by the wrap
    angles_u, uid = np.unique(angles_w, return_index=True)
    radii_u = radii_w[uid]

    if len(angles_u) < 4:
        return {"n_teeth": 0, "mean_height_mm": 0.0, "mean_sharpness": 0.0,
                "peak_angles_rad": np.array([])}

    f = interp1d(angles_u, radii_u, kind="linear", fill_value="extrapolate")
    theta_grid = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
    radii_grid = f(theta_grid)

    # Minimum angular separation: ~6° (prevents counting same tooth twice)
    min_distance = max(3, n_samples // 60)
    peaks, props = find_peaks(
        radii_grid,
        prominence=min_prominence_mm,
        distance=min_distance,
    )

    if len(peaks) == 0:
        return {"n_teeth": 0, "mean_height_mm": 0.0, "mean_sharpness": 0.0,
                "peak_angles_rad": np.array([])}

    heights = props["prominences"]

    # Sharpness: |d²r/dθ²| at each peak, normalised by mean radius.
    # A sharper tip concentrates energy and engages armour more aggressively.
    dtheta = 2.0 * np.pi / n_samples
    r_mean = max(radii_grid.mean(), 1.0)
    sharpness = []
    for idx in peaks:
        if 1 <= idx < n_samples - 1:
            d2 = radii_grid[idx - 1] - 2.0 * radii_grid[idx] + radii_grid[idx + 1]
            sharpness.append(abs(d2) / (dtheta ** 2 * r_mean))

    return {
        "n_teeth":        int(len(peaks)),
        "mean_height_mm": float(np.mean(heights)),
        "mean_sharpness": float(np.mean(sharpness)) if sharpness else 0.0,
        "peak_angles_rad": theta_grid[peaks],
    }


# ---------------------------------------------------------------------------
# Kinematic spiral bite simulation
# ---------------------------------------------------------------------------

def kinematic_spiral_bite(
    poly: Polygon | MultiPolygon,
    rpm: float,
    drive_speed_mps: float = 3.0,
    n_samples: int = 1440,
) -> dict:
    """Compute bite depth by simulating the Archimedean spiral contact.

    In the weapon's rotating reference frame the approaching enemy robot's
    face traces an Archimedean spiral:

        r_enemy(θ) = r_start − (v_mm_s / ω) · θ

    where ω = 2π·RPM/60 (rad/s) and v_mm_s = drive_speed_mps × 1000 (mm/s).

    Every time this spiral crosses the weapon's radial profile from outside
    to inside, one tooth contact occurs.  The effective bite per contact is:

        bite_mm = (v_mm_s / ω) · 2π / n_contacts

    A weapon with ONE large protrusion gives n_contacts = 1 and achieves
    the theoretical maximum bite v / f.  Twelve small teeth give n_contacts
    = 12 and cut only 1/12 as deep — correctly penalising saw-blade profiles
    and rewarding pronounced, oblong spinners.

    Parameters
    ----------
    poly            : weapon polygon (with any holes already punched)
    rpm             : weapon spin rate
    drive_speed_mps : enemy approach speed (m/s)  — default 3 m/s
    n_samples       : angular resolution for profile sampling (higher = more
                      accurate contact detection for fine features)

    Returns
    -------
    dict with keys:
        bite_mm        : effective bite depth in mm
        n_contacts     : number of spiral–profile intersections per revolution
        v_per_rad_mm   : mm of approach advance per radian of weapon rotation
        max_bite_mm    : theoretical maximum = v_per_rad_mm × 2π (single-tooth)
    """
    from scipy.interpolate import interp1d

    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)

    omega    = 2.0 * np.pi * max(rpm, 1.0) / 60.0   # rad/s
    v_mm_s   = drive_speed_mps * 1000.0              # mm/s
    v_per_rad = v_mm_s / omega                        # mm per radian of rotation
    max_bite = v_per_rad * 2.0 * np.pi               # theoretical max (single tooth)

    # ── Radial profile r(θ) ─────────────────────────────────────────────
    cx, cy = poly.centroid.x, poly.centroid.y
    ext    = np.array(poly.exterior.coords[:-1])
    angles = np.arctan2(ext[:, 1] - cy, ext[:, 0] - cx)
    radii  = np.hypot(ext[:, 0] - cx, ext[:, 1] - cy)

    sort_idx = np.argsort(angles)
    angles_s = angles[sort_idx]
    radii_s  = radii[sort_idx]

    # Wrap ±2π copies for periodic interpolation at boundaries
    angles_w = np.concatenate([angles_s - 2.0*np.pi, angles_s, angles_s + 2.0*np.pi])
    radii_w  = np.tile(radii_s, 3)
    angles_u, uid = np.unique(angles_w, return_index=True)
    radii_u = radii_w[uid]

    if len(angles_u) < 4:
        return {"bite_mm": max_bite, "n_contacts": 1,
                "v_per_rad_mm": v_per_rad, "max_bite_mm": max_bite}

    f_r = interp1d(angles_u, radii_u, kind="linear", fill_value="extrapolate")
    theta_grid = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
    r_profile  = f_r(theta_grid)

    # ── Archimedean spiral in weapon frame ──────────────────────────────
    # Enemy starts just outside the weapon's maximum radius.
    r_start  = r_profile.max() * 1.01
    # Spiral descends as the weapon rotates (θ = 0 → 2π)
    theta_01 = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    r_spiral = r_start - v_per_rad * theta_01

    # ── Contact detection ───────────────────────────────────────────────
    # Align the spiral with the profile (profile is sampled at -π to +π;
    # spiral is sampled at 0 to 2π — they are the same angular range).
    diff = r_spiral - r_profile
    # A zero-crossing from positive to negative → spiral enters weapon
    contacts = np.sum((diff[:-1] > 0) & (diff[1:] <= 0))

    if contacts == 0:
        # Spiral never crosses profile: weapon is convex / disc with no teeth.
        # Treat as single continuous contact: 1 contact per revolution.
        contacts = 1

    bite_mm = v_per_rad * 2.0 * np.pi / contacts

    return {
        "bite_mm":      float(bite_mm),
        "n_contacts":   int(contacts),
        "v_per_rad_mm": float(v_per_rad),
        "max_bite_mm":  float(max_bite),
    }


# ---------------------------------------------------------------------------
# Contact angle quality (spiral tangent vs. surface normal)
# ---------------------------------------------------------------------------

def contact_angle_quality(
    poly: Polygon | MultiPolygon,
    rpm: float,
    drive_speed_mps: float = 3.0,
    n_spirals: int = 8,
    n_eval: int = 360,
) -> float:
    """Mean |cos(contact_angle)| across Archimedean spiral contacts.

    ``contact_angle`` is the angle between the spiral path tangent and the
    weapon surface's outward normal at each contact point.

    Physical interpretation
    -----------------------
    - cos = 1.0 → spiral hits perpendicular to the face (tooth face-on) →
      all kinetic energy transfers into the weapon; maximum bite depth
    - cos = 0.0 → spiral grazes tangentially → glancing blow, no bite

    A sharp tooth whose leading face is perpendicular to the approach
    direction scores ~1.0; a smooth disk scores ~0.2–0.4 (contacts are
    nearly tangential because the circular boundary is aligned with the
    spiral's approach direction at high RPM).

    Parameters
    ----------
    poly            : weapon polygon (outer profile)
    rpm             : weapon spin rate (used for spiral rate)
    drive_speed_mps : opponent approach speed (m/s) — default 3 m/s
    n_spirals       : number of spiral starting angles to sample (8 is fast)
    n_eval          : angular resolution for profile sampling (360 for speed)

    Returns
    -------
    float in [0, 1] — mean contact quality; 0.5 is a neutral fallback
    """
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)

    try:
        contacts, _ = _spiral_analyse(
            poly,
            n_spirals=n_spirals,
            v_ms=drive_speed_mps,
            rpm=rpm,
            n_eval=n_eval,
        )
        if not contacts:
            return 0.5  # neutral fallback: no contacts detected
        cos_vals = [c.contact_angle_cos for c in contacts]
        return float(np.mean(cos_vals))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Enhanced metrics (FEA in the objective loop)
# ---------------------------------------------------------------------------

def compute_metrics_enhanced(
    poly: Polygon | MultiPolygon,
    cfg: WeaponConfig,
    fea_spacing: float | None = None,
) -> dict:
    """Compute all objective metrics with FEA-based structural scoring.

    Drop-in replacement for objectives.compute_metrics() used inside the
    enhanced optimizer objective functions.

    FEA runs at coarse mesh spacing (fast) during optimisation; call with
    fea_spacing=cfg.optimization.fea_fine_spacing_mm for higher-quality
    evaluation at the end of a run.

    Returns the same key set as compute_metrics() plus enhanced bite keys
    and always-populated FEA keys.
    """
    from .fea import fea_stress_analysis

    t   = cfg.sheet_thickness_mm
    rho = cfg.material.density_kg_m3

    mass   = polygon_mass_kg(poly, t, rho)
    moi    = mass_moi_kg_mm2(poly, t, rho)
    energy = stored_energy_joules(moi, cfg.rpm)
    com    = com_offset_mm(poly)
    impact = impact_zone_score(poly)
    mass_util = mass / cfg.weight_budget_kg if cfg.weight_budget_kg > 0 else 0.0

    # Kinematic spiral bite (replaces constant formula and peak-counting heuristic)
    bite_info = kinematic_spiral_bite(poly, cfg.rpm)

    # Contact angle quality: |cos(angle between spiral tangent and surface normal)|
    # Measures how face-on each contact is — high quality → more bite energy transferred
    # Uses a fast low-resolution scan (8 spirals × 360 angular bins ≈ negligible overhead)
    caq = contact_angle_quality(poly, cfg.rpm)

    # Supplementary tooth geometry (for logging / FEA frame annotations)
    tooth_info = detect_teeth(poly)

    # FEA structural score in the loop (coarse mesh for speed)
    spacing = fea_spacing if fea_spacing is not None else cfg.optimization.fea_coarse_spacing_mm
    fea = fea_stress_analysis(
        poly,
        rpm=cfg.rpm,
        density_kg_m3=rho,
        thickness_mm=t,
        yield_strength_mpa=cfg.material.yield_strength_mpa,
        bore_diameter_mm=cfg.mounting.bore_diameter_mm,
        mesh_spacing=spacing,
    )

    return {
        # Standard keys (compatible with baseline weighted_score)
        "mass_kg":            mass,
        "moi_kg_mm2":         moi,
        "energy_joules":      energy,
        "bite_mm":            bite_info["bite_mm"],
        "com_offset_mm":      com,
        "structural_integrity": fea["fea_score"],   # FEA-based, not geometric proxy
        "mass_utilization":   mass_util,
        "num_teeth":          bite_info["n_contacts"],
        "impact_zone":        impact,
        # Enhanced-only bite keys
        "n_contacts":           bite_info["n_contacts"],
        "n_teeth":              bite_info["n_contacts"],   # alias for logging compatibility
        "v_per_rad_mm":         bite_info["v_per_rad_mm"],
        "max_bite_mm":          bite_info["max_bite_mm"],
        # Contact angle quality (spiral tangent vs. surface normal at each contact)
        # 1.0 = perfectly face-on (max bite energy transfer); 0.0 = pure glancing
        "contact_quality":      caq,
        # Supplementary tooth geometry (detect_teeth, for frame annotations)
        "mean_tooth_height_mm": tooth_info["mean_height_mm"],
        "mean_sharpness":       tooth_info["mean_sharpness"],
        # FEA keys
        "fea_peak_stress_mpa":  fea["peak_stress_mpa"],
        "fea_safety_factor":    fea["safety_factor"],
        "fea_score":            fea["fea_score"],
        "fea_n_elements":       fea["n_elements"],
    }


# ---------------------------------------------------------------------------
# Enhanced weighted score
# ---------------------------------------------------------------------------

def weighted_score_enhanced(metrics: dict, cfg: WeaponConfig) -> float:
    """Weighted multi-objective score using enhanced metrics.

    Identical structure to baseline weighted_score() but uses:
    • effective_bite_mm  (geometry-aware) instead of formula bite
    • FEA-based structural_integrity score
    """
    w     = cfg.optimization.weights
    max_r = cfg.envelope.max_radius_mm

    # MOI score: same normalisation as baseline
    max_moi  = 0.5 * cfg.weight_budget_kg * (max_r ** 2)
    moi_score = min(metrics["moi_kg_mm2"] / max(max_moi, 1e-6), 1.0)

    # Bite score: spiral model — monotone, maximise bite depth.
    # max_bite_mm = v/f (single-tooth theoretical maximum).
    # Multiplied by contact_quality (|cos contact_angle|) so a large bite with a
    # face-on contact scores higher than the same bite depth at a glancing angle.
    # contact_quality ∈ [0, 1]; smooth disks ≈ 0.2–0.4, sharp teeth ≈ 0.7–1.0.
    max_bite   = max(metrics.get("max_bite_mm", 25.0), 1.0)
    bite_depth_score = min(metrics["bite_mm"] / max_bite, 1.0)
    contact_quality  = float(metrics.get("contact_quality", 0.5))
    bite_score = bite_depth_score * contact_quality

    # Structural: already FEA-based (in [0, 1])
    struct_score = metrics["structural_integrity"]

    # Mass utilisation: identical to baseline
    mu = metrics["mass_utilization"]
    if mu > 1.0:
        mass_score = max(0.0, 1.0 - (mu - 1.0) * 5.0)
    else:
        mass_score = mu

    # Balance: identical to baseline
    balance_score = max(0.0, 1.0 - metrics["com_offset_mm"] / max(max_r * 0.1, 1.0))

    # Impact zone: identical to baseline
    iz_score = metrics.get("impact_zone", 0.0)

    total = (
        w.moment_of_inertia  * moi_score
        + w.bite             * bite_score
        + w.structural_integrity * struct_score
        + w.mass_utilization * mass_score
        + w.balance          * balance_score
        + w.impact_zone      * iz_score
    )
    return total
