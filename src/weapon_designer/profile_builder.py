"""Unified profile-parametrisation dispatcher.

The optimizer imports this module instead of individual profile modules so
that swapping the outer-profile family is a single config-key change:

    cfg.optimization.profile_type = "bspline" | "bezier" | "catmull_rom" | "fourier"

Public API
──────────
    build_profile(profile_type, radii, cfg, n_eval=360) -> Polygon | None
    get_profile_bounds(profile_type, cfg)               -> list[tuple[float, float]]

Both functions are thin dispatchers that call the appropriate module.

Profile families
────────────────
  "fourier"     — Fourier radial series (parametric.py, baseline; bounds from
                  _get_profile_bounds in optimizer.py).  Note: the radii
                  array is not used for this family — the caller must pass
                  the raw Fourier coefficient vector instead.
  "bspline"     — Periodic cubic B-spline through N control points
                  (bspline_profile.py).  Local support, C² continuity.
  "bezier"      — Composite cubic Bézier with CR-style tangents
                  (profile_splines.py).  Local support, C¹ continuity.
  "catmull_rom" — Closed centripetal Catmull-Rom spline
                  (profile_splines.py).  Interpolates control points, C¹.

The "fourier" family uses a different optimizer (baseline) and different
bounds structure; ``build_profile`` forwards to ``build_bspline_profile``
in that case (sensible fallback if caller passes radii accidentally).
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon

from .config import WeaponConfig
from .bspline_profile import build_bspline_profile, get_bspline_bounds
from .profile_splines import build_bezier_profile, build_catmull_rom_profile


# ---------------------------------------------------------------------------
# Dispatcher: profile builder
# ---------------------------------------------------------------------------

def build_profile(
    profile_type: str,
    radii: np.ndarray,
    cfg: WeaponConfig,
    n_eval: int = 360,
) -> Polygon | None:
    """Build an outer-profile polygon for any supported profile family.

    Parameters
    ----------
    profile_type : "bspline", "bezier", "catmull_rom", or "fourier"
    radii        : 1-D array of N radial control values in mm
    cfg          : weapon configuration (used for radius bounds)
    n_eval       : number of polygon vertices

    Returns a closed Shapely Polygon, or None on degenerate input.
    """
    # Compute radius bounds from config — same for all spline families
    if cfg.weapon_style == "bar":
        max_r = float(np.hypot(cfg.envelope.max_length_mm / 2.0,
                               cfg.envelope.max_width_mm / 2.0))
    else:
        max_r = float(cfg.envelope.max_radius_mm)

    bore_r = cfg.mounting.bore_diameter_mm / 2.0
    min_r  = max(bore_r + cfg.optimization.min_wall_thickness_mm + 5.0, 15.0)
    max_r  = max(max_r, min_r + 10.0)

    ptype = profile_type.lower().strip()

    if ptype in ("bspline", "b_spline", "fourier"):
        # "fourier" fallback: if caller passes spline radii use bspline
        poly = build_bspline_profile(radii, max_r, min_r, n_eval)

    elif ptype in ("bezier", "composite_bezier"):
        poly = build_bezier_profile(radii, max_r, min_r, n_eval)

    elif ptype in ("catmull_rom", "catmull-rom", "cr"):
        alpha = getattr(cfg.optimization, "catmull_rom_alpha", 0.5)
        poly = build_catmull_rom_profile(radii, max_r, min_r, n_eval, alpha=alpha)

    elif ptype in ("functional", "lobed"):
        from .functional_profiles import build_functional_profile
        N_ctrl = getattr(cfg.optimization, "n_bspline_points", 12)
        poly = build_functional_profile(radii, max_r, min_r, N_ctrl, n_eval)

    elif ptype == "spiral_weapon":
        # spiral_weapon builds its own complete polygon (outer + cutouts).
        # `radii` is the 7-element parameter vector for build_spiral_weapon.
        from .spiral_weapon import build_spiral_weapon
        return build_spiral_weapon(np.asarray(radii), cfg)

    elif ptype == "spiral_outer":
        # Outer spiral profile only — interior cutouts handled by Phase 2
        # (topology or parametric).  `radii` is [spiral_pitch, n_starts].
        from .spiral_outer import build_spiral_outer
        return build_spiral_outer(np.asarray(radii), cfg)

    else:
        raise ValueError(
            f"Unknown profile_type={profile_type!r}. "
            "Valid choices: 'bspline', 'bezier', 'catmull_rom', 'fourier', "
            "'functional', 'spiral_weapon', 'spiral_outer'."
        )

    if poly is None:
        return None

    # ── Enforce radius bounds after spline generation ──────────────────────
    # Splines can overshoot control-point bounds (Gibbs-like ringing, Bézier
    # convex-hull property not always tight).  Clip to the design envelope so
    # FEA boundary conditions and mass normalisation are always consistent.
    try:
        envelope_circle = Point(0.0, 0.0).buffer(max_r, resolution=64)
        poly = poly.intersection(envelope_circle)
        if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
            return None
        # Take largest component if intersection produced a MultiPolygon
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    except Exception:
        pass  # if shapely fails, return the raw polygon unchanged

    return poly


# ---------------------------------------------------------------------------
# Dispatcher: DE bounds
# ---------------------------------------------------------------------------

def get_profile_bounds(
    profile_type: str,
    cfg: WeaponConfig,
) -> list[tuple[float, float]]:
    """Return differential-evolution bounds for the profile parameter vector.

    All spline families ("bspline", "bezier", "catmull_rom") share the same
    bounds: N × (r_min, r_max) where N = cfg.optimization.n_bspline_points.
    The "fourier" family uses its own bounds structure (from optimizer.py)
    and is *not* handled here — callers using Fourier profiles should call
    ``optimizer._get_profile_bounds(cfg)`` directly.

    Parameters
    ----------
    profile_type : "bspline", "bezier", "catmull_rom", or "fourier"
    cfg          : weapon configuration

    Returns
    -------
    list of (lo, hi) tuples, one per parameter
    """
    ptype = profile_type.lower().strip()

    if ptype == "fourier":
        # Import lazily to avoid circular dependency with optimizer.py
        from .optimizer import _get_profile_bounds as _fourier_bounds
        return _fourier_bounds(cfg)

    if ptype in ("functional", "lobed"):
        from .functional_profiles import get_functional_bounds
        return get_functional_bounds(cfg)

    if ptype == "spiral_weapon":
        from .spiral_weapon import get_spiral_weapon_bounds
        return get_spiral_weapon_bounds(cfg)

    if ptype == "spiral_outer":
        from .spiral_outer import get_spiral_outer_bounds
        return get_spiral_outer_bounds(cfg)

    # All spline families use the same bounds as the B-spline
    return get_bspline_bounds(cfg)


# ---------------------------------------------------------------------------
# Dispatcher: interior voids
# ---------------------------------------------------------------------------

def build_interior(
    interior_type: str,
    params,
    cfg,
    outer_profile=None,
):
    """Dispatcher: build interior void polygons for any interior_type.

    Returns list of Shapely Polygon to subtract from outer profile.

    interior_type: "superellipse" | "spoke" | "slot" | "web"
    Note: "topology" is handled separately by topo_optimizer.py.
    """
    import numpy as np

    ptype = interior_type.lower()

    if ptype == "spoke":
        from .interior_templates import spoke_voids
        n_spokes = max(2, round(float(params[0])))
        return spoke_voids(n_spokes, float(params[1]), float(params[2]),
                           float(params[3]), float(params[4]))

    elif ptype == "slot":
        from .interior_templates import slot_voids
        n_slots = max(2, round(float(params[0])))
        return slot_voids(n_slots, float(params[1]), float(params[2]),
                          float(params[3]), float(params[4]))

    elif ptype == "web":
        if outer_profile is None:
            return []
        from .interior_templates import web_offset_void
        bore_r = cfg.mounting.bore_diameter_mm / 2.0
        void = web_offset_void(outer_profile, float(params[0]), bore_r)
        return [void] if void is not None else []

    else:
        # "superellipse" or unknown -- return empty (handled by existing parametric_cad code)
        return []


# ---------------------------------------------------------------------------
# Single-phase profile flag
# ---------------------------------------------------------------------------

#: Profile types that build complete weapon geometry internally (outer + cutouts).
#: For these types, optimizer_enhanced skips Phase 2 (topology / polar cutouts).
SINGLE_PHASE_PROFILES: frozenset[str] = frozenset({"spiral_weapon"})


def is_single_phase_profile(profile_type: str) -> bool:
    """Return True if the profile builds its own cutouts — Phase 2 should be skipped."""
    return profile_type.lower().strip() in SINGLE_PHASE_PROFILES


def get_interior_bounds(
    interior_type: str,
    cfg,
):
    """Return DE parameter bounds for the given interior_type."""
    ptype = interior_type.lower()
    if ptype == "spoke":
        from .interior_templates import get_spoke_bounds
        return get_spoke_bounds(cfg)
    elif ptype == "slot":
        from .interior_templates import get_slot_bounds
        return get_slot_bounds(cfg)
    elif ptype == "web":
        from .interior_templates import get_web_bounds
        return get_web_bounds(cfg)
    else:
        return []
