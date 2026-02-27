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
from shapely.geometry import Polygon

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
        return build_bspline_profile(radii, max_r, min_r, n_eval)

    if ptype in ("bezier", "composite_bezier"):
        return build_bezier_profile(radii, max_r, min_r, n_eval)

    if ptype in ("catmull_rom", "catmull-rom", "cr"):
        alpha = getattr(cfg.optimization, "catmull_rom_alpha", 0.5)
        return build_catmull_rom_profile(radii, max_r, min_r, n_eval, alpha=alpha)

    raise ValueError(
        f"Unknown profile_type={profile_type!r}. "
        "Valid choices: 'bspline', 'bezier', 'catmull_rom', 'fourier'."
    )


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

    # All spline families use the same bounds as the B-spline
    return get_bspline_bounds(cfg)
