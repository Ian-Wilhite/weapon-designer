"""Spiral weapon assembler — combines outer profile + interior cutouts.

This module is the public entry point for the 7-parameter spiral weapon.
The geometry is intentionally split into two independent sub-modules:

  spiral_outer.py    — Archimedean outer profile (pitch, n_starts)
  spiral_cutouts.py  — Parametric rib cutouts    (t_rim, t_hub, n_supports,
                                                   t_support, r_fillet)

The two offset parameters are deliberately separated:
  t_rim — inward from the outer spiral edge   (rim rigidity)
  t_hub — outward from the bolt-hole extents  (hub clearance / mounting safety)

Keeping them in different sub-modules means either half can be swapped
independently, e.g.:

  • Spiral outer + topology Phase 2
      profile_type = "spiral_outer"  →  Phase 1 optimises outer shape
      cutout_type  = "topology"      →  Phase 2 SIMP-optimises the interior

  • Spiral outer + parametric ribs (this module)
      build_spiral_weapon(params, cfg)  →  full weapon in one call

  • B-spline outer + spiral ribs
      outer = build_bspline_profile(...)
      void  = build_spiral_cutouts(outer, cutout_params, cfg)
      weapon = outer.difference(void)

Parameter vector (7 values, all mm unless noted)
-------------------------------------------------
  [0] spiral_pitch  — step height per tooth face, mm         (5 – 40)
  [1] n_starts      — spiral starts / tooth faces            (1.0 – 4.0, rounded)
  [2] t_rim         — rim wall thickness from outer edge, mm (4 – 14)
  [3] t_hub         — hub collar beyond bolt-hole extents, mm(4 – 12)
  [4] n_supports    — radial rib count                       (2.0 – 6.0, rounded)
  [5] t_support     — rib width, mm                         (4 – 10)
  [6] r_fillet      — fillet radius, mm                      (1.5 – 5)

Integration
-----------
  profile_type = "spiral_weapon"  in OptimizationParams.
  build_profile() returns the complete weapon polygon with cutouts punched
  (bore hole absent — assemble_weapon() handles that).
  Phase 2 of optimizer_enhanced is skipped (is_single_phase_profile = True).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from shapely.ops import unary_union

from .spiral_outer   import build_spiral_outer,   get_spiral_outer_bounds,   OUTER_N_PARAMS
from .spiral_cutouts import build_spiral_cutouts, get_spiral_cutout_bounds, CUTOUT_N_PARAMS

if TYPE_CHECKING:
    from shapely.geometry import Polygon
    from .config import WeaponConfig


TOTAL_N_PARAMS: int = OUTER_N_PARAMS + CUTOUT_N_PARAMS  # 7


# ---------------------------------------------------------------------------
# Public bounds  (concatenation of both sub-modules)
# ---------------------------------------------------------------------------

def get_spiral_weapon_bounds(cfg: "WeaponConfig") -> list[tuple[float, float]]:
    """Return DE bounds for the full 7-parameter spiral weapon."""
    return get_spiral_outer_bounds(cfg) + get_spiral_cutout_bounds(cfg)


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

def build_spiral_weapon(
    params: np.ndarray,
    cfg: "WeaponConfig",
    n_spiral_pts: int = 360,
) -> "Polygon | None":
    """Build the complete spiral weapon polygon (outer − void, no bore hole).

    Parameters
    ----------
    params       : 7-element array  [spiral_pitch, n_starts,
                   t_rim, t_hub, n_supports, t_support, r_fillet]
    cfg          : weapon configuration
    n_spiral_pts : polygon resolution for the outer spiral arc

    Returns
    -------
    Shapely Polygon with interior cutout rings, or None on degenerate input.
    The bore hole and bolt holes are NOT removed — call assemble_weapon() for that.
    """
    params = np.asarray(params, dtype=float)
    outer_params  = params[:OUTER_N_PARAMS]
    cutout_params = params[OUTER_N_PARAMS:OUTER_N_PARAMS + CUTOUT_N_PARAMS]

    # ── Step 1: outer profile ──────────────────────────────────────────────
    outer_poly = build_spiral_outer(outer_params, cfg, n_spiral_pts)
    if outer_poly is None:
        return None

    # ── Step 2: interior void geometry ────────────────────────────────────
    cutout = build_spiral_cutouts(outer_poly, cutout_params, cfg)
    if cutout is None or cutout.is_empty:
        # All void covered by supports or hub too large → return solid outer
        return outer_poly

    # ── Step 3: subtract void from outer ──────────────────────────────────
    weapon = outer_poly.difference(cutout)
    if weapon is None or weapon.is_empty:
        return None
    if not weapon.is_valid:
        weapon = weapon.buffer(0)
    if weapon.geom_type == "MultiPolygon":
        weapon = max(weapon.geoms, key=lambda g: g.area)

    return weapon if (not weapon.is_empty and weapon.area > 50.0) else None


# ---------------------------------------------------------------------------
# Quick metrics (no FEA)
# ---------------------------------------------------------------------------

def spiral_weapon_metrics(weapon: "Polygon", cfg: "WeaponConfig") -> dict:
    """Return lightweight geometric metrics for a spiral weapon polygon."""
    from .physics import polygon_mass_kg, mass_moi_kg_mm2, com_offset_mm

    R   = float(cfg.envelope.max_radius_mm)
    t   = float(cfg.sheet_thickness_mm)
    rho = float(cfg.material.density_kg_m3)

    solid_area = math.pi * R ** 2
    mass_frac  = weapon.area / solid_area

    mass  = polygon_mass_kg(weapon, t, rho)
    moi   = mass_moi_kg_mm2(weapon, t, rho)
    com   = com_offset_mm(weapon)

    omega = 2.0 * math.pi * cfg.rpm / 60.0
    ke    = 0.5 * (moi * 1e-6) * omega ** 2   # Joules (moi in kg·m²)

    return {
        "mass_kg":       mass,
        "mass_frac":     mass_frac,
        "moi_kg_mm2":    moi,
        "energy_joules": ke,
        "com_offset_mm": com,
        "area_mm2":      weapon.area,
    }
