"""Multi-start Archimedean spiral outer profile generator.

Produces only the outer boundary polygon of a spiral weapon — no interior
cutouts.  This module is intentionally decoupled from the interior geometry
so that any cutout strategy (parametric ribs, topology optimisation, etc.)
can be applied independently.

Geometry
--------
  n_starts arcs are tiled evenly around the circumference.  Each arc sweeps
  2π/n_starts radians, ramping radially from R_spiral_min up to R_outer.
  The abrupt radial step at each arc start is the tooth (contact) face.

  With n_starts = 1:  classic single-tooth spiral (shark-fin profile).
  With n_starts = 2:  two-tooth yin-yang / dual-contact.
  With n_starts = 3+: multi-tooth, progressively more circular.

Parameter vector — OUTER_N_PARAMS = 2
--------------------------------------
  [0] spiral_pitch  — radial step height per tooth face, mm  (5 – 40)
  [1] n_starts      — number of spiral arcs / tooth faces    (1.0 – 4.0, rounded)

Integration
-----------
  Used standalone via build_spiral_outer(params, cfg).
  Combined with spiral_cutouts.build_spiral_cutouts() inside spiral_weapon.py.
  Registered in profile_builder as profile_type="spiral_outer" for use with
  topology Phase 2 inside optimizer_enhanced.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Polygon

if TYPE_CHECKING:
    from .config import WeaponConfig


OUTER_N_PARAMS: int = 2  # [spiral_pitch, n_starts]


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def get_spiral_outer_bounds(cfg: "WeaponConfig") -> list[tuple[float, float]]:
    """DE parameter bounds for the 2-parameter spiral outer profile."""
    R = float(cfg.envelope.max_radius_mm)
    return [
        (5.0, min(40.0, R * 0.50)),  # spiral_pitch: 5 mm to half-radius step
        (1.0, 4.0),                   # n_starts: 1 to 4 (continuous, int-rounded)
    ]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_spiral_outer(
    params: np.ndarray,
    cfg: "WeaponConfig",
    n_spiral_pts: int = 360,
) -> Polygon | None:
    """Return the outer spiral boundary polygon (no interior cutouts).

    Parameters
    ----------
    params       : [spiral_pitch, n_starts]
    cfg          : weapon configuration (envelope used for R_outer)
    n_spiral_pts : total polygon vertex count, distributed across all arcs

    Returns
    -------
    Closed CCW Shapely Polygon, or None on degenerate input.
    """
    spiral_pitch = float(params[0])
    n_starts     = max(1, min(4, int(round(float(params[1])))))

    R_outer = float(cfg.envelope.max_radius_mm)
    R_bore  = float(cfg.mounting.bore_diameter_mm) / 2.0

    # Clamp pitch so the spiral minimum radius stays above the bore
    pitch_clamped = min(spiral_pitch, R_outer - R_bore * 2.0 - 2.0)
    if pitch_clamped < 1.0:
        pitch_clamped = 1.0

    R_spiral_min = R_outer - pitch_clamped
    if R_spiral_min <= R_bore + 1.0:
        R_spiral_min = R_bore + 1.0

    # Each arc spans arc_angle = 2π / n_starts radians.
    # Within arc k: θ_local ∈ [0, arc_angle)
    #   r(θ_local) = R_spiral_min + pitch_clamped * (θ_local / arc_angle)
    # CCW winding → valid Shapely exterior.
    pts_per_arc = max(60, n_spiral_pts // n_starts)
    arc_angle   = 2.0 * math.pi / n_starts

    pts: list[tuple[float, float]] = []
    for k in range(n_starts):
        theta0 = k * arc_angle
        for j in range(pts_per_arc):
            frac  = j / pts_per_arc            # [0, 1) — endpoint excluded
            theta = theta0 + frac * arc_angle
            r     = R_spiral_min + pitch_clamped * frac
            pts.append((r * math.cos(theta), r * math.sin(theta)))

    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area < 50.0:
        return None

    return poly
