"""Parametric interior cutout generator for spiral (or any) outer profiles.

Takes any outer boundary polygon and carves the interior into:
  • A solid rim band  — inward offset from the outer edge by t_rim.
  • A solid hub disk  — outward from the bolt-hole pattern extents by t_hub.
  • A void annulus    — the region between rim inner boundary and hub disk.
  • Radial support ribs — n_supports strips of width t_support spanning the void.
  • Filleted corners  — Minkowski rounding of all concave rib-root corners.

The two wall offsets are intentionally independent parameters:
  t_rim  — measured inward from the outer profile edge  (controls rim rigidity)
  t_hub  — measured outward from the bolt-hole extents  (controls hub clearance)

This decoupling lets the optimizer independently trade rim thickness against
hub size without them being tied to a single "wall thickness" parameter.

Parameter vector — CUTOUT_N_PARAMS = 5
---------------------------------------
  [0] t_rim       — rim wall thickness inward from outer edge, mm    (4 – 12)
  [1] t_hub       — hub collar thickness beyond bolt-hole extents, mm (4 – 12)
  [2] n_supports  — number of radial ribs                             (2.0 – 6.0, rounded)
  [3] t_support   — rib width, mm                                     (4 – 10)
  [4] r_fillet    — concave corner fillet radius, mm                  (1.5 – 5)

Public API
----------
  get_spiral_cutout_bounds(cfg) → list[tuple[float, float]]
  build_spiral_cutouts(outer_poly, params, cfg) → Shapely geometry | None

  build_spiral_cutouts returns the *void region* (the material to remove).
  Caller subtracts it from the outer polygon:
      weapon = outer_poly.difference(build_spiral_cutouts(outer_poly, params, cfg))

  Returns None if the geometry degenerates (e.g., hub larger than rim interior).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

if TYPE_CHECKING:
    from .config import WeaponConfig


CUTOUT_N_PARAMS: int = 5  # [t_rim, t_hub, n_supports, t_support, r_fillet]


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def get_spiral_cutout_bounds(cfg: "WeaponConfig") -> list[tuple[float, float]]:
    """DE parameter bounds for the 5-parameter spiral cutout generator.

    t_rim is bounded relative to weapon radius (rim must stay structurally
    useful but not so thick it leaves no void to optimise).
    t_hub is bounded relative to the mounting pattern size (extra collar
    beyond the bolt-hole extents — independent of t_rim).
    """
    R = float(cfg.envelope.max_radius_mm)
    return [
        (4.0, min(14.0, R * 0.16)),  # t_rim:      inward from outer edge
        (4.0, min(12.0, R * 0.12)),  # t_hub:      outward from bolt-hole extents
        (2.0, 6.0),                   # n_supports: rib count (int-rounded)
        (4.0, min(10.0, R * 0.10)),  # t_support:  rib width
        (1.5, 5.0),                   # r_fillet:   corner rounding radius
    ]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_spiral_cutouts(
    outer_poly: Polygon,
    params: np.ndarray,
    cfg: "WeaponConfig",
) -> "Polygon | None":
    """Compute the void region to subtract from outer_poly.

    Parameters
    ----------
    outer_poly : closed outer boundary polygon (from build_spiral_outer or
                 any other outer profile builder)
    params     : [t_rim, t_hub, n_supports, t_support, r_fillet]
    cfg        : weapon configuration (mounting dimensions for hub calculation)

    Returns
    -------
    Shapely geometry representing the interior void, or None if degenerate.
    Caller should do: weapon = outer_poly.difference(returned_void)
    """
    t_rim, t_hub, n_sup_f, t_support, r_fillet = (float(p) for p in params[:5])
    n_supports = max(2, min(8, int(round(n_sup_f))))

    # Clamp to physically realisable values (plasma kerf constraint)
    r_fillet  = max(1.0,             min(r_fillet,  5.0))
    t_rim     = max(2.0 * r_fillet,  t_rim)
    t_hub     = max(2.0 * r_fillet,  t_hub)
    t_support = max(2.0 * r_fillet,  t_support)

    R_outer = float(cfg.envelope.max_radius_mm)

    # ── Hub boundary: measured from outer extent of bolt-hole pattern ──────
    # Guarantees bolt holes are fully enclosed in solid hub material.
    bolt_circle_r  = float(cfg.mounting.bolt_circle_diameter_mm) / 2.0
    bolt_hole_r    = float(cfg.mounting.bolt_hole_diameter_mm) / 2.0
    R_mount_extent = bolt_circle_r + bolt_hole_r   # furthest reach of any bolt hole
    R_hub          = R_mount_extent + t_hub

    # ── 1. Rim inner boundary (uniform inward offset from outer edge) ──────
    rim_inner = outer_poly.buffer(-t_rim, resolution=64)
    if rim_inner is None or rim_inner.is_empty:
        return None

    # ── 2. Hub disk ────────────────────────────────────────────────────────
    hub = Point(0.0, 0.0).buffer(R_hub, resolution=64)

    # ── 3. Void annulus = rim_inner − hub ──────────────────────────────────
    void_ring = rim_inner.difference(hub)
    if void_ring is None or void_ring.is_empty:
        return None   # hub larger than rim interior — geometry degenerate

    # ── 4. Radial support ribs ─────────────────────────────────────────────
    # Each rib is a full-width rectangle from origin to R_outer+margin,
    # then clipped to the void annulus.  Works for any outer profile shape.
    half_w = t_support / 2.0
    base_strip = Polygon([
        (0.0,           -half_w),
        (R_outer + 2.0, -half_w),
        (R_outer + 2.0,  half_w),
        (0.0,            half_w),
    ])

    support_polys = []
    for k in range(n_supports):
        angle_deg = k * 360.0 / n_supports
        strip   = shp_rotate(base_strip, angle_deg, origin=(0.0, 0.0), use_radians=False)
        clipped = strip.intersection(void_ring)
        if not clipped.is_empty:
            support_polys.append(clipped)

    supports_union = (
        unary_union(support_polys) if support_polys
        else Point(0, 0).buffer(0)
    )

    # ── 5. Cutout = void − supports ────────────────────────────────────────
    cutout = void_ring.difference(supports_union)
    if cutout is None or cutout.is_empty:
        return None   # entire void covered by supports → caller keeps solid outer

    # ── 6. Fillet concave corners (rib roots) ──────────────────────────────
    # buffer(-r).buffer(r) on the void rounds the concave corners of the
    # weapon material, reducing stress concentrations at rib-root junctions.
    if r_fillet >= 1.0:
        try:
            cutout_filleted = cutout.buffer(-r_fillet, resolution=16).buffer(
                r_fillet, resolution=16
            )
            if not cutout_filleted.is_empty:
                cutout = cutout_filleted
        except Exception:
            pass  # fallback: keep un-filleted

    return cutout
