"""Structured interior void templates for weapon weight reduction.

These parameterizations replace free superellipse pockets with
manufacturable layouts: spokes, slots, and web offsets.
All returned voids are pre-conditioned with a fillet radius so they
are manufacturable (no sharp concave corners).

Parallel to parametric_cad.py -- never modifies that file.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.affinity import rotate
from shapely.validation import make_valid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_annulus(r_inner: float, r_outer: float, n_pts: int = 128) -> Polygon:
    """Return a filled annular polygon (outer ring minus inner hole)."""
    theta = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
    outer = list(zip(r_outer * np.cos(theta), r_outer * np.sin(theta)))
    inner = list(zip(r_inner * np.cos(theta[::-1]), r_inner * np.sin(theta[::-1])))
    return Polygon(outer, [inner])


def _arc_points(r: float, theta_start: float, theta_end: float, n: int = 32) -> list[tuple[float, float]]:
    """CCW arc points at radius r from theta_start to theta_end."""
    thetas = np.linspace(theta_start, theta_end, n)
    return [(r * math.cos(t), r * math.sin(t)) for t in thetas]


def _fillet(poly: Polygon, R: float) -> Polygon:
    """Apply Minkowski fillet: erode then dilate by R.

    If the erosion collapses the polygon, return the original.
    """
    if R <= 0.0:
        return poly
    eroded = poly.buffer(-R)
    if eroded.is_empty:
        return poly
    result = eroded.buffer(R)
    if result.is_empty or not result.is_valid:
        return poly
    return result


def _largest_polygon(geom) -> Polygon | None:
    """Extract the largest-area Polygon from a geometry."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if geom.area > 0 else None
    if isinstance(geom, MultiPolygon):
        parts = [g for g in geom.geoms if isinstance(g, Polygon) and g.area > 0]
        return max(parts, key=lambda p: p.area) if parts else None
    # GeometryCollection fallback
    parts = []
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            p = _largest_polygon(g)
            if p is not None:
                parts.append(p)
    return max(parts, key=lambda p: p.area) if parts else None


# ---------------------------------------------------------------------------
# spoke_voids
# ---------------------------------------------------------------------------

def spoke_voids(
    n_spokes: int,
    spoke_width_mm: float,
    hub_r: float,
    rim_r_inner: float,
    phi_offset: float = 0.0,
    R_fillet: float = 2.0,
) -> list[Polygon]:
    """Create void regions between adjacent spokes.

    Parameters
    ----------
    n_spokes       : number of spokes (>= 2)
    spoke_width_mm : width of each spoke in mm
    hub_r          : inner radius of annular void region (= hub outer edge)
    rim_r_inner    : outer radius of annular void region (= rim inner edge)
    phi_offset     : angular offset of first spoke, radians
    R_fillet       : Minkowski fillet radius applied to each void

    Returns
    -------
    List of Shapely Polygons — the spaces BETWEEN spokes.
    """
    n_spokes = max(2, int(round(n_spokes)))
    spoke_width_mm = max(1.0, float(spoke_width_mm))
    hub_r = max(1.0, float(hub_r))
    rim_r_inner = max(hub_r + 1.0, float(rim_r_inner))
    phi_offset = float(phi_offset)

    if hub_r >= rim_r_inner:
        return []

    spoke_length = rim_r_inner - hub_r

    # Build one spoke as a rectangle centred on the X axis, then rotate.
    # Rectangle: from hub_r to rim_r_inner in X, width spoke_width_mm in Y.
    def _spoke_poly(angle_rad: float) -> Polygon:
        x0, x1 = hub_r, rim_r_inner
        y0, y1 = -spoke_width_mm / 2.0, spoke_width_mm / 2.0
        rect = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        return rotate(rect, math.degrees(angle_rad), origin=(0.0, 0.0), use_radians=False)

    # Union of all spoke rectangles
    from shapely.ops import unary_union
    spokes_union = unary_union([
        _spoke_poly(phi_offset + i * 2.0 * math.pi / n_spokes)
        for i in range(n_spokes)
    ])

    # Full annulus that the voids live in
    annulus = _make_annulus(hub_r, rim_r_inner)

    # Voids = annulus minus spoke material
    raw_void = annulus.difference(spokes_union)
    if raw_void.is_empty:
        return []

    # Split into individual void cells and fillet each
    voids: list[Polygon] = []
    if isinstance(raw_void, Polygon):
        candidates = [raw_void]
    elif isinstance(raw_void, MultiPolygon):
        candidates = list(raw_void.geoms)
    elif hasattr(raw_void, "geoms"):
        candidates = [g for g in raw_void.geoms if isinstance(g, Polygon)]
    else:
        candidates = []

    for cell in candidates:
        if cell.is_empty or cell.area < 1.0:
            continue
        filleted = _fillet(cell, R_fillet)
        p = _largest_polygon(filleted)
        if p is not None and p.area > 1.0:
            voids.append(p)

    return voids


# ---------------------------------------------------------------------------
# slot_voids
# ---------------------------------------------------------------------------

def slot_voids(
    n_slots: int,
    slot_width_mm: float,
    slot_r_inner: float,
    slot_r_outer: float,
    phi_offset: float = 0.0,
    R_fillet: float = 2.0,
) -> list[Polygon]:
    """Create N radially-aligned arc slot voids.

    Each slot is a wedge (sector) of the annular region from slot_r_inner
    to slot_r_outer.  The angular half-width is chosen so that the chord
    at the inner radius equals slot_width_mm.

    Parameters
    ----------
    n_slots        : number of slots (>= 2)
    slot_width_mm  : approximate slot width at inner radius (mm)
    slot_r_inner   : inner radius of each slot
    slot_r_outer   : outer radius of each slot
    phi_offset     : angular offset for first slot, radians
    R_fillet       : Minkowski fillet radius

    Returns
    -------
    List of Shapely Polygons (one per slot).
    """
    n_slots = max(2, int(round(n_slots)))
    slot_width_mm = max(1.0, float(slot_width_mm))
    slot_r_inner = max(1.0, float(slot_r_inner))
    slot_r_outer = max(slot_r_inner + 1.0, float(slot_r_outer))
    phi_offset = float(phi_offset)

    # Angular half-width: arc length = slot_width_mm at inner radius
    half_dtheta = (slot_width_mm / slot_r_inner) / 2.0
    # Clamp so slots don't overlap
    max_half = math.pi / n_slots * 0.9
    half_dtheta = min(half_dtheta, max_half)

    voids: list[Polygon] = []
    step = 2.0 * math.pi / n_slots

    for i in range(n_slots):
        centre_angle = phi_offset + i * step
        t0 = centre_angle - half_dtheta
        t1 = centre_angle + half_dtheta

        # Wedge: inner arc (CCW), then outer arc (CW)
        inner_pts = _arc_points(slot_r_inner, t0, t1, n=32)
        outer_pts = _arc_points(slot_r_outer, t1, t0, n=32)
        coords = inner_pts + outer_pts + [inner_pts[0]]

        try:
            wedge = Polygon(coords)
            if not wedge.is_valid:
                wedge = make_valid(wedge)
                wedge = _largest_polygon(wedge)
                if wedge is None:
                    continue
        except Exception:
            continue

        filleted = _fillet(wedge, R_fillet)
        p = _largest_polygon(filleted)
        if p is not None and p.area > 1.0:
            voids.append(p)

    return voids


# ---------------------------------------------------------------------------
# web_offset_void
# ---------------------------------------------------------------------------

def web_offset_void(
    outer_profile: Polygon,
    web_thickness_mm: float,
    bore_r: float,
    hub_multiplier: float = 2.5,
    R_fillet: float = 2.0,
) -> Polygon | None:
    """Single void: inward-offset of the outer profile minus hub circle.

    Erodes the outer profile inward by web_thickness_mm to form the inner
    boundary of a web ring, then subtracts the hub circle to leave an
    annular void between hub and web.

    Parameters
    ----------
    outer_profile     : Shapely Polygon of the weapon outer boundary
    web_thickness_mm  : web wall thickness (erosion depth) in mm
    bore_r            : bore radius in mm
    hub_multiplier    : hub_r = bore_r * hub_multiplier
    R_fillet          : Minkowski fillet radius applied to void

    Returns
    -------
    Shapely Polygon or None if the resulting void is degenerate.
    """
    web_thickness_mm = max(1.0, float(web_thickness_mm))
    bore_r = max(1.0, float(bore_r))
    hub_r = bore_r * float(hub_multiplier)

    eroded = outer_profile.buffer(-web_thickness_mm)
    if eroded.is_empty:
        return None

    hub_circle = Point(0.0, 0.0).buffer(hub_r, resolution=64)
    void = eroded.difference(hub_circle)

    if void.is_empty:
        return None

    p = _largest_polygon(void)
    if p is None or p.area < 1.0:
        return None

    filleted = _fillet(p, R_fillet)
    result = _largest_polygon(filleted)
    if result is None or result.area < 1.0:
        return None

    return result


# ---------------------------------------------------------------------------
# Bounds helpers
# ---------------------------------------------------------------------------

def get_spoke_bounds(cfg) -> list[tuple[float, float]]:
    """DE parameter bounds for spoke_voids.

    Parameter vector: [n_spokes, spoke_width_mm, hub_r, rim_r_inner, phi_offset]
    """
    bore_r = cfg.mounting.bore_diameter_mm / 2.0

    if cfg.weapon_style == "bar":
        max_r = float(
            np.hypot(cfg.envelope.max_length_mm / 2.0, cfg.envelope.max_width_mm / 2.0)
        )
    else:
        max_r = float(cfg.envelope.max_radius_mm)

    return [
        (2.0,              8.0),                         # n_spokes
        (3.0,              20.0),                        # spoke_width_mm
        (bore_r * 1.5,     bore_r * 4.0),               # hub_r
        (max_r * 0.5,      max_r * 0.9),                # rim_r_inner
        (0.0,              2.0 * math.pi / 8.0),        # phi_offset
    ]


def get_slot_bounds(cfg) -> list[tuple[float, float]]:
    """DE parameter bounds for slot_voids.

    Parameter vector: [n_slots, slot_width_mm, slot_r_inner, slot_r_outer, phi_offset]
    """
    bore_r = cfg.mounting.bore_diameter_mm / 2.0

    if cfg.weapon_style == "bar":
        max_r = float(
            np.hypot(cfg.envelope.max_length_mm / 2.0, cfg.envelope.max_width_mm / 2.0)
        )
    else:
        max_r = float(cfg.envelope.max_radius_mm)

    return [
        (2.0,              12.0),                        # n_slots
        (5.0,              30.0),                        # slot_width_mm
        (bore_r * 2.0,     max_r * 0.4),               # slot_r_inner
        (max_r * 0.5,      max_r * 0.85),              # slot_r_outer
        (0.0,              2.0 * math.pi),              # phi_offset
    ]


def get_web_bounds(cfg) -> list[tuple[float, float]]:
    """DE parameter bounds for web_offset_void.

    Parameter vector: [web_thickness_mm]
    """
    if cfg.weapon_style == "bar":
        max_r = float(
            np.hypot(cfg.envelope.max_length_mm / 2.0, cfg.envelope.max_width_mm / 2.0)
        )
    else:
        max_r = float(cfg.envelope.max_radius_mm)

    return [
        (3.0, max_r * 0.3),   # web_thickness_mm
    ]
