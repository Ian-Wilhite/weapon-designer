"""Structural integrity scoring.

Combines fast geometric proxies with optional lightweight 2D FEA.
Geometric checks are used during optimization; FEA for validation scoring.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import nearest_points


def min_section_width(poly: Polygon | MultiPolygon, n_rays: int = 72) -> float:
    """Estimate minimum cross-section width by casting radial rays through the centroid.

    Returns the minimum chord length across the weapon in mm.
    """
    if isinstance(poly, MultiPolygon):
        return min(min_section_width(p, n_rays) for p in poly.geoms)

    cx, cy = poly.centroid.x, poly.centroid.y
    bounds = poly.bounds
    diag = np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])

    min_width = float("inf")
    for i in range(n_rays):
        angle = np.pi * i / n_rays  # 0 to π (symmetric pairs)
        dx = diag * np.cos(angle)
        dy = diag * np.sin(angle)
        ray = LineString([(cx - dx, cy - dy), (cx + dx, cy + dy)])
        intersection = ray.intersection(poly)
        if intersection.is_empty:
            continue
        if intersection.geom_type == "LineString":
            min_width = min(min_width, intersection.length)
        elif intersection.geom_type == "MultiLineString":
            for seg in intersection.geoms:
                min_width = min(min_width, seg.length)

    return min_width if min_width < float("inf") else 0.0


def wall_to_hole_distances(poly: Polygon | MultiPolygon) -> list[float]:
    """Compute minimum distances from each interior ring (hole) to the exterior boundary.

    Returns a list of distances in mm, one per hole.
    """
    if isinstance(poly, MultiPolygon):
        dists = []
        for p in poly.geoms:
            dists.extend(wall_to_hole_distances(p))
        return dists

    dists = []
    ext_ring = poly.exterior
    for interior in poly.interiors:
        hole_line = LineString(interior.coords)
        ext_line = LineString(ext_ring.coords)
        d = hole_line.distance(ext_line)
        dists.append(d)
    return dists


def min_wall_thickness(poly: Polygon | MultiPolygon) -> float:
    """Return the minimum wall thickness (hole-to-exterior distance) in mm.

    Returns inf if there are no holes.
    """
    dists = wall_to_hole_distances(poly)
    if not dists:
        return float("inf")
    return min(dists)


def hole_to_hole_distances(poly: Polygon | MultiPolygon) -> list[float]:
    """Compute minimum distances between each pair of interior rings."""
    if isinstance(poly, MultiPolygon):
        all_interiors = []
        for p in poly.geoms:
            all_interiors.extend(p.interiors)
    else:
        all_interiors = list(poly.interiors)

    dists = []
    for i in range(len(all_interiors)):
        for j in range(i + 1, len(all_interiors)):
            l1 = LineString(all_interiors[i].coords)
            l2 = LineString(all_interiors[j].coords)
            dists.append(l1.distance(l2))
    return dists


def structural_score(
    poly: Polygon | MultiPolygon,
    min_feature_size_mm: float,
    min_wall_mm: float,
) -> float:
    """Compute a [0, 1] structural integrity score.

    Penalises:
    - Thin cross-sections below min_feature_size_mm
    - Wall thicknesses below min_wall_mm
    - Holes too close to each other
    """
    score = 1.0

    # Cross-section width check
    msw = min_section_width(poly)
    if msw < min_feature_size_mm:
        score *= max(0.0, msw / min_feature_size_mm)

    # Wall thickness check
    mwt = min_wall_thickness(poly)
    if mwt < min_wall_mm:
        score *= max(0.0, mwt / min_wall_mm)

    # Hole-to-hole proximity
    hh_dists = hole_to_hole_distances(poly)
    for d in hh_dists:
        if d < min_wall_mm:
            score *= max(0.0, d / min_wall_mm)

    return np.clip(score, 0.0, 1.0)
