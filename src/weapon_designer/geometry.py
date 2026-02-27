"""Mounting features and boolean geometry assembly."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .config import Mounting


def make_bore(mounting: Mounting) -> Polygon:
    """Create the centre bore circle as a Shapely polygon."""
    return Point(0, 0).buffer(mounting.bore_diameter_mm / 2, resolution=64)


def make_bolt_holes(mounting: Mounting) -> list[Polygon]:
    """Create bolt holes arranged on the bolt circle."""
    r = mounting.bolt_circle_diameter_mm / 2
    holes = []
    for i in range(mounting.num_bolts):
        angle = 2 * np.pi * i / mounting.num_bolts
        cx = r * np.cos(angle)
        cy = r * np.sin(angle)
        holes.append(
            Point(cx, cy).buffer(mounting.bolt_hole_diameter_mm / 2, resolution=32)
        )
    return holes


def make_mounting_cutouts(mounting: Mounting) -> Polygon:
    """Return the union of bore + bolt holes as a single polygon to subtract."""
    parts = [make_bore(mounting)] + make_bolt_holes(mounting)
    return unary_union(parts)


def check_mounting_clearance(
    mounting: Mounting,
    cutout_polys: list[Polygon],
    min_clearance_mm: float = 3.0,
) -> bool:
    """Check that no cutout polygon is within min_clearance_mm of any mounting hole.

    Returns True if all clearances are satisfied, False if any violation.
    """
    if not cutout_polys:
        return True

    bore = make_bore(mounting)
    bolt_holes = make_bolt_holes(mounting)
    mounting_features = [bore] + bolt_holes

    for cutout in cutout_polys:
        for feat in mounting_features:
            dist = cutout.distance(feat)
            if dist < min_clearance_mm:
                return False
    return True


def assemble_weapon(
    outer_profile: Polygon,
    mounting: Mounting,
    cutout_polygons: list[Polygon] | None = None,
) -> Polygon:
    """Subtract mounting features and optional cutouts from the outer profile.

    Returns the final weapon polygon ready for export.
    """
    to_subtract = [make_mounting_cutouts(mounting)]
    if cutout_polygons:
        to_subtract.extend(cutout_polygons)
    subtraction = unary_union(to_subtract)
    result = outer_profile.difference(subtraction)
    return result
