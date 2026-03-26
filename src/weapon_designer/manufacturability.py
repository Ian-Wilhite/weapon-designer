"""Manufacturability conditioning for weapon geometry.

Applies minimum concave-radius conditioning to interior cutout boundaries so
that all re-entrant corners have a radius of curvature no smaller than R_min.
The outer profile is never modified.

Two methods are supported:

  "minkowski"  — erode by -R_min (join_style=2, i.e. mitre) then dilate by
                 +R_min (join_style=1, i.e. round).  Fast, robust, always
                 produces a convex-rounded result.  If erosion collapses the
                 shape, the original is returned unchanged.

  "vertex"     — classify each vertex of the input polygon by its signed
                 turning angle.  Concave vertices (turning angle < -threshold)
                 are individually replaced by circular arcs of radius R_min.
                 Gives finer control at the cost of more code.

All coordinates and distances are in millimetres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class ConditioningDelta:
    """Summary of changes made by a conditioning pass.

    Attributes
    ----------
    area_change_mm2
        Change in area of the conditioned region (conditioned - original).
        Negative means material was added to fill sharp corners; for cutouts
        this means the void area shrank (material was restored to the weapon).
    n_corners_modified
        Number of concave vertices that were modified.
    max_curvature_before
        Maximum concave curvature (1/R mm⁻¹) found before conditioning.
        0.0 if no concave corners were detected.
    max_curvature_after
        Maximum concave curvature remaining after conditioning.
    method
        The method name that was applied ("minkowski" or "vertex").
    """
    area_change_mm2: float
    n_corners_modified: int
    max_curvature_before: float
    max_curvature_after: float
    method: str


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GeometryConditioner:
    """Apply minimum-concave-radius conditioning to weapon geometry.

    Parameters
    ----------
    R_min_mm
        Minimum allowed concave radius in mm.  Sharp interior corners whose
        radius of curvature is smaller than this will be rounded.
    method
        "minkowski" (default) or "vertex".

    Usage
    -----
    >>> gc = GeometryConditioner(R_min_mm=2.0)
    >>> conditioned_hole, delta = gc.condition_cutout(square_hole)
    >>> conditioned_weapon, delta = gc.condition_weapon(weapon_poly, outer_profile)
    """

    def __init__(self, R_min_mm: float = 2.0, method: str = "minkowski") -> None:
        if R_min_mm <= 0.0:
            raise ValueError(f"R_min_mm must be positive, got {R_min_mm}")
        if method not in ("minkowski", "vertex"):
            raise ValueError(f"method must be 'minkowski' or 'vertex', got {method!r}")
        self.R_min_mm = float(R_min_mm)
        self.method = method

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def condition_cutout(self, cutout_poly: Polygon) -> tuple[Polygon, ConditioningDelta]:
        """Condition a single interior void polygon.

        Parameters
        ----------
        cutout_poly
            A Shapely Polygon representing one interior cutout (void).

        Returns
        -------
        (conditioned_polygon, delta)
        """
        curv_before = self._measure_max_curvature(cutout_poly)

        if self.method == "minkowski":
            conditioned = self._minkowski_method(cutout_poly)
            n_mod = _count_concave_vertices(cutout_poly, threshold_rad=0.0)
        else:
            conditioned, n_mod = self._vertex_method(cutout_poly)

        curv_after = self._measure_max_curvature(conditioned)
        area_change = conditioned.area - cutout_poly.area

        delta = ConditioningDelta(
            area_change_mm2=area_change,
            n_corners_modified=n_mod,
            max_curvature_before=curv_before,
            max_curvature_after=curv_after,
            method=self.method,
        )
        return conditioned, delta

    def condition_weapon(
        self,
        weapon: Polygon,
        outer_profile: Polygon,
    ) -> tuple[Polygon, ConditioningDelta]:
        """Condition all interior rings (holes) of an assembled weapon polygon.

        The outer profile is never touched.  Each interior ring (cutout) is
        conditioned independently and the result is rebuilt by subtracting all
        conditioned holes from *outer_profile*.

        Parameters
        ----------
        weapon
            Assembled weapon polygon that may have interior rings (holes).
        outer_profile
            The exterior boundary polygon.  Used as the base for reconstruction
            so that the outer shape is guaranteed unchanged.

        Returns
        -------
        (conditioned_weapon, delta)
            delta summarises the cumulative change across all interior rings.
        """
        interiors = list(weapon.interiors)

        if not interiors:
            # No holes — nothing to condition.
            delta = ConditioningDelta(
                area_change_mm2=0.0,
                n_corners_modified=0,
                max_curvature_before=0.0,
                max_curvature_after=0.0,
                method=self.method,
            )
            return weapon, delta

        total_area_change = 0.0
        total_n_mod = 0
        max_curv_before = 0.0
        max_curv_after = 0.0

        conditioned_holes: list[Polygon] = []

        for ring in interiors:
            # Build a filled polygon from the interior ring so we can apply
            # standard Shapely buffer operations on it.
            hole_poly = Polygon(ring)
            if not hole_poly.is_valid:
                hole_poly = hole_poly.buffer(0)
            if hole_poly.is_empty:
                continue

            curv_before = self._measure_max_curvature(hole_poly)
            max_curv_before = max(max_curv_before, curv_before)

            if self.method == "minkowski":
                cond = self._minkowski_method(hole_poly)
                n_mod = _count_concave_vertices(hole_poly, threshold_rad=0.0)
            else:
                cond, n_mod = self._vertex_method(hole_poly)

            curv_after = self._measure_max_curvature(cond)
            max_curv_after = max(max_curv_after, curv_after)

            # Area change: conditioned void is smaller → weapon gains material
            total_area_change += cond.area - hole_poly.area
            total_n_mod += n_mod
            conditioned_holes.append(cond)

        # Reconstruct weapon: outer_profile minus all conditioned holes.
        if conditioned_holes:
            holes_union = unary_union(conditioned_holes)
            result = outer_profile.difference(holes_union)
        else:
            result = outer_profile

        if not result.is_valid:
            result = result.buffer(0)

        delta = ConditioningDelta(
            area_change_mm2=total_area_change,
            n_corners_modified=total_n_mod,
            max_curvature_before=max_curv_before,
            max_curvature_after=max_curv_after,
            method=self.method,
        )
        return result, delta

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _minkowski_method(self, poly: Polygon) -> Polygon:
        """Round concave corners via erosion-then-dilation.

        Erosion (negative buffer, join_style=2 / mitre) peels back the shape
        uniformly including into concave corners.  Dilation (positive buffer,
        join_style=1 / round) re-expands convex edges but leaves concave
        corners rounded at radius R_min.

        If the erosion would collapse the shape, the original is returned.
        """
        R = self.R_min_mm

        # Erode: mitre join to avoid artefacts on convex corners during erosion
        eroded = poly.buffer(-R, join_style=2)

        if eroded is None or eroded.is_empty or not eroded.is_valid:
            return poly

        # For MultiPolygon results take the largest fragment
        if isinstance(eroded, MultiPolygon):
            eroded = max(eroded.geoms, key=lambda g: g.area)

        if eroded.area < 1.0:
            return poly

        # Dilate: round join smooths the re-expanded boundary
        dilated = eroded.buffer(R, join_style=1)

        if dilated is None or dilated.is_empty or not dilated.is_valid:
            return poly

        if isinstance(dilated, MultiPolygon):
            dilated = max(dilated.geoms, key=lambda g: g.area)

        return dilated

    def _vertex_method(self, poly: Polygon) -> tuple[Polygon, int]:
        """Round corners that are concave from the material's perspective.

        When the input polygon is a cutout (void region), corners that are
        convex in the void polygon are re-entrant (concave) from the
        surrounding material's perspective — these are the stress risers we
        want to soften.

        Shapely exterior rings are wound CCW, so convex corners of the void
        have a positive turning angle.  We place a small circular disc of
        radius R_min at each such corner and take the union with the void
        polygon, which grows the void slightly to round the sharp notch.

        For a solid polygon (no enclosing material context), corners with
        negative turning angle (concave indentations into the solid) are
        targeted instead.

        Returns (conditioned_polygon, n_corners_modified).
        """
        R = self.R_min_mm
        coords = np.array(poly.exterior.coords[:-1])  # drop closing duplicate
        n = len(coords)
        if n < 3:
            return poly, 0

        # Determine which sign of turning angle corresponds to "concave from
        # the material view".  For a CCW-wound polygon being used as a cutout,
        # positive turning angles are convex corners of the void (= re-entrant
        # notches from the material).  For a CW-wound polygon (e.g. an outer
        # profile being conditioned directly) negative angles are the concave
        # indentations.
        is_ccw = poly.exterior.is_ccw
        # Convex corners of the void (CCW polygon) → concave notches in material
        # Concave corners of a solid (CW polygon) → concave indentations
        # In both cases, if we want the corners that bite into the material:
        #   CCW polygon (void): positive turning angle
        #   CW polygon (solid): negative turning angle
        target_sign = 1 if is_ccw else -1

        fills: list[Polygon] = []
        n_mod = 0

        for i in range(n):
            prev_pt = coords[(i - 1) % n]
            curr_pt = coords[i]
            next_pt = coords[(i + 1) % n]

            angle = _signed_turning_angle(prev_pt, curr_pt, next_pt)

            # Detect corners that represent concave notches from material's view.
            if target_sign * angle > 1e-6:
                # Place a circular disc to fill/round the sharp notch.
                disc = Point(curr_pt[0], curr_pt[1]).buffer(R)
                fills.append(disc)
                n_mod += 1

        if not fills:
            return poly, 0

        result = unary_union([poly] + fills)

        if isinstance(result, MultiPolygon):
            result = max(result.geoms, key=lambda g: g.area)

        if not result.is_valid:
            result = result.buffer(0)

        return result, n_mod

    def _measure_max_curvature(self, poly: Polygon) -> float:
        """Approximate maximum curvature (1/R mm⁻¹) at corners that represent
        concave notches from the material's perspective.

        For a void polygon (CCW winding): convex corners of the void are
        re-entrant notches in the material → positive turning angle.
        For a solid polygon (CW winding): concave indentations → negative angle.

        For a vertex with turning angle α, a rough estimate of the local radius
        of curvature is:

            R_approx = L / (2 * sin(|α| / 2))

        where L is the average of the two adjacent edge lengths.
        Curvature = 1 / R_approx.

        Returns 0.0 if no such corners are detected.
        """
        coords = np.array(poly.exterior.coords[:-1])
        n = len(coords)
        if n < 3:
            return 0.0

        is_ccw = poly.exterior.is_ccw
        target_sign = 1 if is_ccw else -1

        max_curv = 0.0

        for i in range(n):
            prev_pt = coords[(i - 1) % n]
            curr_pt = coords[i]
            next_pt = coords[(i + 1) % n]

            angle = _signed_turning_angle(prev_pt, curr_pt, next_pt)

            # Only consider corners that are concave from material's view.
            if target_sign * angle <= 1e-6:
                continue

            abs_angle = abs(angle)
            if abs_angle < 1e-9:
                continue

            # Edge lengths on either side of the vertex
            L1 = math.hypot(curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
            L2 = math.hypot(next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1])
            L_avg = (L1 + L2) / 2.0

            if L_avg < 1e-12:
                continue

            half_angle = abs_angle / 2.0
            sin_half = math.sin(half_angle)
            if sin_half < 1e-12:
                continue

            R_approx = L_avg / (2.0 * sin_half)
            curvature = 1.0 / R_approx if R_approx > 1e-12 else 0.0
            max_curv = max(max_curv, curvature)

        return max_curv


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _signed_turning_angle(prev_pt: np.ndarray, curr_pt: np.ndarray, next_pt: np.ndarray) -> float:
    """Signed turning angle at curr_pt (radians).

    Positive for a left (CCW) turn, negative for a right (CW/concave) turn
    when the polygon exterior is wound CCW (Shapely default).

    Uses the cross product of the two edge vectors.
    """
    v1 = curr_pt - prev_pt
    v2 = next_pt - curr_pt

    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    dot   = float(v1[0] * v2[0] + v1[1] * v2[1])

    return math.atan2(cross, dot)


def _count_concave_vertices(poly: Polygon, threshold_rad: float = 0.0) -> int:
    """Count vertices that represent concave notches from the material's view.

    For a CCW-wound polygon (used as a void/cutout): convex corners of the
    void (positive turning angle > threshold_rad) are re-entrant notches from
    the surrounding material's perspective.

    For a CW-wound polygon (solid): concave indentations have negative turning
    angle (< -threshold_rad).
    """
    coords = np.array(poly.exterior.coords[:-1])
    n = len(coords)
    is_ccw = poly.exterior.is_ccw
    target_sign = 1 if is_ccw else -1
    count = 0
    for i in range(n):
        angle = _signed_turning_angle(
            coords[(i - 1) % n],
            coords[i],
            coords[(i + 1) % n],
        )
        if target_sign * angle > threshold_rad:
            count += 1
    return count
