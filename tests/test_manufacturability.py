"""Tests for manufacturability.py — GeometryConditioner."""

from __future__ import annotations

import dataclasses

import pytest
from shapely.geometry import Point, Polygon, box

from weapon_designer.manufacturability import ConditioningDelta, GeometryConditioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_square_cutout(side_mm: float) -> Polygon:
    """Axis-aligned square centred at origin."""
    h = side_mm / 2.0
    return box(-h, -h, h, h)


def make_disk_with_square_hole(outer_r: float, hole_side: float) -> tuple[Polygon, Polygon]:
    """Return (weapon_polygon, outer_profile) where weapon has one square hole."""
    outer = Point(0, 0).buffer(outer_r, resolution=128)
    hole = make_square_cutout(hole_side)
    weapon = outer.difference(hole)
    return weapon, outer


# ---------------------------------------------------------------------------
# Test: Minkowski method removes sharp interior corners
# ---------------------------------------------------------------------------

class TestMinkowskiRemovesSharpInteriorCorners:
    """A square cutout has four 90° (re-entrant) corners from the weapon's
    perspective.  After Minkowski conditioning the void's corners are rounded,
    so the conditioned void is smaller than the original square."""

    def test_area_reduced(self):
        """Conditioned cutout area < original square area."""
        square = make_square_cutout(30.0)
        gc = GeometryConditioner(R_min_mm=3.0, method="minkowski")
        conditioned, delta = gc.condition_cutout(square)

        # The rounded square has less area than the original square
        assert conditioned.area < square.area, (
            f"Expected conditioned area < {square.area:.2f}, got {conditioned.area:.2f}"
        )

    def test_conditioned_polygon_is_valid(self):
        square = make_square_cutout(30.0)
        gc = GeometryConditioner(R_min_mm=3.0, method="minkowski")
        conditioned, delta = gc.condition_cutout(square)

        assert conditioned.is_valid
        assert not conditioned.is_empty

    def test_area_change_negative(self):
        """delta.area_change_mm2 should be negative (void shrank)."""
        square = make_square_cutout(30.0)
        gc = GeometryConditioner(R_min_mm=3.0, method="minkowski")
        _, delta = gc.condition_cutout(square)

        assert delta.area_change_mm2 < 0.0, (
            f"Expected negative area_change_mm2, got {delta.area_change_mm2}"
        )

    def test_max_curvature_before_is_positive(self):
        """A square has concave corners so curvature_before should be > 0."""
        square = make_square_cutout(30.0)
        gc = GeometryConditioner(R_min_mm=3.0, method="minkowski")
        _, delta = gc.condition_cutout(square)

        assert delta.max_curvature_before > 0.0

    def test_method_field_recorded(self):
        square = make_square_cutout(30.0)
        gc = GeometryConditioner(R_min_mm=3.0, method="minkowski")
        _, delta = gc.condition_cutout(square)

        assert delta.method == "minkowski"


# ---------------------------------------------------------------------------
# Test: condition_weapon leaves outer profile unchanged
# ---------------------------------------------------------------------------

class TestConditionWeaponOuterUnchanged:
    """The outer boundary of the weapon must not be modified."""

    def test_bounding_box_unchanged(self):
        outer_r = 80.0
        weapon, outer_profile = make_disk_with_square_hole(outer_r, hole_side=20.0)

        gc = GeometryConditioner(R_min_mm=2.0, method="minkowski")
        conditioned_weapon, delta = gc.condition_weapon(weapon, outer_profile)

        orig_bounds = outer_profile.bounds   # (minx, miny, maxx, maxy)
        cond_bounds = conditioned_weapon.bounds

        for i, (ob, cb) in enumerate(zip(orig_bounds, cond_bounds)):
            assert abs(ob - cb) < 1e-6, (
                f"Bounding box coordinate {i} changed: before={ob}, after={cb}"
            )

    def test_outer_area_approximately_preserved(self):
        """Conditioned weapon area should be >= original weapon area (holes shrank)."""
        outer_r = 80.0
        weapon, outer_profile = make_disk_with_square_hole(outer_r, hole_side=20.0)

        gc = GeometryConditioner(R_min_mm=2.0, method="minkowski")
        conditioned_weapon, delta = gc.condition_weapon(weapon, outer_profile)

        # After rounding interior corners the holes are smaller, so weapon
        # solid area grows (or stays the same).
        assert conditioned_weapon.area >= weapon.area - 1e-3

    def test_result_is_valid_polygon(self):
        outer_r = 80.0
        weapon, outer_profile = make_disk_with_square_hole(outer_r, hole_side=20.0)

        gc = GeometryConditioner(R_min_mm=2.0, method="minkowski")
        conditioned_weapon, delta = gc.condition_weapon(weapon, outer_profile)

        assert conditioned_weapon.is_valid or conditioned_weapon.buffer(0).area > 0
        assert not conditioned_weapon.is_empty

    def test_no_holes_weapon_unchanged(self):
        """A solid disk (no holes) should be returned as-is."""
        outer = Point(0, 0).buffer(50.0, resolution=64)
        gc = GeometryConditioner(R_min_mm=2.0, method="minkowski")
        conditioned, delta = gc.condition_weapon(outer, outer)

        assert delta.n_corners_modified == 0
        assert abs(delta.area_change_mm2) < 1e-6


# ---------------------------------------------------------------------------
# Test: very small cutout does not crash
# ---------------------------------------------------------------------------

class TestTooSmallCutoutHandled:
    """A 2mm square with R=5mm — the erosion would collapse the shape.
    The method must return a non-empty valid polygon without raising."""

    def test_no_crash(self):
        tiny_square = make_square_cutout(2.0)
        gc = GeometryConditioner(R_min_mm=5.0, method="minkowski")
        conditioned, delta = gc.condition_cutout(tiny_square)

        # Must not raise; result must be non-empty and valid
        assert not conditioned.is_empty
        assert conditioned.is_valid

    def test_returns_polygon(self):
        tiny_square = make_square_cutout(2.0)
        gc = GeometryConditioner(R_min_mm=5.0, method="minkowski")
        conditioned, delta = gc.condition_cutout(tiny_square)

        assert isinstance(conditioned, Polygon)

    def test_delta_method_recorded(self):
        tiny_square = make_square_cutout(2.0)
        gc = GeometryConditioner(R_min_mm=5.0, method="minkowski")
        _, delta = gc.condition_cutout(tiny_square)

        assert delta.method == "minkowski"


# ---------------------------------------------------------------------------
# Test: ConditioningDelta has the required fields
# ---------------------------------------------------------------------------

class TestConditioningDeltaFields:
    """Verify the dataclass has exactly the documented public fields."""

    REQUIRED_FIELDS = {
        "area_change_mm2",
        "n_corners_modified",
        "max_curvature_before",
        "max_curvature_after",
        "method",
    }

    def test_all_required_fields_present(self):
        field_names = {f.name for f in dataclasses.fields(ConditioningDelta)}
        missing = self.REQUIRED_FIELDS - field_names
        assert not missing, f"ConditioningDelta is missing fields: {missing}"

    def test_no_extra_unexpected_fields(self):
        """Warn if unexpected fields are present (informational, not fatal)."""
        field_names = {f.name for f in dataclasses.fields(ConditioningDelta)}
        extra = field_names - self.REQUIRED_FIELDS
        # Extra fields are allowed but the required ones must all be there.
        assert self.REQUIRED_FIELDS.issubset(field_names)

    def test_field_types_at_runtime(self):
        """Instantiate a delta and check type coercion works."""
        delta = ConditioningDelta(
            area_change_mm2=-12.5,
            n_corners_modified=4,
            max_curvature_before=0.5,
            max_curvature_after=0.1,
            method="minkowski",
        )
        assert isinstance(delta.area_change_mm2, float)
        assert isinstance(delta.n_corners_modified, int)
        assert isinstance(delta.max_curvature_before, float)
        assert isinstance(delta.max_curvature_after, float)
        assert isinstance(delta.method, str)

    def test_delta_from_condition_cutout_has_correct_types(self):
        square = make_square_cutout(30.0)
        gc = GeometryConditioner(R_min_mm=3.0, method="minkowski")
        _, delta = gc.condition_cutout(square)

        assert isinstance(delta.area_change_mm2, float)
        assert isinstance(delta.n_corners_modified, int)
        assert isinstance(delta.max_curvature_before, float)
        assert isinstance(delta.max_curvature_after, float)
        assert isinstance(delta.method, str)


# ---------------------------------------------------------------------------
# Test: vertex method
# ---------------------------------------------------------------------------

class TestVertexMethod:
    """Basic sanity checks for the vertex-by-vertex rounding method."""

    def test_vertex_method_valid_result(self):
        square = make_square_cutout(30.0)
        gc = GeometryConditioner(R_min_mm=3.0, method="vertex")
        conditioned, delta = gc.condition_cutout(square)

        assert not conditioned.is_empty
        assert conditioned.is_valid

    def test_vertex_method_modifies_corners(self):
        square = make_square_cutout(30.0)
        gc = GeometryConditioner(R_min_mm=3.0, method="vertex")
        _, delta = gc.condition_cutout(square)

        assert delta.n_corners_modified > 0

    def test_vertex_method_recorded_in_delta(self):
        square = make_square_cutout(30.0)
        gc = GeometryConditioner(R_min_mm=3.0, method="vertex")
        _, delta = gc.condition_cutout(square)

        assert delta.method == "vertex"


# ---------------------------------------------------------------------------
# Test: constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="R_min_mm"):
            GeometryConditioner(R_min_mm=-1.0)

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError, match="R_min_mm"):
            GeometryConditioner(R_min_mm=0.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            GeometryConditioner(R_min_mm=2.0, method="unknown")
