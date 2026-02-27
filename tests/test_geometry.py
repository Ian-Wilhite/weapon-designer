"""Tests for geometry.py — assemble_weapon(), mounting features, boolean ops."""

from __future__ import annotations

import math
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from weapon_designer.config import Mounting
from weapon_designer.geometry import (
    make_bore,
    make_bolt_holes,
    make_mounting_cutouts,
    check_mounting_clearance,
    assemble_weapon,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mounting():
    return Mounting(
        bore_diameter_mm=25.4,
        bolt_circle_diameter_mm=50.0,
        num_bolts=4,
        bolt_hole_diameter_mm=6.5,
    )


@pytest.fixture
def outer_disk():
    """150mm radius disk centred at origin."""
    return Point(0, 0).buffer(150, resolution=128)


# ---------------------------------------------------------------------------
# make_bore
# ---------------------------------------------------------------------------

class TestMakeBore:
    def test_bore_radius(self, mounting):
        bore = make_bore(mounting)
        expected_r = mounting.bore_diameter_mm / 2
        bounds = bore.bounds
        # centred at origin
        assert abs(bounds[0] + expected_r) < 1.0
        assert abs(bounds[2] - expected_r) < 1.0

    def test_bore_is_polygon(self, mounting):
        bore = make_bore(mounting)
        assert isinstance(bore, Polygon)
        assert not bore.is_empty

    def test_bore_area(self, mounting):
        bore = make_bore(mounting)
        r = mounting.bore_diameter_mm / 2
        expected = math.pi * r**2
        assert bore.area == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# make_bolt_holes
# ---------------------------------------------------------------------------

class TestMakeBoltHoles:
    def test_count(self, mounting):
        holes = make_bolt_holes(mounting)
        assert len(holes) == mounting.num_bolts

    def test_hole_radius(self, mounting):
        holes = make_bolt_holes(mounting)
        r_expected = mounting.bolt_hole_diameter_mm / 2
        for h in holes:
            # area ≈ π r²
            assert h.area == pytest.approx(math.pi * r_expected**2, rel=0.02)

    def test_holes_on_bolt_circle(self, mounting):
        """Each hole centre should lie on the bolt circle."""
        holes = make_bolt_holes(mounting)
        r_bc = mounting.bolt_circle_diameter_mm / 2
        for h in holes:
            cx, cy = h.centroid.x, h.centroid.y
            dist = math.hypot(cx, cy)
            assert dist == pytest.approx(r_bc, abs=0.5)

    def test_holes_angularly_uniform(self, mounting):
        """Holes should be equally spaced angularly."""
        holes = make_bolt_holes(mounting)
        angles = sorted([
            math.atan2(h.centroid.y, h.centroid.x) for h in holes
        ])
        expected_step = 2 * math.pi / mounting.num_bolts
        diffs = [angles[i+1] - angles[i] for i in range(len(angles)-1)]
        for d in diffs:
            assert d == pytest.approx(expected_step, abs=0.01)


# ---------------------------------------------------------------------------
# make_mounting_cutouts
# ---------------------------------------------------------------------------

class TestMakeMountingCutouts:
    def test_is_polygon_or_multipolygon(self, mounting):
        from shapely.geometry import MultiPolygon
        mc = make_mounting_cutouts(mounting)
        assert isinstance(mc, (Polygon, MultiPolygon))

    def test_covers_bore(self, mounting):
        mc = make_mounting_cutouts(mounting)
        bore = make_bore(mounting)
        assert mc.contains(bore) or mc.intersection(bore).area == pytest.approx(bore.area, rel=0.01)

    def test_covers_bolt_holes(self, mounting):
        mc = make_mounting_cutouts(mounting)
        holes = make_bolt_holes(mounting)
        for h in holes:
            inter = mc.intersection(h).area
            assert inter == pytest.approx(h.area, rel=0.01)


# ---------------------------------------------------------------------------
# check_mounting_clearance
# ---------------------------------------------------------------------------

class TestCheckMountingClearance:
    def test_no_cutouts(self, mounting):
        assert check_mounting_clearance(mounting, []) is True

    def test_far_cutout_ok(self, mounting):
        far_pocket = Point(100, 0).buffer(5)
        assert check_mounting_clearance(mounting, [far_pocket]) is True

    def test_overlapping_cutout_fails(self, mounting):
        # Place a cutout right on top of the bore
        overlap = Point(0, 0).buffer(5)
        assert check_mounting_clearance(mounting, [overlap]) is False

    def test_too_close_fails(self, mounting):
        # Place a cutout just inside the minimum clearance zone
        bore_r = mounting.bore_diameter_mm / 2
        near = Point(bore_r + 0.5, 0).buffer(2)
        assert check_mounting_clearance(mounting, [near], min_clearance_mm=3.0) is False


# ---------------------------------------------------------------------------
# assemble_weapon
# ---------------------------------------------------------------------------

class TestAssembleWeapon:
    def test_result_smaller_than_outer(self, mounting, outer_disk):
        weapon = assemble_weapon(outer_disk, mounting)
        assert weapon.area < outer_disk.area

    def test_bore_subtracted(self, mounting, outer_disk):
        weapon = assemble_weapon(outer_disk, mounting)
        bore = make_bore(mounting)
        # bore centroid should not be inside weapon
        assert not weapon.contains(Point(0, 0))

    def test_no_cutouts_returns_valid(self, mounting, outer_disk):
        weapon = assemble_weapon(outer_disk, mounting)
        assert weapon.is_valid or weapon.buffer(0).area > 0

    def test_with_cutouts_reduces_area(self, mounting, outer_disk):
        pocket = Point(80, 0).buffer(15)
        no_pocket = assemble_weapon(outer_disk, mounting)
        with_pocket = assemble_weapon(outer_disk, mounting, [pocket])
        assert with_pocket.area < no_pocket.area

    def test_cutout_hole_removed(self, mounting, outer_disk):
        pocket = Point(80, 0).buffer(10)
        weapon = assemble_weapon(outer_disk, mounting, [pocket])
        # The centre of the pocket should not be inside the weapon
        assert not weapon.contains(Point(80, 0))
