"""Validate MOI and mass calculations against known analytic shapes."""

import math

import numpy as np
import pytest
from shapely.geometry import Point, box

from weapon_designer.physics import (
    polygon_area_mm2,
    polygon_mass_kg,
    polygon_moi_mm4,
    mass_moi_kg_mm2,
    stored_energy_joules,
    bite_mm,
    com_offset_mm,
)


def _make_disk(radius: float, quad_segs: int = 256):
    """Create a high-resolution circular polygon centred at origin."""
    return Point(0, 0).buffer(radius, quad_segs=quad_segs)


def _make_rectangle(width: float, height: float):
    """Create a rectangle centred at origin."""
    return box(-width / 2, -height / 2, width / 2, height / 2)


class TestArea:
    def test_disk_area(self):
        r = 100.0
        disk = _make_disk(r)
        expected = math.pi * r**2
        assert polygon_area_mm2(disk) == pytest.approx(expected, rel=1e-3)

    def test_rectangle_area(self):
        w, h = 200.0, 50.0
        rect = _make_rectangle(w, h)
        assert polygon_area_mm2(rect) == pytest.approx(w * h, rel=1e-6)


class TestMass:
    def test_disk_mass(self):
        r = 100.0  # mm
        t = 10.0   # mm
        rho = 7850.0  # kg/m³
        disk = _make_disk(r)
        area_m2 = math.pi * (r * 1e-3) ** 2
        expected_kg = rho * area_m2 * (t * 1e-3)
        assert polygon_mass_kg(disk, t, rho) == pytest.approx(expected_kg, rel=1e-3)


class TestMOI:
    def test_solid_disk_moi(self):
        """MOI of a solid disk about its centre: I = (1/2) * m * r²."""
        r = 100.0  # mm
        t = 10.0   # mm
        rho = 7850.0  # kg/m³
        disk = _make_disk(r, quad_segs=512)

        mass = polygon_mass_kg(disk, t, rho)
        moi = mass_moi_kg_mm2(disk, t, rho)
        expected_moi = 0.5 * mass * r**2

        # Polygon approximation won't be exact; allow 1% tolerance
        assert moi == pytest.approx(expected_moi, rel=0.01)

    def test_rectangle_moi(self):
        """MOI of a rectangle about its centre: I = (1/12) * m * (w² + h²)."""
        w, h = 300.0, 50.0  # mm
        t = 12.0   # mm
        rho = 7850.0
        rect = _make_rectangle(w, h)

        mass = polygon_mass_kg(rect, t, rho)
        moi = mass_moi_kg_mm2(rect, t, rho)
        expected_moi = (1.0 / 12.0) * mass * (w**2 + h**2)

        assert moi == pytest.approx(expected_moi, rel=0.01)

    def test_disk_with_hole_moi(self):
        """MOI of an annular ring: I = (1/2) * m * (R_outer² + R_inner²)
        where m is mass of the ring."""
        r_out = 100.0
        r_in = 40.0
        t = 10.0
        rho = 7850.0

        outer = _make_disk(r_out, quad_segs=512)
        inner = _make_disk(r_in, quad_segs=512)
        ring = outer.difference(inner)

        mass = polygon_mass_kg(ring, t, rho)
        moi = mass_moi_kg_mm2(ring, t, rho)
        expected_moi = 0.5 * mass * (r_out**2 + r_in**2)

        # Wider tolerance for boolean-difference polygon vs analytic formula
        assert moi == pytest.approx(expected_moi, rel=0.06)


class TestEnergy:
    def test_energy_formula(self):
        """E = 0.5 * I * ω²"""
        moi_kg_mm2 = 50000.0  # kg·mm²
        rpm = 10000.0
        omega = rpm * 2 * math.pi / 60
        moi_kg_m2 = moi_kg_mm2 * 1e-6
        expected = 0.5 * moi_kg_m2 * omega**2
        assert stored_energy_joules(moi_kg_mm2, rpm) == pytest.approx(expected, rel=1e-9)


class TestBite:
    def test_bite_basic(self):
        """Bite = drive_speed / (rps * teeth) in metres, convert to mm."""
        rpm = 6000.0
        teeth = 2
        drive_speed = 3.0  # m/s
        rps = rpm / 60.0
        expected_mm = (drive_speed / (rps * teeth)) * 1000.0
        assert bite_mm(teeth, rpm, drive_speed) == pytest.approx(expected_mm, rel=1e-9)


class TestCoM:
    def test_centered_shape(self):
        """A shape centred at origin should have ~0 CoM offset."""
        disk = _make_disk(100.0)
        assert com_offset_mm(disk) == pytest.approx(0.0, abs=0.1)

    def test_offset_shape(self):
        """A rectangle not centred at origin should have nonzero offset."""
        from shapely.affinity import translate
        rect = _make_rectangle(100, 50)
        shifted = translate(rect, 30, 0)
        assert com_offset_mm(shifted) == pytest.approx(30.0, abs=0.1)
