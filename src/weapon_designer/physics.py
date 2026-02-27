"""Mass, MOI (polygon vertex formula), stored energy, bite, and centre of mass."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, MultiPolygon


def polygon_area_mm2(poly: Polygon | MultiPolygon) -> float:
    """Signed area in mm² (positive for CCW)."""
    return abs(poly.area)


def polygon_mass_kg(
    poly: Polygon | MultiPolygon, thickness_mm: float, density_kg_m3: float
) -> float:
    """Mass in kg from a 2D polygon extruded to thickness."""
    area_m2 = polygon_area_mm2(poly) * 1e-6
    thickness_m = thickness_mm * 1e-3
    return density_kg_m3 * area_m2 * thickness_m


def _ring_moi(coords: np.ndarray) -> float:
    """MOI about origin for a single polygon ring using the vertex formula.

    Uses the shoelace-style summation:
        I = (1/12) * Σ |xᵢyᵢ₊₁ - xᵢ₊₁yᵢ| *
            (xᵢ² + xᵢxᵢ₊₁ + xᵢ₊₁² + yᵢ² + yᵢyᵢ₊₁ + yᵢ₊₁²)

    Returns MOI in mm⁴ (area moment, needs multiplication by ρ*t for mass MOI).
    """
    x = coords[:, 0]
    y = coords[:, 1]
    # Shifted arrays
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    terms = x**2 + x * x1 + x1**2 + y**2 + y * y1 + y1**2
    return np.sum(cross * terms) / 12.0


def polygon_moi_mm4(poly: Polygon | MultiPolygon) -> float:
    """Second moment of area about the origin (mm⁴) for a polygon with holes.

    Positive for exterior rings, negative for interior rings (holes).
    """
    if isinstance(poly, MultiPolygon):
        return sum(polygon_moi_mm4(p) for p in poly.geoms)

    coords_ext = np.array(poly.exterior.coords[:-1])
    moi = _ring_moi(coords_ext)

    for interior in poly.interiors:
        coords_int = np.array(interior.coords[:-1])
        moi -= abs(_ring_moi(coords_int))

    return abs(moi)


def mass_moi_kg_mm2(
    poly: Polygon | MultiPolygon, thickness_mm: float, density_kg_m3: float
) -> float:
    """Mass moment of inertia about the spin axis (kg·mm²).

    I_mass = ρ * t * I_area  (with consistent units).
    """
    area_moi_mm4 = polygon_moi_mm4(poly)
    # ρ in kg/mm³, t in mm → ρ*t has units kg/mm²
    density_kg_mm3 = density_kg_m3 * 1e-9
    return density_kg_mm3 * thickness_mm * area_moi_mm4


def stored_energy_joules(moi_kg_mm2: float, rpm: float) -> float:
    """Rotational kinetic energy E = ½Iω² in joules.

    moi_kg_mm2 is converted to kg·m² internally.
    """
    moi_kg_m2 = moi_kg_mm2 * 1e-6
    omega = rpm * 2 * np.pi / 60
    return 0.5 * moi_kg_m2 * omega**2


def bite_mm(num_teeth: int, rpm: float, drive_speed_mps: float = 3.0) -> float:
    """Bite distance in mm.

    bite = (drive_speed / (rpm/60)) / num_teeth * 1000
    Lower bite → more hits but less energy per hit.
    """
    if num_teeth <= 0 or rpm <= 0:
        return 0.0
    rps = rpm / 60.0
    bite_m = drive_speed_mps / (rps * num_teeth)
    return bite_m * 1000.0


def centre_of_mass_mm(poly: Polygon | MultiPolygon) -> tuple[float, float]:
    """Return (cx, cy) of the polygon centroid in mm."""
    c = poly.centroid
    return (c.x, c.y)


def com_offset_mm(poly: Polygon | MultiPolygon) -> float:
    """Distance of the centre of mass from the origin (spin axis)."""
    cx, cy = centre_of_mass_mm(poly)
    return np.hypot(cx, cy)
