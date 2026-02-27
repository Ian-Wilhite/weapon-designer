"""Shared fixtures for weapon-designer test suite."""

from __future__ import annotations

import math
import numpy as np
import pytest
from shapely.geometry import Point, box

from weapon_designer.config import (
    WeaponConfig, Material, Mounting, Envelope,
    OptimizationParams, OptimizationWeights, OutputParams,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def make_disk(radius: float, quad_segs: int = 128) -> "Polygon":
    return Point(0, 0).buffer(radius, quad_segs=quad_segs)


def make_annulus(r_outer: float, r_inner: float, quad_segs: int = 128):
    return make_disk(r_outer, quad_segs).difference(make_disk(r_inner, quad_segs))


def make_rect(w: float, h: float):
    return box(-w / 2, -h / 2, w / 2, h / 2)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def disk_cfg():
    """Minimal disk config, 150mm radius, 5kg budget, 8000rpm."""
    return WeaponConfig(
        material=Material(name="AR500", density_kg_m3=7850, yield_strength_mpa=1400),
        weapon_style="disk",
        sheet_thickness_mm=10.0,
        weight_budget_kg=5.0,
        rpm=8000,
        mounting=Mounting(
            bore_diameter_mm=25.4,
            bolt_circle_diameter_mm=50.0,
            num_bolts=4,
            bolt_hole_diameter_mm=6.5,
        ),
        envelope=Envelope(max_radius_mm=150.0),
        optimization=OptimizationParams(
            weights=OptimizationWeights(
                moment_of_inertia=0.30, bite=0.15, structural_integrity=0.20,
                mass_utilization=0.10, balance=0.10, impact_zone=0.15,
            ),
            num_fourier_terms=3,
            num_cutout_pairs=1,
            min_feature_size_mm=3.0,
            min_wall_thickness_mm=5.0,
        ),
    )


@pytest.fixture
def bar_cfg():
    """Bar style config."""
    return WeaponConfig(
        material=Material(name="AR500", density_kg_m3=7850, yield_strength_mpa=1400),
        weapon_style="bar",
        sheet_thickness_mm=12.0,
        weight_budget_kg=5.0,
        rpm=9000,
        mounting=Mounting(
            bore_diameter_mm=19.05,
            bolt_circle_diameter_mm=40.0,
            num_bolts=4,
            bolt_hole_diameter_mm=6.0,
        ),
        envelope=Envelope(max_radius_mm=200.0, max_length_mm=400.0, max_width_mm=70.0),
        optimization=OptimizationParams(
            num_fourier_terms=3, num_cutout_pairs=1,
            min_feature_size_mm=3.0, min_wall_thickness_mm=5.0,
        ),
    )


@pytest.fixture
def simple_disk():
    """Disk polygon centred at origin, r=100mm."""
    return make_disk(100.0)


@pytest.fixture
def simple_annulus():
    """Annular ring: outer 100mm, inner 20mm."""
    return make_annulus(100.0, 20.0)
