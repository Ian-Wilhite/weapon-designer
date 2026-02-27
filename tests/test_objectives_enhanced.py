"""Tests for objectives_enhanced.py — spiral bite and enhanced metrics."""

from __future__ import annotations

import math
import numpy as np
import pytest
from shapely.geometry import Point
from shapely.affinity import scale

from weapon_designer.objectives_enhanced import (
    kinematic_spiral_bite,
    detect_teeth,
)


@pytest.fixture
def smooth_disk():
    """Smooth circular disk: r=100mm."""
    return Point(0, 0).buffer(100, resolution=256)


@pytest.fixture
def toothed_disk():
    """Star-shaped polygon with 6 clear protrusions — multiple contacts expected.

    r(θ) oscillates between 50mm (valleys) and 80mm (peaks) at 6× frequency.
    This creates a genuine multi-toothed radial profile visible to the spiral
    contact simulation, unlike bumps unioned onto a large base disk (which fills
    in the valleys and looks smooth to the radial sampler).
    """
    from shapely.geometry import Polygon

    n_teeth = 6
    n_pts = 720
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r = 50.0 + 30.0 * np.maximum(0.0, np.cos(n_teeth * theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    coords = list(zip(x.tolist(), y.tolist()))
    coords.append(coords[0])
    return Polygon(coords)


# ---------------------------------------------------------------------------
# kinematic_spiral_bite
# ---------------------------------------------------------------------------

class TestKinematicSpiralBite:
    def test_returns_required_keys(self, smooth_disk):
        result = kinematic_spiral_bite(smooth_disk, rpm=8000)
        for key in ("bite_mm", "n_contacts", "v_per_rad_mm", "max_bite_mm"):
            assert key in result

    def test_smooth_disk_max_bite(self, smooth_disk):
        """A smooth disk should have 1 contact → maximum possible bite."""
        result = kinematic_spiral_bite(smooth_disk, rpm=8000, drive_speed_mps=3.0)
        assert result["n_contacts"] == 1
        assert result["bite_mm"] == pytest.approx(result["max_bite_mm"], rel=0.01)

    def test_toothed_disk_more_contacts(self, toothed_disk):
        """A toothed profile should generate more than 1 contact."""
        result = kinematic_spiral_bite(toothed_disk, rpm=8000, drive_speed_mps=3.0)
        assert result["n_contacts"] > 1

    def test_toothed_less_bite_than_smooth(self, smooth_disk, toothed_disk):
        """More contacts → less bite per contact."""
        r_smooth = kinematic_spiral_bite(smooth_disk, rpm=8000, drive_speed_mps=3.0)
        r_toothed = kinematic_spiral_bite(toothed_disk, rpm=8000, drive_speed_mps=3.0)
        assert r_toothed["bite_mm"] < r_smooth["bite_mm"]

    def test_bite_positive(self, smooth_disk):
        result = kinematic_spiral_bite(smooth_disk, rpm=8000)
        assert result["bite_mm"] > 0

    def test_v_per_rad_formula(self, smooth_disk):
        """v_per_rad = v_mm_s / omega."""
        rpm = 8000
        v_mps = 3.0
        omega = 2 * math.pi * rpm / 60
        expected_v_per_rad = (v_mps * 1000) / omega
        result = kinematic_spiral_bite(smooth_disk, rpm=rpm, drive_speed_mps=v_mps)
        assert result["v_per_rad_mm"] == pytest.approx(expected_v_per_rad, rel=0.01)

    def test_max_bite_formula(self, smooth_disk):
        """max_bite_mm = v_per_rad_mm * 2π."""
        result = kinematic_spiral_bite(smooth_disk, rpm=8000, drive_speed_mps=3.0)
        assert result["max_bite_mm"] == pytest.approx(
            result["v_per_rad_mm"] * 2 * math.pi, rel=0.01
        )

    def test_higher_rpm_lower_v_per_rad(self, smooth_disk):
        """Higher RPM → lower advance per radian → smaller bite window."""
        r_lo = kinematic_spiral_bite(smooth_disk, rpm=2000, drive_speed_mps=3.0)
        r_hi = kinematic_spiral_bite(smooth_disk, rpm=10000, drive_speed_mps=3.0)
        assert r_hi["v_per_rad_mm"] < r_lo["v_per_rad_mm"]


# ---------------------------------------------------------------------------
# detect_teeth
# ---------------------------------------------------------------------------

class TestDetectTeeth:
    def test_smooth_disk_no_teeth(self, smooth_disk):
        result = detect_teeth(smooth_disk, min_prominence_mm=5.0)
        # A smooth disk has no prominent teeth
        assert result["n_teeth"] == 0

    def test_toothed_disk_detects_teeth(self, toothed_disk):
        result = detect_teeth(toothed_disk, min_prominence_mm=3.0)
        assert result["n_teeth"] > 0

    def test_returns_required_keys(self, smooth_disk):
        result = detect_teeth(smooth_disk)
        for key in ("n_teeth", "mean_height_mm", "mean_sharpness", "peak_angles_rad"):
            assert key in result

    def test_mean_height_nonneg(self, toothed_disk):
        result = detect_teeth(toothed_disk, min_prominence_mm=3.0)
        assert result["mean_height_mm"] >= 0
