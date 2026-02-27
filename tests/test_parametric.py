"""Tests for parametric.py — Fourier profiles and Fourier cutouts (BASELINE)."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon, Point

from weapon_designer.parametric import (
    fourier_radius,
    make_disk_profile,
    make_bar_profile,
    make_eggbeater_profile,
    make_fourier_cutout,
    make_cutouts,
    build_weapon_polygon,
)


# ---------------------------------------------------------------------------
# fourier_radius
# ---------------------------------------------------------------------------

class TestFourierRadius:
    def test_zero_coefficients_returns_r_base(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        r = fourier_radius(theta, 100.0, np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(r, 100.0)

    def test_cos_term_modulates(self):
        theta = np.array([0.0, np.pi])
        cos_c = np.array([10.0])
        sin_c = np.array([0.0])
        r = fourier_radius(theta, 100.0, cos_c, sin_c)
        # At theta=0: r = 100 + 10*cos(0) = 110
        # At theta=pi: r = 100 + 10*cos(pi) = 90
        assert r[0] == pytest.approx(110.0)
        assert r[1] == pytest.approx(90.0)

    def test_output_shape_matches_input(self):
        theta = np.linspace(0, 2 * np.pi, 720)
        r = fourier_radius(theta, 50.0, np.zeros(4), np.zeros(4))
        assert r.shape == theta.shape


# ---------------------------------------------------------------------------
# make_disk_profile
# ---------------------------------------------------------------------------

class TestMakeDiskProfile:
    def test_returns_polygon(self):
        p = make_disk_profile(80.0, np.zeros(3), np.zeros(3), 150.0)
        assert isinstance(p, Polygon)
        assert not p.is_empty

    def test_profile_is_valid(self):
        p = make_disk_profile(80.0, np.zeros(3), np.zeros(3), 150.0)
        assert p.is_valid or p.buffer(0).area > 0

    def test_radius_clamped_to_max(self):
        # Large coefficients push radius over max — should be clamped
        big_coeff = np.array([200.0])
        p = make_disk_profile(80.0, big_coeff, np.zeros(1), 100.0)
        # Bounding box extent should not exceed max_radius significantly
        bounds = p.bounds
        assert max(abs(bounds[0]), abs(bounds[1]), abs(bounds[2]), abs(bounds[3])) <= 110.0

    def test_area_positive(self):
        p = make_disk_profile(80.0, np.array([5.0, 3.0]), np.array([2.0, 1.0]), 150.0)
        assert p.area > 0


# ---------------------------------------------------------------------------
# make_bar_profile
# ---------------------------------------------------------------------------

class TestMakeBarProfile:
    def test_returns_polygon(self):
        p = make_bar_profile(300.0, 50.0, np.zeros(3), 400.0, 70.0)
        assert isinstance(p, Polygon)
        assert not p.is_empty

    def test_length_within_limit(self):
        p = make_bar_profile(300.0, 50.0, np.zeros(2), 400.0, 70.0)
        bounds = p.bounds
        length = bounds[2] - bounds[0]
        assert length <= 400.0 * 1.1  # slight tolerance for tip sculpting

    def test_bar_area_positive(self):
        p = make_bar_profile(250.0, 40.0, np.zeros(3), 400.0, 70.0)
        assert p.area > 0

    def test_bar_length_clamped(self):
        # Requesting length > max -> clamped
        p1 = make_bar_profile(500.0, 50.0, np.zeros(2), 400.0, 70.0)
        p2 = make_bar_profile(400.0, 50.0, np.zeros(2), 400.0, 70.0)
        bounds1 = p1.bounds
        length1 = bounds1[2] - bounds1[0]
        assert length1 <= 410.0


# ---------------------------------------------------------------------------
# make_eggbeater_profile
# ---------------------------------------------------------------------------

class TestMakeEggbeaterProfile:
    def test_returns_valid_polygon(self):
        from shapely.geometry import MultiPolygon
        p = make_eggbeater_profile(80.0, np.zeros(3), np.zeros(3), 3, 130.0)
        assert isinstance(p, (Polygon, MultiPolygon))
        assert not p.is_empty

    def test_two_blades(self):
        from shapely.geometry import MultiPolygon
        p = make_eggbeater_profile(80.0, np.zeros(3), np.zeros(3), 2, 130.0)
        assert p.area > 0

    def test_four_blades(self):
        from shapely.geometry import MultiPolygon
        p = make_eggbeater_profile(80.0, np.zeros(3), np.zeros(3), 4, 130.0)
        assert p.area > 0


# ---------------------------------------------------------------------------
# make_fourier_cutout
# ---------------------------------------------------------------------------

class TestMakeFourierCutout:
    def test_basic_cutout(self):
        cutout = make_fourier_cutout(50.0, 0.0, 15.0, np.zeros(2), np.zeros(2))
        assert cutout is not None
        assert cutout.area > 0

    def test_zero_radius_returns_none(self):
        result = make_fourier_cutout(50.0, 0.0, 0.5, np.zeros(2), np.zeros(2))
        assert result is None

    def test_cutout_is_valid(self):
        cutout = make_fourier_cutout(50.0, 0.0, 15.0, np.zeros(2), np.zeros(2))
        assert cutout is not None
        assert cutout.is_valid or cutout.buffer(0).area > 0

    def test_cutout_centred_at_cx_cy(self):
        cutout = make_fourier_cutout(60.0, 20.0, 10.0, np.zeros(1), np.zeros(1))
        assert cutout is not None
        cx, cy = cutout.centroid.x, cutout.centroid.y
        assert abs(cx - 60.0) < 3.0
        assert abs(cy - 20.0) < 3.0


# ---------------------------------------------------------------------------
# make_cutouts
# ---------------------------------------------------------------------------

class TestMakeCutouts:
    def test_empty_params_returns_empty(self):
        params = np.zeros((0, 5))
        result = make_cutouts(params, 0)
        assert result == []

    def test_one_cutout_no_symmetry(self):
        # layout: (cx, cy, r_base, c1, s1)
        params = np.array([[50.0, 0.0, 15.0, 0.0, 0.0]])
        result = make_cutouts(params, 1, symmetry=1)
        assert len(result) == 1

    def test_symmetry_doubles_count(self):
        params = np.array([[50.0, 0.0, 15.0, 0.0, 0.0]])
        result = make_cutouts(params, 1, symmetry=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# build_weapon_polygon
# ---------------------------------------------------------------------------

class TestBuildWeaponPolygon:
    def test_disk_returns_polygon(self, disk_cfg):
        N = disk_cfg.optimization.num_fourier_terms
        C = disk_cfg.optimization.num_cutout_pairs
        S = 3 + 2 * disk_cfg.optimization.num_cutout_fourier_terms
        x = np.zeros(1 + 2*N + C*S)
        x[0] = 80.0  # r_base
        outer, params, cutouts = build_weapon_polygon(x, disk_cfg)
        assert isinstance(outer, Polygon)
        assert outer.area > 0

    def test_bar_returns_polygon(self, bar_cfg):
        N = bar_cfg.optimization.num_fourier_terms
        C = bar_cfg.optimization.num_cutout_pairs
        S = 3 + 2 * bar_cfg.optimization.num_cutout_fourier_terms
        x = np.zeros(2 + N + C*S)
        x[0] = 300.0  # length
        x[1] = 50.0   # width
        outer, params, cutouts = build_weapon_polygon(x, bar_cfg)
        assert isinstance(outer, Polygon)

    def test_unknown_style_raises(self, disk_cfg):
        from weapon_designer.config import WeaponConfig
        cfg = WeaponConfig(weapon_style="unknown")
        with pytest.raises(ValueError):
            build_weapon_polygon(np.zeros(20), cfg)
