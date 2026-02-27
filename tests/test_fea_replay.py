"""Tests for FEA sidecar roundtrip — save .npz + meta JSON, reload, reconstruct."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, Polygon


# ---------------------------------------------------------------------------
# Sidecar data helpers (mirrors what optimizer_enhanced.py writes)
# ---------------------------------------------------------------------------

def _make_dummy_fea_data():
    """Create minimal fake FEA arrays for sidecar test."""
    # Simple triangulated annulus approximation
    r_out, r_in = 80.0, 15.0
    n = 20
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(theta), r_out * np.sin(theta)])
    inner_pts = np.column_stack([r_in * np.cos(theta), r_in * np.sin(theta)])
    nodes = np.vstack([outer_pts, inner_pts])

    # Triangles connecting outer and inner ring segments
    elements = []
    for i in range(n):
        j = (i + 1) % n
        elements.append([i, j, n + i])
        elements.append([j, n + j, n + i])
    elements = np.array(elements, dtype=int)

    vm_stresses = np.random.default_rng(42).uniform(50, 300, len(elements))

    polygon = Point(0, 0).buffer(r_out, resolution=64)
    polygon_xy = np.array(polygon.exterior.coords)

    return nodes, elements, vm_stresses, polygon_xy


def _save_sidecar_npz(path: Path, nodes, elements, vm_stresses, polygon_xy):
    """Write the .npz sidecar (matches optimizer_enhanced.py format)."""
    np.savez_compressed(
        path,
        nodes=nodes,
        elements=elements,
        vm_stresses=vm_stresses,
        polygon_xy=polygon_xy,
    )


def _save_sidecar_json(path: Path, meta: dict):
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSidecarRoundtrip:
    def test_npz_roundtrip(self, tmp_path):
        nodes, elements, vm_stresses, polygon_xy = _make_dummy_fea_data()
        npz_path = tmp_path / "frame_0000.npz"
        _save_sidecar_npz(npz_path, nodes, elements, vm_stresses, polygon_xy)

        assert npz_path.exists()
        loaded = np.load(npz_path)

        np.testing.assert_array_equal(loaded["nodes"], nodes)
        np.testing.assert_array_equal(loaded["elements"], elements)
        np.testing.assert_allclose(loaded["vm_stresses"], vm_stresses)
        np.testing.assert_allclose(loaded["polygon_xy"], polygon_xy)

    def test_meta_json_roundtrip(self, tmp_path):
        meta = {
            "step": 5,
            "phase": "P1",
            "score": 0.72,
            "metrics": {"moi_kg_mm2": 1234.5, "bite_mm": 15.0},
            "cfg_snapshot": {"weapon_style": "disk", "rpm": 8000},
        }
        json_path = tmp_path / "frame_0000_meta.json"
        _save_sidecar_json(json_path, meta)

        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)

        assert loaded["step"] == 5
        assert loaded["phase"] == "P1"
        assert loaded["score"] == pytest.approx(0.72)
        assert loaded["metrics"]["moi_kg_mm2"] == pytest.approx(1234.5)

    def test_polygon_reconstruct_from_npz(self, tmp_path):
        """Polygon coordinates saved in .npz should reconstruct the original polygon."""
        nodes, elements, vm_stresses, polygon_xy = _make_dummy_fea_data()
        npz_path = tmp_path / "frame_0000.npz"
        _save_sidecar_npz(npz_path, nodes, elements, vm_stresses, polygon_xy)

        loaded = np.load(npz_path)
        coords = loaded["polygon_xy"].tolist()
        poly = Polygon(coords)

        assert poly.is_valid or poly.buffer(0).area > 0
        original = Point(0, 0).buffer(80.0)
        # Areas should be approximately equal
        assert poly.area == pytest.approx(original.area, rel=0.02)

    def test_multiple_frames_saved(self, tmp_path):
        """Save 5 frames; verify all .npz and .json files exist."""
        nodes, elements, vm_stresses, polygon_xy = _make_dummy_fea_data()

        for i in range(5):
            npz = tmp_path / f"frame_{i:04d}.npz"
            js  = tmp_path / f"frame_{i:04d}_meta.json"
            _save_sidecar_npz(npz, nodes, elements, vm_stresses, polygon_xy)
            _save_sidecar_json(js, {"step": i, "score": float(i) * 0.1})

        npz_files = sorted(tmp_path.glob("frame_*.npz"))
        json_files = sorted(tmp_path.glob("frame_*_meta.json"))
        assert len(npz_files) == 5
        assert len(json_files) == 5

    def test_stresses_shape_matches_elements(self, tmp_path):
        nodes, elements, vm_stresses, polygon_xy = _make_dummy_fea_data()
        assert vm_stresses.shape == (len(elements),)

        npz_path = tmp_path / "frame_0000.npz"
        _save_sidecar_npz(npz_path, nodes, elements, vm_stresses, polygon_xy)
        loaded = np.load(npz_path)
        assert loaded["vm_stresses"].shape[0] == loaded["elements"].shape[0]
