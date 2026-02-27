"""DXF export via ezdxf and stats JSON output."""

from __future__ import annotations

import json
from pathlib import Path

import ezdxf
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from .config import WeaponConfig


def _add_polygon_to_dxf(msp, poly: Polygon, layer: str = "WEAPON"):
    """Add a Shapely polygon (with holes) to a DXF modelspace as LWPOLYLINE entities."""
    # Exterior ring
    coords = list(poly.exterior.coords)
    msp.add_lwpolyline(coords, close=True, dxfattribs={"layer": layer})

    # Interior rings (holes)
    for interior in poly.interiors:
        coords = list(interior.coords)
        msp.add_lwpolyline(coords, close=True, dxfattribs={"layer": f"{layer}_HOLES"})


def export_dxf(
    weapon: Polygon | MultiPolygon,
    path: str | Path,
    cfg: WeaponConfig | None = None,
):
    """Export the weapon polygon to a DXF file.

    All units in mm. The DXF is ready for waterjet/laser cutting.
    """
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Create layers
    doc.layers.add("WEAPON", color=7)       # white - outer profile
    doc.layers.add("WEAPON_HOLES", color=1)  # red - holes/cutouts

    if isinstance(weapon, MultiPolygon):
        for geom in weapon.geoms:
            _add_polygon_to_dxf(msp, geom)
    else:
        _add_polygon_to_dxf(msp, weapon)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(path))
    print(f"DXF exported to {path}")


def export_snapshot(
    weapon: Polygon | MultiPolygon,
    cfg: WeaponConfig,
    output_dir: str | Path,
    label: str,
):
    """Export a DXF snapshot for intermediate visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"snapshot_{label}.dxf"
    export_dxf(weapon, path, cfg)


def export_stats(
    metrics: dict,
    score: float,
    penalty: float,
    cfg: WeaponConfig,
    path: str | Path,
    best_params=None,
):
    """Export optimisation statistics to a JSON file."""
    stats = {
        "weapon_style": cfg.weapon_style,
        "material": cfg.material.name,
        "sheet_thickness_mm": cfg.sheet_thickness_mm,
        "rpm": cfg.rpm,
        "weight_budget_kg": cfg.weight_budget_kg,
        "optimization_score": round(score, 6),
        "constraint_penalty": round(penalty, 6),
        "metrics": {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in metrics.items()
        },
        "output_dxf": cfg.output.dxf_path,
        "run_dir": cfg.output.run_dir,
    }

    if best_params is not None:
        stats["best_params"] = [float(x) for x in best_params]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats exported to {path}")
