"""DXF export via ezdxf and stats JSON output.

Lasercut-ready DXF export:
- GeometryConditioner applied before writing (rounds concave corners to mfg_concave_radius_mm)
- WEAPON layer: outer profile (closed LWPOLYLINE)
- WEAPON_HOLES layer: interior cutouts (closed LWPOLYLINE)
- TEXT layer: material callout, scale indicator, cutting tolerance note
- Validation: all polylines checked for closure; self-intersections flagged
"""

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


def _validate_dxf_geometry(weapon: Polygon | MultiPolygon, cfg: WeaponConfig | None) -> list[str]:
    """Run pre-export geometry checks. Returns list of warning strings (empty = OK)."""
    warnings = []

    polys = list(weapon.geoms) if isinstance(weapon, MultiPolygon) else [weapon]

    for i, poly in enumerate(polys):
        prefix = f"poly[{i}]" if len(polys) > 1 else "weapon"

        if not poly.is_valid:
            warnings.append(f"{prefix}: self-intersecting geometry (Shapely is_valid=False)")

        if not poly.exterior.is_ring:
            warnings.append(f"{prefix}: exterior ring is not closed")

        for j, ring in enumerate(poly.interiors):
            if not ring.is_ring:
                warnings.append(f"{prefix} hole[{j}]: interior ring is not closed")

        if cfg is not None:
            min_feat = getattr(cfg.optimization, "mfg_concave_radius_mm", 2.0)
            # Check via buffer erosion: if eroding by min_feat collapses the polygon,
            # some feature is smaller than the minimum.
            eroded = poly.buffer(-min_feat)
            if eroded is None or eroded.is_empty:
                warnings.append(
                    f"{prefix}: one or more features may be smaller than "
                    f"mfg_concave_radius_mm={min_feat:.1f} mm"
                )

    return warnings


def _apply_conditioner(
    weapon: Polygon | MultiPolygon,
    cfg: WeaponConfig,
) -> Polygon | MultiPolygon:
    """Apply GeometryConditioner to round concave corners before DXF export.

    Uses cfg.optimization.mfg_concave_radius_mm and cfg.optimization.mfg_method.
    Returns the conditioned polygon (outer profile unchanged; holes rounded).
    """
    try:
        from .manufacturability import GeometryConditioner
        r_min = float(getattr(cfg.optimization, "mfg_concave_radius_mm", 2.0))
        method = str(getattr(cfg.optimization, "mfg_method", "minkowski"))
        gc = GeometryConditioner(R_min_mm=r_min, method=method)

        if isinstance(weapon, MultiPolygon):
            conditioned_parts = []
            for part in weapon.geoms:
                outer = Polygon(part.exterior)
                c, _ = gc.condition_weapon(part, outer)
                conditioned_parts.append(c)
            from shapely.ops import unary_union
            return unary_union(conditioned_parts)
        else:
            outer = Polygon(weapon.exterior)
            conditioned, delta = gc.condition_weapon(weapon, outer)
            if delta.n_corners_modified > 0:
                print(
                    f"[exporter] GeometryConditioner: {delta.n_corners_modified} corners rounded "
                    f"(area change {delta.area_change_mm2:+.1f} mm²)"
                )
            return conditioned
    except Exception as e:
        print(f"[exporter] GeometryConditioner skipped: {e}")
        return weapon


def _add_dxf_annotations(msp, weapon: Polygon | MultiPolygon, cfg: WeaponConfig):
    """Add lasercut annotation text on the TEXT layer.

    Annotations:
    - Material name and thickness
    - Scale indicator (1:1)
    - Cutting tolerance note
    - Weapon mass and RPM
    """
    # Bounding box for text placement (just below the weapon)
    if isinstance(weapon, MultiPolygon):
        bounds = weapon.bounds  # (minx, miny, maxx, maxy)
    else:
        bounds = weapon.bounds

    minx, miny, maxx, maxy = bounds
    text_x = minx
    line_h = (maxy - miny) * 0.08   # 8% of weapon height per line
    text_y = miny - line_h          # start just below weapon

    def _add_line(text: str, dy_factor: float = 0.0):
        y = text_y - dy_factor * line_h
        msp.add_text(
            text,
            dxfattribs={
                "layer": "TEXT",
                "height": max(line_h * 0.6, 3.0),
                "insert": (text_x, y),
            },
        )

    r_min_mm = getattr(cfg.optimization, "mfg_concave_radius_mm", 2.0)
    _add_line(f"MATERIAL: {cfg.material.name}  t={cfg.sheet_thickness_mm:.0f}mm", 0.0)
    _add_line(f"SCALE: 1:1  ALL DIMENSIONS IN mm", 1.2)
    _add_line(f"CUT TOL: +/-0.1mm  MIN INSIDE R: {r_min_mm:.1f}mm", 2.4)
    _add_line(f"WEAPON: {cfg.weapon_style.upper()}  {cfg.rpm} RPM  budget={cfg.weight_budget_kg:.2f}kg", 3.6)


def export_dxf(
    weapon: Polygon | MultiPolygon,
    path: str | Path,
    cfg: WeaponConfig | None = None,
    apply_conditioning: bool = True,
    add_annotations: bool = True,
):
    """Export the weapon polygon to a DXF file.

    All units in mm. The DXF is lasercut/waterjet ready.

    Parameters
    ----------
    weapon             : assembled weapon polygon (holes already punched)
    path               : output .dxf path
    cfg                : weapon config (used for annotations and conditioner params)
    apply_conditioning : round concave corners via GeometryConditioner (default True)
    add_annotations    : add material/scale/tolerance text on TEXT layer (default True)
    """
    # ── Geometry conditioning (round sharp concave corners) ──────────────
    if apply_conditioning and cfg is not None:
        weapon = _apply_conditioner(weapon, cfg)

    # ── Pre-export validation ─────────────────────────────────────────────
    warnings = _validate_dxf_geometry(weapon, cfg)
    for w in warnings:
        print(f"[exporter] WARNING: {w}")

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Create layers
    doc.layers.add("WEAPON", color=7)       # white - outer profile
    doc.layers.add("WEAPON_HOLES", color=1)  # red - holes/cutouts
    doc.layers.add("TEXT", color=3)          # green - annotations

    if isinstance(weapon, MultiPolygon):
        for geom in weapon.geoms:
            _add_polygon_to_dxf(msp, geom)
    else:
        _add_polygon_to_dxf(msp, weapon)

    # ── Annotations ──────────────────────────────────────────────────────
    if add_annotations and cfg is not None:
        _add_dxf_annotations(msp, weapon, cfg)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(path))
    print(f"DXF exported to {path}")


def export_weapon_dxf(
    weapon: Polygon | MultiPolygon,
    cfg: WeaponConfig,
    output_dir: str | Path,
    stem: str = "weapon",
) -> Path | None:
    """Export the final weapon polygon as a DXF in output_dir/<stem>.dxf.

    Thin wrapper around export_dxf() for use in any run script that has a
    Shapely weapon polygon.  Returns the path written, or None on failure.

    Parameters
    ----------
    weapon     : final weapon Shapely polygon (mm units, all holes included)
    cfg        : weapon configuration (used for metadata layers)
    output_dir : directory to write into (created if needed)
    stem       : filename stem without extension (default "weapon")
    """
    if weapon is None or (hasattr(weapon, "is_empty") and weapon.is_empty):
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.dxf"
    try:
        export_dxf(weapon, path, cfg)
        return path
    except Exception as e:
        print(f"[exporter] DXF export failed: {e}")
        return None


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
