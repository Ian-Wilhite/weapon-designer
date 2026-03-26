#!/usr/bin/env python3
"""apply_mfg — Apply manufacturability conditioning to a DXF weapon file.

Reads a DXF file, extracts the weapon geometry as a Shapely polygon,
conditions all interior cutout boundaries to meet a minimum concave-corner
radius (R_min), prints a ConditioningDelta summary, and exports the
conditioned geometry to a new DXF file.

Usage
─────
    python3 apply_mfg.py weapon.dxf
    python3 apply_mfg.py weapon.dxf --method vertex --radius 3.0
    python3 apply_mfg.py weapon.dxf --radius 2.5 --output weapon_r25.dxf

Arguments
─────────
  dxf_file          Path to the input DXF file.
  --method          Conditioning method: 'minkowski' (default) or 'vertex'.
  --radius          Minimum concave radius in mm (default: 2.0).
  --output PATH     Output DXF path.  Defaults to <input_stem>_mfg.dxf in
                    the same directory as the input file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# DXF reading helpers
# ---------------------------------------------------------------------------

def _load_polygon_from_dxf(dxf_path: Path):
    """Read a DXF file and return the assembled Shapely polygon.

    Strategy
    ────────
    1. Open the DXF with ezdxf.
    2. Collect all LWPOLYLINE entities from the modelspace.
    3. Convert each closed polyline to a Shapely LinearRing.
    4. Use Shapely's polygonize() or area heuristics to identify the outer
       profile and interior holes.
    5. Return a Shapely Polygon (exterior + interiors).

    If the DXF contains a WEAPON layer and a WEAPON_HOLES layer (matching the
    format written by exporter.py) those are used preferentially.
    """
    import ezdxf
    from shapely.geometry import Polygon, LinearRing
    from shapely.ops import polygonize, unary_union

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # Gather LWPOLYLINE coords, split by layer
    layer_coords: dict[str, list[list[tuple[float, float]]]] = {}

    for entity in msp.query("LWPOLYLINE"):
        layer = entity.dxf.layer.upper()
        pts = [(p[0], p[1]) for p in entity.get_points()]
        if len(pts) >= 3:
            layer_coords.setdefault(layer, []).append(pts)

    if not layer_coords:
        print(
            "WARNING: No LWPOLYLINE entities found in DXF.  "
            "The file may use a different entity type (SPLINE, LINE, etc.).",
            file=sys.stderr,
        )
        return None

    # Prefer named layers matching the exporter.py convention.
    outer_candidates: list[Polygon] = []
    hole_candidates: list[Polygon] = []

    for layer, coord_groups in layer_coords.items():
        for pts in coord_groups:
            try:
                ring = LinearRing(pts)
                poly = Polygon(ring)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
            except Exception:
                continue

            if "HOLE" in layer:
                hole_candidates.append(poly)
            else:
                outer_candidates.append(poly)

    if not outer_candidates:
        # Fallback: treat the largest polygon as outer, rest as holes.
        all_polys = outer_candidates + hole_candidates
        if not all_polys:
            return None
        all_polys.sort(key=lambda p: p.area, reverse=True)
        outer_candidates = [all_polys[0]]
        hole_candidates = all_polys[1:]

    # Pick the largest outer polygon.
    outer = max(outer_candidates, key=lambda p: p.area)

    # Build a Polygon with holes by subtracting each hole.
    if hole_candidates:
        holes_union = unary_union(hole_candidates)
        result = outer.difference(holes_union)
    else:
        result = outer

    if not result.is_valid:
        result = result.buffer(0)

    return result


# ---------------------------------------------------------------------------
# DXF writing helpers
# ---------------------------------------------------------------------------

def _export_polygon_to_dxf(poly, out_path: Path) -> None:
    """Write a Shapely Polygon (with holes) to a DXF file."""
    import ezdxf

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    doc.layers.add("WEAPON", color=7)       # white
    doc.layers.add("WEAPON_HOLES", color=1)  # red

    def _add_ring(coords, layer: str):
        msp.add_lwpolyline(coords, close=True, dxfattribs={"layer": layer})

    from shapely.geometry import MultiPolygon

    polys = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
    for p in polys:
        _add_ring(list(p.exterior.coords), "WEAPON")
        for interior in p.interiors:
            _add_ring(list(interior.coords), "WEAPON_HOLES")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(out_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="apply_mfg",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dxf_file",
        help="Path to the input DXF weapon file.",
    )
    parser.add_argument(
        "--method",
        choices=("minkowski", "vertex"),
        default="minkowski",
        help="Conditioning method (default: minkowski).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        metavar="R_MM",
        help="Minimum concave radius in mm (default: 2.0).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Output DXF path.  Defaults to <input_stem>_mfg.dxf "
            "in the same directory as the input file."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    args = _parse_args(argv)

    dxf_path = Path(args.dxf_file).resolve()
    if not dxf_path.exists():
        print(f"ERROR: File not found: {dxf_path}", file=sys.stderr)
        return 1
    if dxf_path.suffix.lower() != ".dxf":
        print(f"WARNING: Expected a .dxf file, got: {dxf_path.suffix}", file=sys.stderr)

    # Determine output path
    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = dxf_path.with_name(dxf_path.stem + "_mfg.dxf")

    print(f"Input  : {dxf_path}")
    print(f"Method : {args.method}")
    print(f"R_min  : {args.radius} mm")
    print(f"Output : {out_path}")
    print()

    # Load polygon from DXF
    print("Reading DXF ...", end=" ", flush=True)
    weapon = _load_polygon_from_dxf(dxf_path)
    if weapon is None or weapon.is_empty:
        print("FAILED")
        print("ERROR: Could not extract a valid polygon from the DXF file.", file=sys.stderr)
        return 1
    print(f"OK  (area = {weapon.area:.1f} mm²)")

    n_holes = len(list(weapon.interiors))
    print(f"  Interior rings (holes): {n_holes}")

    # Identify the outer profile — outer boundary only (no holes)
    from shapely.geometry import Polygon
    outer_profile = Polygon(weapon.exterior)

    # Apply conditioning
    from weapon_designer.manufacturability import GeometryConditioner
    gc = GeometryConditioner(R_min_mm=args.radius, method=args.method)

    print()
    print("Conditioning ...", end=" ", flush=True)
    conditioned, delta = gc.condition_weapon(weapon, outer_profile)
    print("OK")

    # Print delta summary
    print()
    print("=== ConditioningDelta ===")
    print(f"  method               : {delta.method}")
    print(f"  n_corners_modified   : {delta.n_corners_modified}")
    print(f"  area_change_mm2      : {delta.area_change_mm2:+.4f}")
    print(f"  max_curvature_before : {delta.max_curvature_before:.6f}  (1/mm)")
    print(f"  max_curvature_after  : {delta.max_curvature_after:.6f}  (1/mm)")
    if delta.max_curvature_before > 0.0:
        r_min_before = 1.0 / delta.max_curvature_before
        print(f"  min_radius_before    : {r_min_before:.3f} mm")
    if delta.max_curvature_after > 0.0:
        r_min_after = 1.0 / delta.max_curvature_after
        print(f"  min_radius_after     : {r_min_after:.3f} mm")
    print()

    # Area comparison
    print(f"  weapon area before   : {weapon.area:.2f} mm²")
    print(f"  weapon area after    : {conditioned.area:.2f} mm²")
    area_diff = conditioned.area - weapon.area
    print(f"  weapon area delta    : {area_diff:+.4f} mm²")
    print()

    # Export
    print(f"Exporting to {out_path} ...", end=" ", flush=True)
    try:
        _export_polygon_to_dxf(conditioned, out_path)
        print("OK")
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
