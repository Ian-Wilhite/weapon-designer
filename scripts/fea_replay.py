#!/usr/bin/env python3
"""FEA Replay — assemble GIFs from saved sidecar files without re-running optimisation.

Each optimizer step that saves a PNG frame also writes two sidecar files:
    frame_NNNN.npz       — mesh arrays (nodes, elements, vm_stresses) + polygon coords
    frame_NNNN_meta.json — step metadata (phase, score, metrics, cfg_snapshot)

This tool reads those sidecars, optionally re-renders each frame (allowing
different colourscale or DPI), and stitches the result into an animated GIF.

Usage examples
──────────────
# Assemble GIF from both phases, re-rendering at default settings
    python3 fea_replay.py runs/my_run/frames_p1 runs/my_run/frames_p2

# Use existing PNGs without re-rendering
    python3 fea_replay.py runs/my_run/frames_p1 --skip-render

# Cap stress colourscale at 80 % of yield and increase DPI
    python3 fea_replay.py runs/my_run/frames_p1 --vmax 0.8 --dpi 120

# Only frames 5 through 20
    python3 fea_replay.py runs/my_run/frames_p1 --frame-range 5:20

# Combine both phases into one GIF with explicit output path
    python3 fea_replay.py --phase1 runs/my_run/frames_p1 \\
                          --phase2 runs/my_run/frames_p2 \\
                          --output runs/my_run/replay_combined.gif
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Sidecar discovery helpers
# ---------------------------------------------------------------------------

def _discover_sidecars(frames_dir: Path) -> list[dict]:
    """Return sorted list of {npz, meta, png} dicts for all complete frames."""
    npz_files = sorted(frames_dir.glob("frame_*.npz"))
    records: list[dict] = []
    for npz in npz_files:
        stem = npz.stem  # e.g. "frame_0003"
        meta_path = frames_dir / (stem + "_meta.json")
        png_path  = frames_dir / (stem + ".png")
        records.append({
            "stem":  stem,
            "npz":   npz,
            "meta":  meta_path if meta_path.exists() else None,
            "png":   png_path  if png_path.exists()  else None,
        })
    return records


def _load_sidecar(record: dict) -> dict | None:
    """Load arrays and metadata from a sidecar pair.

    Returns a dict with keys: nodes, elements, vm_stresses, polygon_xy,
    holes_xy, meta.  Returns None on any load error.
    """
    try:
        data = np.load(record["npz"], allow_pickle=False)
    except Exception as e:
        print(f"[replay] Cannot load {record['npz']}: {e}", file=sys.stderr)
        return None

    meta = {}
    if record["meta"] is not None:
        try:
            meta = json.loads(record["meta"].read_text())
        except Exception as e:
            print(f"[replay] Cannot parse {record['meta']}: {e}", file=sys.stderr)

    return {
        "nodes":       data["nodes"],
        "elements":    data["elements"],
        "vm_stresses": data["vm_stresses"],
        "polygon_xy":  data["polygon_xy"],
        "holes_xy":    data["holes_xy"],
        "meta":        meta,
    }


# ---------------------------------------------------------------------------
# Polygon reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_polygon(polygon_xy: np.ndarray, holes_xy: np.ndarray):
    """Reconstruct a Shapely Polygon from saved coordinate arrays.

    holes_xy uses NaN-row sentinels to separate hole rings.
    """
    from shapely.geometry import Polygon

    exterior = list(map(tuple, polygon_xy.tolist()))

    holes: list[list[tuple]] = []
    if holes_xy.size > 0 and not np.all(np.isnan(holes_xy)):
        current_hole: list[tuple] = []
        for row in holes_xy:
            if np.any(np.isnan(row)):
                if current_hole:
                    holes.append(current_hole)
                    current_hole = []
            else:
                current_hole.append((float(row[0]), float(row[1])))
        if current_hole:
            holes.append(current_hole)

    return Polygon(exterior, holes)


# ---------------------------------------------------------------------------
# Frame-range parsing
# ---------------------------------------------------------------------------

def _parse_frame_range(frame_range: str | None, n_frames: int) -> slice:
    """Parse a "N:M" frame-range string into a slice."""
    if frame_range is None:
        return slice(None)
    parts = frame_range.split(":")
    lo = int(parts[0]) if parts[0] else 0
    hi = int(parts[1]) if len(parts) > 1 and parts[1] else n_frames
    return slice(lo, hi)


# ---------------------------------------------------------------------------
# Re-render a single frame
# ---------------------------------------------------------------------------

def _rerender_frame(
    loaded: dict,
    out_png: Path,
    vmax: float,
    dpi: int,
) -> Path | None:
    """Re-render a frame PNG from sidecar data with optional vmax override."""
    try:
        from weapon_designer.fea_viz import render_fea_frame
        from weapon_designer.config import WeaponConfig, Material, OptimizationParams
    except ImportError as e:
        print(f"[replay] Cannot import weapon_designer: {e}", file=sys.stderr)
        return None

    poly = _reconstruct_polygon(loaded["polygon_xy"], loaded["holes_xy"])

    cfg_snap = loaded["meta"].get("cfg_snapshot", {})
    cfg = WeaponConfig()
    cfg.material.yield_strength_mpa = cfg_snap.get(
        "yield_strength_mpa", cfg.material.yield_strength_mpa
    )
    cfg.material.density_kg_m3      = cfg_snap.get(
        "density_kg_m3", cfg.material.density_kg_m3
    )
    cfg.sheet_thickness_mm = cfg_snap.get("sheet_thickness_mm", cfg.sheet_thickness_mm)
    cfg.rpm                = cfg_snap.get("rpm", cfg.rpm)
    cfg.mounting.bore_diameter_mm = cfg_snap.get(
        "bore_diameter_mm", cfg.mounting.bore_diameter_mm
    )

    # Apply vmax override: scale yield_strength so ratio colourscale clamps at vmax
    if vmax != 1.0:
        cfg.material.yield_strength_mpa /= max(vmax, 1e-6)

    meta    = loaded["meta"]
    metrics = meta.get("metrics")
    phase   = meta.get("phase", "?")
    step    = meta.get("step", 0)

    fea_result = {
        "nodes":          loaded["nodes"],
        "elements":       loaded["elements"],
        "vm_stresses":    loaded["vm_stresses"],
        "peak_stress_mpa": float(loaded["vm_stresses"].max()) if loaded["vm_stresses"].size else 0.0,
        "safety_factor":  (cfg_snap.get("yield_strength_mpa", cfg.material.yield_strength_mpa)
                           / max(float(loaded["vm_stresses"].max()) if loaded["vm_stresses"].size else 1.0, 1e-6)),
    }

    return render_fea_frame(
        poly, fea_result, cfg,
        step_label=f"{phase}-{step:03d}",
        metrics=metrics,
        save_path=out_png,
        dpi=dpi,
    )


# ---------------------------------------------------------------------------
# Main replay logic
# ---------------------------------------------------------------------------

def replay(
    dirs: list[Path],
    output: Path,
    fps: int = 5,
    hold_last: int = 4,
    vmax: float = 1.0,
    dpi: int = 90,
    skip_render: bool = False,
    frame_range: str | None = None,
) -> Path | None:
    """Core replay logic — collect frames from one or more directories and make a GIF."""
    try:
        from weapon_designer.fea_viz import export_gif
    except ImportError as e:
        print(f"[replay] Cannot import weapon_designer.fea_viz: {e}", file=sys.stderr)
        return None

    all_records: list[dict] = []
    for d in dirs:
        recs = _discover_sidecars(d)
        if not recs:
            print(f"[replay] No frame_*.npz files found in {d}", file=sys.stderr)
        all_records.extend(recs)

    if not all_records:
        print("[replay] No frames found — nothing to do.", file=sys.stderr)
        return None

    # Apply frame-range filter
    sl = _parse_frame_range(frame_range, len(all_records))
    all_records = all_records[sl]
    print(f"[replay] {len(all_records)} frame(s) to process.")

    if skip_render:
        # Collect existing PNGs
        png_paths = [r["png"] for r in all_records if r["png"] is not None]
        if not png_paths:
            print("[replay] --skip-render: no existing PNGs found.", file=sys.stderr)
            return None
        print(f"[replay] Assembling GIF from {len(png_paths)} existing PNG(s)...")
        return export_gif(png_paths, output, fps=fps, hold_last=hold_last)

    # Re-render frames into a temp directory
    with tempfile.TemporaryDirectory(prefix="fea_replay_") as tmpdir:
        tmp = Path(tmpdir)
        png_paths = []

        for idx, record in enumerate(all_records):
            loaded = _load_sidecar(record)
            if loaded is None:
                print(f"[replay] Skipping {record['stem']} (load error).")
                continue

            out_png = tmp / f"frame_{idx:04d}.png"
            result  = _rerender_frame(loaded, out_png, vmax=vmax, dpi=dpi)
            if result is None:
                print(f"[replay] Render failed for {record['stem']}.")
            else:
                png_paths.append(result)
                phase = loaded["meta"].get("phase", "?")
                step  = loaded["meta"].get("step", idx)
                print(f"  rendered {record['stem']}  phase={phase} step={step}")

        if not png_paths:
            print("[replay] All frames failed to render.", file=sys.stderr)
            return None

        print(f"[replay] Assembling GIF ({len(png_paths)} frames → {output})...")
        result = export_gif(png_paths, output, fps=fps, hold_last=hold_last)

    if result:
        print(f"[replay] GIF written: {result}")
    else:
        print("[replay] GIF assembly failed.", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble FEA convergence GIFs from saved sidecar files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "frames_dir",
        nargs="*",
        type=Path,
        help="One or more directories containing frame_NNNN.npz sidecars.",
    )
    parser.add_argument(
        "--phase1",
        type=Path,
        metavar="DIR",
        help="Phase-1 frames directory (alternative to positional args).",
    )
    parser.add_argument(
        "--phase2",
        type=Path,
        metavar="DIR",
        help="Phase-2 frames directory; combined with --phase1 into one GIF.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        metavar="PATH",
        default=None,
        help="Output GIF path (default: <first_dir>/../replay.gif).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        metavar="INT",
        help="Frames per second in the output GIF (default: 5).",
    )
    parser.add_argument(
        "--hold-last",
        type=int,
        default=4,
        metavar="INT",
        help="Extra copies of the final frame to hold (default: 4).",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Stress colourscale cap as fraction of yield (default: 1.0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=90,
        metavar="INT",
        help="Output DPI for re-rendered frames (default: 90).",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        default=False,
        help="Assemble existing PNGs without re-rendering from sidecar data.",
    )
    parser.add_argument(
        "--frame-range",
        metavar="N:M",
        default=None,
        help='Only process frames N through M (zero-based, Python slice syntax).',
    )

    args = parser.parse_args()

    # Collect directories
    dirs: list[Path] = list(args.frames_dir or [])
    if args.phase1:
        dirs.insert(0, args.phase1)
    if args.phase2:
        dirs.append(args.phase2)

    if not dirs:
        parser.error(
            "Provide at least one frames directory as a positional argument, "
            "or use --phase1 / --phase2."
        )

    # Resolve output path
    if args.output is None:
        output = dirs[0].parent / "replay.gif"
    else:
        output = args.output

    replay(
        dirs=dirs,
        output=output,
        fps=args.fps,
        hold_last=args.hold_last,
        vmax=args.vmax,
        dpi=args.dpi,
        skip_render=args.skip_render,
        frame_range=args.frame_range,
    )


if __name__ == "__main__":
    main()
