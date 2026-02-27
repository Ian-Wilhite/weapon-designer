"""Command-line entry point for weapon-designer."""

from __future__ import annotations

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

from .config import load_config
from .optimizer import optimize
from .exporter import export_dxf, export_stats


def main():
    parser = argparse.ArgumentParser(
        description="Combat robot spinning weapon profile optimizer",
    )
    parser.add_argument(
        "config",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show matplotlib preview of the optimised weapon",
    )
    parser.add_argument(
        "--dxf",
        type=str,
        default=None,
        help="Override DXF output path",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Override stats JSON output path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Base directory to place run subfolders (default: config output.base_dir)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run subfolder (default: timestamp)",
    )
    parser.add_argument(
        "--resume-stats",
        type=str,
        default=None,
        help="Path to a previous stats JSON to seed optimisation and reuse its folder",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.preview:
        cfg.output.preview = True
    if args.dxf:
        cfg.output.dxf_path = args.dxf
    if args.stats:
        cfg.output.stats_path = args.stats
    if args.output_root:
        cfg.output.base_dir = args.output_root
    if args.run_name:
        cfg.output.run_subdir = args.run_name
    if args.resume_stats:
        cfg.output.resume_stats = args.resume_stats

    verbose = not args.quiet

    # Determine run directory
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = Path(cfg.output.base_dir)
    resume_params = None

    # If resuming, prefer the folder of the resume stats unless overridden
    if cfg.output.resume_stats:
        resume_path = Path(cfg.output.resume_stats)
        with open(resume_path) as f:
            prev_stats = json.load(f)
        resume_params = prev_stats.get("best_params")
        # If no run_subdir provided, reuse previous run directory
        if cfg.output.run_subdir is None:
            cfg.output.run_subdir = resume_path.parent.name
        # Use previous base dir unless explicitly overridden
        if args.output_root is None:
            run_root = resume_path.parent.parent

    run_dir = run_root / (cfg.output.run_subdir or now)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.output.run_dir = str(run_dir)

    # Resolve output paths inside run dir unless absolute
    if not Path(cfg.output.dxf_path).is_absolute():
        cfg.output.dxf_path = str(run_dir / Path(cfg.output.dxf_path).name)
    if not Path(cfg.output.stats_path).is_absolute():
        cfg.output.stats_path = str(run_dir / Path(cfg.output.stats_path).name)

    # Run optimisation
    result = optimize(cfg, verbose=verbose, resume_params=resume_params)

    # Export outputs
    export_dxf(result["weapon_polygon"], cfg.output.dxf_path, cfg)
    export_stats(
        result["metrics"],
        result["score"],
        result["penalty"],
        cfg,
        cfg.output.stats_path,
        best_params=result["result_phase1"].x,
    )

    # Preview
    if cfg.output.preview:
        from .visualization import preview_weapon

        preview_weapon(
            result["weapon_polygon"],
            cfg,
            metrics=result["metrics"],
            score=result["score"],
        )

    if verbose:
        print(f"\nDone. DXF: {cfg.output.dxf_path} | Stats: {cfg.output.stats_path}")


if __name__ == "__main__":
    main()
