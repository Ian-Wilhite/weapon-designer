#!/usr/bin/env python3
"""
Full exploratory parameter sweep — generates a large, diverse sample of
weapon shapes across all weapon types, weight classes, and objective
emphasis profiles.

Designed to saturate a 20-core machine for ~8 hours.

Usage:
    python sweep_explore.py [--output-dir DIR] [--dry-run] [--resume]

Each job runs explore_diverse() with high iteration counts and large
archives.  Jobs run sequentially (each already parallelizes internally
across all cores).  Results land in per-job subdirectories with merged
summary at the end.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weapon_designer.config import (
    Envelope,
    Material,
    Mounting,
    OptimizationParams,
    OptimizationWeights,
    OutputParams,
    WeaponConfig,
)

from explore_diverse import explore_diverse, export_results

log = logging.getLogger("sweep")

# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

AR500 = Material(name="AR500", density_kg_m3=7850, yield_strength_mpa=1400, hardness_hrc=50)
S7 = Material(name="S7_Tool_Steel", density_kg_m3=7750, yield_strength_mpa=1600, hardness_hrc=56)

# ---------------------------------------------------------------------------
# Weight emphasis presets (sum to 1.0)
# ---------------------------------------------------------------------------

EMPHASIS = {
    "balanced": OptimizationWeights(
        moment_of_inertia=0.25, bite=0.20, structural_integrity=0.20,
        mass_utilization=0.10, balance=0.10, impact_zone=0.15,
    ),
    "moi_heavy": OptimizationWeights(
        moment_of_inertia=0.45, bite=0.10, structural_integrity=0.15,
        mass_utilization=0.10, balance=0.05, impact_zone=0.15,
    ),
    "bite_heavy": OptimizationWeights(
        moment_of_inertia=0.15, bite=0.35, structural_integrity=0.15,
        mass_utilization=0.10, balance=0.10, impact_zone=0.15,
    ),
    "structural": OptimizationWeights(
        moment_of_inertia=0.20, bite=0.10, structural_integrity=0.40,
        mass_utilization=0.10, balance=0.10, impact_zone=0.10,
    ),
}


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

def _make_cfg(
    style: str,
    material: Material,
    thickness_mm: float,
    weight_kg: float,
    rpm: int,
    mounting: Mounting,
    envelope: Envelope,
    weights: OptimizationWeights,
    fourier_terms: int = 4,
    cutout_pairs: int = 2,
) -> WeaponConfig:
    return WeaponConfig(
        material=material,
        weapon_style=style,
        sheet_thickness_mm=thickness_mm,
        weight_budget_kg=weight_kg,
        rpm=rpm,
        mounting=mounting,
        envelope=envelope,
        optimization=OptimizationParams(
            weights=weights,
            num_fourier_terms=fourier_terms,
            num_cutout_pairs=cutout_pairs,
            max_iterations=200,
            population_size=60,
        ),
    )


def build_jobs() -> list[dict]:
    """Build the full sweep job list.

    Strategy: for each weapon style, sweep across weight classes and
    objective emphasis profiles.  Each combo gets a dedicated exploration
    run with a unique seed to avoid duplicate coverage.
    """
    jobs = []
    seed_counter = 100

    # --- DISK variants ---
    disk_mount_sm = Mounting(bore_diameter_mm=12.0, bolt_circle_diameter_mm=25, num_bolts=3, bolt_hole_diameter_mm=4.0)
    disk_mount_md = Mounting(bore_diameter_mm=19.05, bolt_circle_diameter_mm=40, num_bolts=4, bolt_hole_diameter_mm=6.0)
    disk_mount_lg = Mounting(bore_diameter_mm=25.4, bolt_circle_diameter_mm=60, num_bolts=6, bolt_hole_diameter_mm=8.0)

    disk_specs = [
        # (label_suffix, material, thickness, weight, rpm, mounting, envelope, fourier, cutouts)
        ("feather",   S7,   6,  1.5, 12000, disk_mount_sm, Envelope(max_radius_mm=80),  4, 2),
        ("light",     AR500, 8,  2.0, 15000, disk_mount_sm, Envelope(max_radius_mm=60),  3, 1),
        ("mid",       AR500, 10, 4.0,  9000, disk_mount_md, Envelope(max_radius_mm=130), 4, 2),
        ("heavy",     AR500, 12, 7.0,  8000, disk_mount_lg, Envelope(max_radius_mm=180), 5, 3),
        ("max_energy",AR500, 15, 8.0,  7000, disk_mount_lg, Envelope(max_radius_mm=200), 5, 3),
    ]

    for spec_label, mat, thick, weight, rpm, mount, env, ft, cp in disk_specs:
        for emph_label, weights in EMPHASIS.items():
            seed_counter += 1
            jobs.append({
                "name": f"disk_{spec_label}_{emph_label}",
                "cfg": _make_cfg("disk", mat, thick, weight, rpm, mount, env, weights, ft, cp),
                "seed": seed_counter,
            })

    # --- BAR variants ---
    bar_mount_sm = Mounting(bore_diameter_mm=15.0, bolt_circle_diameter_mm=30, num_bolts=4, bolt_hole_diameter_mm=5.0)
    bar_mount_lg = Mounting(bore_diameter_mm=19.05, bolt_circle_diameter_mm=40, num_bolts=4, bolt_hole_diameter_mm=6.0)

    bar_specs = [
        ("compact",   AR500, 10, 3.0,  9000, bar_mount_sm, Envelope(max_radius_mm=150, max_length_mm=280, max_width_mm=55), 3, 2),
        ("heavy",     S7,   12, 5.0, 10000, bar_mount_lg, Envelope(max_radius_mm=200, max_length_mm=400, max_width_mm=70), 4, 2),
        ("undercutter",AR500, 8, 4.0,  8000, bar_mount_lg, Envelope(max_radius_mm=180, max_length_mm=350, max_width_mm=90), 4, 2),
    ]

    for spec_label, mat, thick, weight, rpm, mount, env, ft, cp in bar_specs:
        for emph_label, weights in EMPHASIS.items():
            seed_counter += 1
            jobs.append({
                "name": f"bar_{spec_label}_{emph_label}",
                "cfg": _make_cfg("bar", mat, thick, weight, rpm, mount, env, weights, ft, cp),
                "seed": seed_counter,
            })

    # --- EGGBEATER variants ---
    egg_mount_sm = Mounting(bore_diameter_mm=15.0, bolt_circle_diameter_mm=30, num_bolts=4, bolt_hole_diameter_mm=4.0)
    egg_mount_md = Mounting(bore_diameter_mm=20.0, bolt_circle_diameter_mm=38, num_bolts=3, bolt_hole_diameter_mm=5.5)
    egg_mount_lg = Mounting(bore_diameter_mm=22.0, bolt_circle_diameter_mm=45, num_bolts=4, bolt_hole_diameter_mm=6.0)

    egg_specs = [
        ("2blade_light", S7,   10, 2.0, 10000, egg_mount_sm, Envelope(max_radius_mm=120), 4, 2),
        ("2blade_heavy", S7,   10, 4.0, 10000, egg_mount_lg, Envelope(max_radius_mm=140), 4, 2),
        ("3blade",       AR500, 8, 3.5, 12000, egg_mount_md, Envelope(max_radius_mm=130), 4, 2),
        ("4blade",       AR500, 6, 2.5, 14000, egg_mount_sm, Envelope(max_radius_mm=100), 4, 2),
    ]

    for spec_label, mat, thick, weight, rpm, mount, env, ft, cp in egg_specs:
        for emph_label, weights in EMPHASIS.items():
            seed_counter += 1
            jobs.append({
                "name": f"egg_{spec_label}_{emph_label}",
                "cfg": _make_cfg("eggbeater", mat, thick, weight, rpm, mount, env, weights, ft, cp),
                "seed": seed_counter,
            })

    return jobs


# ---------------------------------------------------------------------------
# Exploration parameters — calibrated for ~8h total on 20 cores
# ---------------------------------------------------------------------------

# Total jobs: 5 disk × 4 emph + 3 bar × 4 + 4 egg × 4 = 20 + 12 + 16 = 48
# Target ~8h = 480 min → ~10 min per job
# Each job: 1000 iterations × 96 candidates/batch ≈ 96k evaluations
# With 19 workers, ~5k evals/min → ~20 min/job ... too high
# Reduce to 600 iter × 64 batch = 38.4k evals → ~8 min/job → 48 × 8 = 384 min ≈ 6.4h
# Add headroom for archetype seeding + export overhead → ~7-8h

SWEEP_MAX_ITER = 600
SWEEP_BATCH_SIZE = 64
SWEEP_N_TARGET = 60      # designs to keep per job
SWEEP_MIN_SCORE = 0.25   # relaxed threshold for diversity
SWEEP_N_BINS = 5         # finer grid = more cells to fill


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_job(
    job: dict,
    output_dir: Path,
    log_file,
) -> dict:
    """Run a single exploration job and return summary stats."""
    name = job["name"]
    cfg = job["cfg"]
    seed = job["seed"]

    job_dir = output_dir / name

    def _log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{name}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    _log(f"Starting: style={cfg.weapon_style}, {cfg.weight_budget_kg}kg, "
         f"{cfg.rpm}rpm, seed={seed}")

    t0 = time.monotonic()

    designs = explore_diverse(
        cfg,
        n_target=SWEEP_N_TARGET,
        max_iter=SWEEP_MAX_ITER,
        batch_size=SWEEP_BATCH_SIZE,
        min_score=SWEEP_MIN_SCORE,
        n_bins=SWEEP_N_BINS,
        seed=seed,
        n_workers=0,  # auto
        quiet=True,    # suppress per-iteration logging (we log at job level)
    )

    elapsed = time.monotonic() - t0
    _log(f"Exploration done: {len(designs)} designs in {elapsed:.0f}s")

    # Export
    export_results(designs, cfg, job_dir)
    _log(f"Exported to {job_dir}")

    # Summary stats
    if designs:
        scores = [m.score for m in designs]
        cells = set(m.cell for m in designs)
        methods = {}
        for m in designs:
            methods[m.generation_method] = methods.get(m.generation_method, 0) + 1

        return {
            "name": name,
            "status": "success",
            "weapon_style": cfg.weapon_style,
            "weight_kg": cfg.weight_budget_kg,
            "rpm": cfg.rpm,
            "seed": seed,
            "n_designs": len(designs),
            "unique_cells": len(cells),
            "score_min": round(min(scores), 4),
            "score_max": round(max(scores), 4),
            "score_mean": round(float(np.mean(scores)), 4),
            "generation_methods": methods,
            "elapsed_s": round(elapsed, 1),
        }
    else:
        return {
            "name": name,
            "status": "empty",
            "weapon_style": cfg.weapon_style,
            "n_designs": 0,
            "elapsed_s": round(elapsed, 1),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Full exploratory sweep across all weapon types (~8h)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="sweep_results",
        help="Root output directory (default: sweep_results)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print job list and estimated time without running",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip jobs whose output directories already exist",
    )
    parser.add_argument(
        "--jobs", type=str, default=None,
        help="Comma-separated job name filter (substring match)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    jobs = build_jobs()

    # Filter if requested
    if args.jobs:
        filters = [f.strip() for f in args.jobs.split(",")]
        jobs = [j for j in jobs if any(f in j["name"] for f in filters)]

    if not jobs:
        print("No jobs matched filter.")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Resume: skip completed jobs
    if args.resume:
        remaining = []
        for j in jobs:
            job_dir = output_dir / j["name"]
            summary_file = job_dir / "summary.json"
            if summary_file.exists():
                print(f"  [skip] {j['name']} (already exists)")
            else:
                remaining.append(j)
        jobs = remaining
        if not jobs:
            print("All jobs already completed.")
            sys.exit(0)

    # --- Dry run ---
    if args.dry_run:
        n_disk = sum(1 for j in jobs if "disk" in j["name"])
        n_bar = sum(1 for j in jobs if "bar" in j["name"])
        n_egg = sum(1 for j in jobs if "egg" in j["name"])

        print(f"\nSweep plan: {len(jobs)} jobs")
        print(f"  Disk:      {n_disk} jobs")
        print(f"  Bar:       {n_bar} jobs")
        print(f"  Eggbeater: {n_egg} jobs")
        print(f"\nPer-job parameters:")
        print(f"  max_iter:   {SWEEP_MAX_ITER}")
        print(f"  batch_size: {SWEEP_BATCH_SIZE}")
        print(f"  n_target:   {SWEEP_N_TARGET} designs")
        print(f"  n_bins:     {SWEEP_N_BINS} (per behavior dim)")
        print(f"  min_score:  {SWEEP_MIN_SCORE}")
        print(f"\nPer-job evals: ~{SWEEP_MAX_ITER * SWEEP_BATCH_SIZE:,}")
        print(f"Total evals:   ~{len(jobs) * SWEEP_MAX_ITER * SWEEP_BATCH_SIZE:,}")
        print(f"Total designs: up to {len(jobs) * SWEEP_N_TARGET:,}")
        print(f"\nEstimated time: ~7-9 hours on {os.cpu_count()} cores")
        print(f"Output dir:     {output_dir}")
        print(f"\nJobs:")
        for j in jobs:
            c = j["cfg"]
            print(f"  {j['name']:<40} {c.weapon_style:<10} "
                  f"{c.weight_budget_kg:>4.1f}kg  {c.rpm:>5}rpm  seed={j['seed']}")
        return

    # --- Run ---
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = output_dir / f"sweep_{timestamp}.log"
    log_file = open(log_path, "w")

    def _log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    # Graceful shutdown on SIGINT/SIGTERM
    shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            _log("Force quit — exiting immediately")
            sys.exit(1)
        shutdown_requested = True
        _log("Shutdown requested — finishing current job then saving summary...")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    _log(f"{'=' * 70}")
    _log(f"DIVERSITY SWEEP — {len(jobs)} jobs")
    _log(f"{'=' * 70}")
    _log(f"Output:     {output_dir}")
    _log(f"Log:        {log_path}")
    _log(f"Per-job:    {SWEEP_MAX_ITER} iter × {SWEEP_BATCH_SIZE} batch → "
         f"~{SWEEP_MAX_ITER * SWEEP_BATCH_SIZE:,} evals, {SWEEP_N_TARGET} target designs")
    _log(f"Total:      ~{len(jobs) * SWEEP_MAX_ITER * SWEEP_BATCH_SIZE:,} evaluations")
    _log(f"Workers:    {max(1, os.cpu_count() - 1)} (auto)")
    _log("")

    all_results = []
    total_start = time.monotonic()

    for i, job in enumerate(jobs, 1):
        if shutdown_requested:
            _log(f"Skipping remaining {len(jobs) - i + 1} jobs due to shutdown request")
            break

        _log(f"{'─' * 70}")
        _log(f"Job {i}/{len(jobs)}: {job['name']}")
        _log(f"{'─' * 70}")

        try:
            result = run_job(job, output_dir, log_file)
        except Exception as e:
            import traceback
            _log(f"FAILED: {e}")
            _log(traceback.format_exc())
            result = {
                "name": job["name"],
                "status": "failed",
                "error": str(e),
                "elapsed_s": 0,
            }

        all_results.append(result)

        # Progress estimate
        elapsed_total = time.monotonic() - total_start
        avg_per_job = elapsed_total / i
        remaining = len(jobs) - i
        eta_s = avg_per_job * remaining

        _log(f"Progress: {i}/{len(jobs)} done | "
             f"elapsed {elapsed_total/3600:.1f}h | "
             f"ETA {eta_s/3600:.1f}h | "
             f"avg {avg_per_job/60:.1f}min/job")
        _log("")

        # Save incremental summary after each job (crash-safe)
        _save_summary(output_dir, timestamp, all_results, total_start)

    # --- Final summary ---
    total_elapsed = time.monotonic() - total_start
    _save_summary(output_dir, timestamp, all_results, total_start)

    _log(f"\n{'=' * 70}")
    _log(f"SWEEP COMPLETE")
    _log(f"{'=' * 70}")
    _log(f"Total time:    {total_elapsed/3600:.1f} hours")
    _log(f"Jobs run:      {len(all_results)}/{len(jobs)}")

    n_success = sum(1 for r in all_results if r.get("status") == "success")
    n_empty = sum(1 for r in all_results if r.get("status") == "empty")
    n_failed = sum(1 for r in all_results if r.get("status") == "failed")
    total_designs = sum(r.get("n_designs", 0) for r in all_results)

    _log(f"Success:       {n_success}")
    _log(f"Empty:         {n_empty}")
    _log(f"Failed:        {n_failed}")
    _log(f"Total designs: {total_designs}")
    _log("")

    # Per-job summary table
    _log(f"{'Job':<42} {'Style':<10} {'Designs':>7} {'Cells':>5} "
         f"{'Score':>12} {'Time':>8} {'Status'}")
    _log("─" * 100)
    for r in all_results:
        if r.get("status") == "success":
            _log(f"{r['name']:<42} {r['weapon_style']:<10} {r['n_designs']:>7} "
                 f"{r['unique_cells']:>5} "
                 f"{r['score_min']:.3f}–{r['score_max']:.3f} "
                 f"{r['elapsed_s']/60:>7.1f}m  OK")
        elif r.get("status") == "empty":
            _log(f"{r['name']:<42} {r.get('weapon_style','?'):<10} {'0':>7} "
                 f"{'—':>5} {'—':>12} "
                 f"{r['elapsed_s']/60:>7.1f}m  EMPTY")
        else:
            _log(f"{r['name']:<42} {'?':<10} {'—':>7} {'—':>5} "
                 f"{'—':>12} {'—':>8}  FAILED")

    _log(f"\nSummary: {output_dir}/sweep_summary_{timestamp}.json")
    _log(f"Log:     {log_path}")

    log_file.close()


def _save_summary(output_dir: Path, timestamp: str, results: list, start_time: float):
    """Save incremental summary JSON (crash-safe)."""
    total_elapsed = time.monotonic() - start_time
    summary = {
        "timestamp": timestamp,
        "total_elapsed_s": round(total_elapsed, 1),
        "total_elapsed_h": round(total_elapsed / 3600, 2),
        "n_jobs_completed": len(results),
        "n_success": sum(1 for r in results if r.get("status") == "success"),
        "n_empty": sum(1 for r in results if r.get("status") == "empty"),
        "n_failed": sum(1 for r in results if r.get("status") == "failed"),
        "total_designs": sum(r.get("n_designs", 0) for r in results),
        "sweep_params": {
            "max_iter": SWEEP_MAX_ITER,
            "batch_size": SWEEP_BATCH_SIZE,
            "n_target": SWEEP_N_TARGET,
            "min_score": SWEEP_MIN_SCORE,
            "n_bins": SWEEP_N_BINS,
        },
        "jobs": results,
    }
    summary_path = output_dir / f"sweep_summary_{timestamp}.json"
    # Write to temp then rename for atomicity
    tmp_path = summary_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(summary, f, indent=2)
    tmp_path.rename(summary_path)


if __name__ == "__main__":
    main()
