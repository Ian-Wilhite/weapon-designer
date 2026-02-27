#!/usr/bin/env python3
"""Comprehensive baseline vs. enhanced comparison harness.

Runs all 10 weapon design cases in both baseline and enhanced modes,
repeats each N times for statistical significance, then prints a
detailed comparison table and saves all results to JSON.

Usage
─────
    python3 scripts/compare_baselines.py                          # all cases, 3 replicates
    python3 scripts/compare_baselines.py --replicates 5          # 5 replicates per case/mode
    python3 scripts/compare_baselines.py --mode enhanced-only    # skip baseline
    python3 scripts/compare_baselines.py --mode baseline-only    # skip enhanced
    python3 scripts/compare_baselines.py --cases heavyweight_disk,compact_bar
    python3 scripts/compare_baselines.py --quick                 # 1 replicate, fewer iters
    python3 scripts/compare_baselines.py --output-dir my_run/

Statistical note
────────────────
Mean ± std is computed across replicates.  A Welch two-sample t-test
(scipy.stats.ttest_ind) is used to assess whether the score difference
between baseline and enhanced is significant at the p < 0.05 level.
With 3 replicates this test has very low power; use --replicates 5
or higher for publication-quality statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure the package is importable when run from any directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weapon_designer.config import (
    WeaponConfig, Material, Mounting, Envelope,
    OptimizationParams, OptimizationWeights, OutputParams,
)


# ──────────────────────────────────────────────────────────────────────────────
# Case definitions
# ──────────────────────────────────────────────────────────────────────────────

def _make_cases() -> dict[str, WeaponConfig]:
    """Return all 10 evaluation cases with default (baseline) optimization params."""
    cases: dict[str, WeaponConfig] = {}

    # Shared weight profiles
    _w_disk  = OptimizationWeights(moment_of_inertia=0.30, bite=0.15, structural_integrity=0.20,
                                    mass_utilization=0.10, balance=0.10, impact_zone=0.15)
    _w_moi   = OptimizationWeights(moment_of_inertia=0.40, bite=0.10, structural_integrity=0.15,
                                    mass_utilization=0.10, balance=0.10, impact_zone=0.15)
    _w_egg3  = OptimizationWeights(moment_of_inertia=0.25, bite=0.20, structural_integrity=0.20,
                                    mass_utilization=0.10, balance=0.10, impact_zone=0.15)
    _w_egg4  = OptimizationWeights(moment_of_inertia=0.20, bite=0.25, structural_integrity=0.20,
                                    mass_utilization=0.10, balance=0.10, impact_zone=0.15)
    _w_drum  = OptimizationWeights(moment_of_inertia=0.25, bite=0.20, structural_integrity=0.25,
                                    mass_utilization=0.10, balance=0.05, impact_zone=0.15)

    cases["heavyweight_disk"] = WeaponConfig(
        material=Material("AR500", 7850, 1400, 50),
        weapon_style="disk", sheet_thickness_mm=12, weight_budget_kg=7.0, rpm=8000,
        mounting=Mounting(25.4, 60, 6, 8.0),
        envelope=Envelope(max_radius_mm=180),
        optimization=OptimizationParams(weights=_w_disk, num_fourier_terms=5,
                                        num_cutout_pairs=3, max_iterations=200, population_size=60),
    )
    cases["featherweight_disk"] = WeaponConfig(
        material=Material("S7_Tool_Steel", 7750, 1600, 56),
        weapon_style="disk", sheet_thickness_mm=6, weight_budget_kg=1.5, rpm=12000,
        mounting=Mounting(12.0, 25, 3, 4.0),
        envelope=Envelope(max_radius_mm=80),
        optimization=OptimizationParams(
            weights=OptimizationWeights(moment_of_inertia=0.35, bite=0.10, structural_integrity=0.20,
                                        mass_utilization=0.10, balance=0.10, impact_zone=0.15),
            num_fourier_terms=4, num_cutout_pairs=2, max_iterations=200, population_size=60),
    )
    cases["heavyweight_bar"] = WeaponConfig(
        material=Material("S7_Tool_Steel", 7750, 1600, 56),
        weapon_style="bar", sheet_thickness_mm=12, weight_budget_kg=5.0, rpm=10000,
        mounting=Mounting(19.05, 40, 4, 6.0),
        envelope=Envelope(max_radius_mm=200, max_length_mm=400, max_width_mm=70),
        optimization=OptimizationParams(weights=_w_disk, num_fourier_terms=4,
                                        num_cutout_pairs=2, max_iterations=200, population_size=60),
    )
    cases["compact_bar"] = WeaponConfig(
        material=Material("AR500", 7850, 1400, 50),
        weapon_style="bar", sheet_thickness_mm=10, weight_budget_kg=3.0, rpm=9000,
        mounting=Mounting(15.0, 30, 4, 5.0),
        envelope=Envelope(max_radius_mm=150, max_length_mm=280, max_width_mm=55),
        optimization=OptimizationParams(
            weights=OptimizationWeights(moment_of_inertia=0.30, bite=0.15, structural_integrity=0.20,
                                        mass_utilization=0.10, balance=0.10, impact_zone=0.15),
            num_fourier_terms=3, num_cutout_pairs=2, max_iterations=200, population_size=60),
    )
    cases["eggbeater_3blade"] = WeaponConfig(
        material=Material("AR500", 7850, 1400, 50),
        weapon_style="eggbeater", sheet_thickness_mm=8, weight_budget_kg=3.5, rpm=12000,
        mounting=Mounting(20.0, 38, 3, 5.5),
        envelope=Envelope(max_radius_mm=130),
        optimization=OptimizationParams(weights=_w_egg3, num_fourier_terms=4,
                                        num_cutout_pairs=2, max_iterations=200, population_size=60),
    )
    cases["eggbeater_2blade"] = WeaponConfig(
        material=Material("S7_Tool_Steel", 7750, 1600, 56),
        weapon_style="eggbeater", sheet_thickness_mm=10, weight_budget_kg=4.0, rpm=10000,
        mounting=Mounting(22.0, 45, 4, 6.0),
        envelope=Envelope(max_radius_mm=140),
        optimization=OptimizationParams(weights=_w_disk, num_fourier_terms=4,
                                        num_cutout_pairs=2, max_iterations=200, population_size=60),
    )
    cases["max_energy_disk"] = WeaponConfig(
        material=Material("AR500", 7850, 1400, 50),
        weapon_style="disk", sheet_thickness_mm=15, weight_budget_kg=8.0, rpm=7000,
        mounting=Mounting(30.0, 70, 6, 8.0),
        envelope=Envelope(max_radius_mm=200),
        optimization=OptimizationParams(weights=_w_moi, num_fourier_terms=5,
                                        num_cutout_pairs=3, max_iterations=200, population_size=60),
    )
    cases["drum_disk"] = WeaponConfig(
        material=Material("AR500", 7850, 1400, 50),
        weapon_style="disk", sheet_thickness_mm=8, weight_budget_kg=0.65, rpm=15000,
        mounting=Mounting(10.0, 20, 3, 3.5),
        envelope=Envelope(max_radius_mm=60),
        optimization=OptimizationParams(weights=_w_drum, num_fourier_terms=3,
                                        num_cutout_pairs=1, max_iterations=200, population_size=60),
    )
    cases["undercutter_bar"] = WeaponConfig(
        material=Material("AR500", 7850, 1400, 50),
        weapon_style="bar", sheet_thickness_mm=8, weight_budget_kg=4.0, rpm=8000,
        mounting=Mounting(20.0, 40, 4, 6.0),
        envelope=Envelope(max_radius_mm=180, max_length_mm=350, max_width_mm=90),
        optimization=OptimizationParams(weights=_w_disk, num_fourier_terms=4,
                                        num_cutout_pairs=2, max_iterations=200, population_size=60),
    )
    cases["eggbeater_4blade"] = WeaponConfig(
        material=Material("AR500", 7850, 1400, 50),
        weapon_style="eggbeater", sheet_thickness_mm=6, weight_budget_kg=2.5, rpm=14000,
        mounting=Mounting(15.0, 30, 4, 4.0),
        envelope=Envelope(max_radius_mm=100),
        optimization=OptimizationParams(weights=_w_egg4, num_fourier_terms=4,
                                        num_cutout_pairs=2, max_iterations=200, population_size=60),
    )
    return cases


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SingleRun:
    case_name: str
    mode: str           # "baseline" | "enhanced"
    replicate: int
    status: str         # "success" | "error"
    elapsed_s: float
    score: float = float("nan")
    metrics: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class AggResult:
    case_name: str
    mode: str
    n_success: int
    score_mean: float
    score_std: float
    scores: list
    metrics_mean: dict = field(default_factory=dict)
    metrics_std: dict = field(default_factory=dict)
    elapsed_mean_s: float = 0.0


@dataclass
class CaseComparison:
    case_name: str
    baseline: Optional[AggResult]
    enhanced: Optional[AggResult]
    score_delta: float = float("nan")
    pct_improvement: float = float("nan")
    p_value: float = float("nan")
    significant: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Single-run helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_quick_mode(cfg: WeaponConfig, quick: bool) -> WeaponConfig:
    """Reduce iteration counts for quick smoke-test runs."""
    if not quick:
        return cfg
    import copy
    cfg2 = copy.deepcopy(cfg)
    cfg2.optimization.max_iterations = 20
    cfg2.optimization.population_size = 20
    cfg2.optimization.phase1_iters = 5
    cfg2.optimization.phase2_iters = 3
    return cfg2


def _apply_medium_mode(cfg: WeaponConfig) -> WeaponConfig:
    """Medium-strength preset targeting <5 min per combination.

    Key driver: scipy DE uses population_size as a MULTIPLIER
    (actual_pop = population_size × n_params).  With the default
    population_size=60 and n_params≈22, actual pop = 1,320 individuals
    each requiring a FEA call — hence 45-min runtimes.

    Medium settings (population_size=8, n_bspline_points=8):
      actual_pop_p1 = 8 × 8  = 64   (phase 1, 25 gens)  → ~1,600 FEA calls
      actual_pop_p2 = 8 × 10 = 80   (phase 2, 12 gens)  → ~960  FEA calls
      total ≈ 2,560 FEA calls @ ~0.07 s each ≈ 3 min  ✓
    """
    import copy
    cfg2 = copy.deepcopy(cfg)
    cfg2.optimization.max_iterations = 50
    cfg2.optimization.population_size = 8
    cfg2.optimization.phase1_iters = 25
    cfg2.optimization.phase2_iters = 12
    # Enhanced-mode FEA settings (applied after _apply_enhanced_mode so these win)
    cfg2.optimization.n_bspline_points = 8
    cfg2.optimization.fea_coarse_spacing_mm = 18.0
    cfg2.optimization.fea_fine_spacing_mm = 9.0
    cfg2.optimization.convergence_patience = 8
    cfg2.optimization.convergence_min_delta = 0.003
    return cfg2


def _apply_enhanced_mode(cfg: WeaponConfig) -> WeaponConfig:
    """Return config with enhanced optimizer settings."""
    import copy
    cfg2 = copy.deepcopy(cfg)
    cfg2.optimization.evaluation_mode = "enhanced"
    cfg2.optimization.profile_type = "bspline"
    cfg2.optimization.cutout_type = "superellipse"
    cfg2.optimization.fea_interval = 0          # no GIF in comparison runs
    cfg2.optimization.fea_coarse_spacing_mm = 12.0
    cfg2.optimization.fea_fine_spacing_mm = 6.0
    cfg2.optimization.n_bspline_points = 12
    return cfg2


def run_baseline_single(name: str, cfg: WeaponConfig, output_dir: Path,
                         replicate: int, quiet: bool = True) -> SingleRun:
    """Run one baseline optimization replicate."""
    from weapon_designer.optimizer import optimize

    t0 = time.perf_counter()
    try:
        result = optimize(cfg, verbose=not quiet)
        poly = result.get("weapon_polygon")
        if poly is None or poly.is_empty:
            return SingleRun(name, "baseline", replicate, "error",
                             time.perf_counter() - t0, error="Empty polygon")

        # optimizer already computes metrics and score; use them directly
        metrics = result.get("metrics", {})
        score   = result.get("score", float("nan"))
        penalty = result.get("penalty", 1.0)
        final   = score * penalty

        return SingleRun(name, "baseline", replicate, "success",
                         time.perf_counter() - t0, score=final, metrics=metrics)

    except Exception as e:
        return SingleRun(name, "baseline", replicate, "error",
                         time.perf_counter() - t0, error=traceback.format_exc(limit=3))


def run_enhanced_single(name: str, cfg: WeaponConfig, output_dir: Path,
                         replicate: int, quiet: bool = True) -> SingleRun:
    """Run one enhanced optimization replicate."""
    from weapon_designer.optimizer_enhanced import optimize_enhanced
    import copy

    t0 = time.perf_counter()
    try:
        sub_dir = output_dir / f"{name}_enh_r{replicate:02d}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        cfg_out = copy.deepcopy(cfg)
        cfg_out.output.dxf_path = str(sub_dir / f"{name}.dxf")
        cfg_out.output.stats_path = str(sub_dir / f"{name}_stats.json")

        # optimize_enhanced(cfg, case_dir, verbose)
        result = optimize_enhanced(cfg_out, case_dir=sub_dir, verbose=not quiet)
        score   = result.get("score", float("nan"))
        penalty = result.get("penalty", 1.0)
        final   = score * penalty
        metrics = result.get("metrics", {})

        return SingleRun(name, "enhanced", replicate, "success",
                         time.perf_counter() - t0, score=final, metrics=metrics)

    except Exception as e:
        return SingleRun(name, "enhanced", replicate, "error",
                         time.perf_counter() - t0, error=traceback.format_exc(limit=3))


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate(runs: list[SingleRun]) -> AggResult:
    success = [r for r in runs if r.status == "success"]
    mode = runs[0].mode if runs else "unknown"
    name = runs[0].case_name if runs else "unknown"

    if not success:
        return AggResult(name, mode, 0, float("nan"), float("nan"), [])

    scores = [r.score for r in success]
    score_mean = float(np.mean(scores))
    score_std  = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    elapsed    = float(np.mean([r.elapsed_s for r in success]))

    # Aggregate per-metric mean/std
    all_keys = set()
    for r in success:
        all_keys.update(r.metrics.keys())
    metrics_mean, metrics_std = {}, {}
    for k in all_keys:
        vals = [r.metrics[k] for r in success if k in r.metrics]
        if vals:
            metrics_mean[k] = float(np.mean(vals))
            metrics_std[k]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    return AggResult(name, mode, len(success), score_mean, score_std,
                     scores, metrics_mean, metrics_std, elapsed)


def _compare(baseline: Optional[AggResult], enhanced: Optional[AggResult]) -> CaseComparison:
    name = (baseline or enhanced).case_name

    if baseline is None or enhanced is None:
        return CaseComparison(name, baseline, enhanced)

    if baseline.n_success == 0 or enhanced.n_success == 0:
        return CaseComparison(name, baseline, enhanced)

    delta = enhanced.score_mean - baseline.score_mean
    pct   = (delta / max(abs(baseline.score_mean), 1e-9)) * 100.0

    p_value = float("nan")
    significant = False
    if len(baseline.scores) >= 2 and len(enhanced.scores) >= 2:
        try:
            from scipy.stats import ttest_ind
            _, p_value = ttest_ind(enhanced.scores, baseline.scores, equal_var=False)
            p_value = float(p_value)
            significant = p_value < 0.05
        except ImportError:
            pass

    return CaseComparison(name, baseline, enhanced, delta, pct, p_value, significant)


# ──────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ──────────────────────────────────────────────────────────────────────────────

METRIC_DISPLAY = [
    ("moi_kg_mm2",          "MOI(kg·mm²)",  "8.0f"),
    ("energy_joules",       "Energy(J)",    "7.1f"),
    ("bite_mm",             "Bite(mm)",     "7.2f"),
    ("fea_safety_factor",   "SF",           "6.2f"),
    ("mass_utilization",    "MassUtil",     "7.3f"),
    ("com_offset_mm",       "CoM(mm)",      "7.2f"),
]


def _fmt(val, fmt_str: str) -> str:
    try:
        return format(val, fmt_str)
    except (TypeError, ValueError):
        return "   N/A "


def print_run_header(log):
    log("=" * 100)
    log("WEAPON DESIGNER — BASELINE vs. ENHANCED COMPARISON")
    log("=" * 100)


def print_agg_row(label: str, agg: Optional[AggResult], log):
    if agg is None or agg.n_success == 0:
        log(f"  {label:12s}  NO DATA")
        return

    score_str = f"{agg.score_mean:.4f} ± {agg.score_std:.4f}"
    m = agg.metrics_mean
    parts = [f"  {label:12s}  Score: {score_str:20s}"]
    for key, disp, fmt in METRIC_DISPLAY:
        if key in m:
            parts.append(f"  {disp}={_fmt(m[key], fmt)}")
    parts.append(f"  time={agg.elapsed_mean_s:.0f}s")
    log("".join(parts))


def print_comparison_table(comparisons: list[CaseComparison], log):
    log("")
    log("=" * 100)
    log("COMPARISON SUMMARY TABLE")
    log("=" * 100)

    hdr = (f"{'Case':28s}  {'Base Score':>12s}  {'Enh Score':>12s}  "
           f"{'Δ Score':>9s}  {'Improv%':>8s}  {'p-value':>8s}  {'Sig?':>5s}")
    log(hdr)
    log("-" * 100)

    for c in comparisons:
        b_score = f"{c.baseline.score_mean:.4f}±{c.baseline.score_std:.4f}" \
                  if c.baseline and c.baseline.n_success > 0 else "    N/A   "
        e_score = f"{c.enhanced.score_mean:.4f}±{c.enhanced.score_std:.4f}" \
                  if c.enhanced and c.enhanced.n_success > 0 else "    N/A   "

        delta_str = f"{c.score_delta:+.4f}" if not np.isnan(c.score_delta) else "    N/A"
        pct_str   = f"{c.pct_improvement:+.1f}%" if not np.isnan(c.pct_improvement) else "   N/A"
        p_str     = f"{c.p_value:.3f}" if not np.isnan(c.p_value) else "   N/A"
        sig_str   = "  *** " if c.significant else "      "

        log(f"{c.case_name:28s}  {b_score:>12s}  {e_score:>12s}  "
            f"{delta_str:>9s}  {pct_str:>8s}  {p_str:>8s}  {sig_str}")

    log("")

    # Per-metric comparison for cases with both modes
    paired = [c for c in comparisons
              if c.baseline and c.enhanced
              and c.baseline.n_success > 0 and c.enhanced.n_success > 0]
    if paired:
        log("PER-METRIC COMPARISON (baseline → enhanced)")
        log("-" * 100)
        for key, disp, fmt in METRIC_DISPLAY:
            log(f"  {disp:15s}", end="")
            for c in paired:
                bv = c.baseline.metrics_mean.get(key, float("nan"))
                ev = c.enhanced.metrics_mean.get(key, float("nan"))
                if np.isnan(bv) or np.isnan(ev):
                    log(f"  {c.case_name[:10]:10s}: N/A", end="")
                else:
                    log(f"  {c.case_name[:10]:10s}: {_fmt(bv, fmt)}→{_fmt(ev, fmt)}", end="")
            log("")


def print_final_summary(comparisons: list[CaseComparison], total_elapsed: float, log):
    log("")
    log("=" * 100)
    log("FINAL SUMMARY")
    log("=" * 100)

    paired = [c for c in comparisons
              if c.baseline and c.enhanced
              and c.baseline.n_success > 0 and c.enhanced.n_success > 0
              and not np.isnan(c.pct_improvement)]

    if paired:
        improvements = [c.pct_improvement for c in paired]
        log(f"  Cases with both modes:   {len(paired)}")
        log(f"  Mean improvement:        {np.mean(improvements):+.1f}%")
        log(f"  Median improvement:      {np.median(improvements):+.1f}%")
        log(f"  Cases enhanced > base:   {sum(1 for x in improvements if x > 0)}/{len(paired)}")
        sig = [c for c in paired if c.significant]
        log(f"  Statistically sig (p<0.05): {len(sig)}/{len(paired)}")

    log(f"  Total wall-clock time:   {total_elapsed/60:.1f} minutes")
    log("")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline vs. enhanced weapon optimizer comparison.")
    parser.add_argument("--cases", default=None,
                        help="Comma-separated case names (default: all 10)")
    parser.add_argument("--mode", default="compare",
                        choices=["compare", "baseline-only", "enhanced-only"],
                        help="Which optimizer mode(s) to run")
    parser.add_argument("--replicates", type=int, default=3,
                        help="Number of independent runs per case/mode (default: 3)")
    parser.add_argument("--output-dir", default="eval_results/compare",
                        help="Directory for results and logs")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-iteration optimizer output")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced iterations for smoke test (1 replicate)")
    parser.add_argument("--medium", action="store_true",
                        help="Medium-strength preset: <5 min per combination "
                             "(popsize=8, 50 iters, 8 bspline pts, 18mm mesh)")
    args = parser.parse_args()

    if args.quick:
        args.replicates = 1

    # Setup output directory and log file
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = out_dir / f"compare_{ts}.log"
    results_path = out_dir / f"compare_{ts}_results.json"

    log_fh = open(log_path, "w")

    def log(*a, end="\n"):
        msg = " ".join(str(x) for x in a) + end
        print(msg, end="")
        log_fh.write(msg)
        log_fh.flush()

    all_cases = _make_cases()
    if args.cases:
        selected = [c.strip() for c in args.cases.split(",")]
        unknown = [c for c in selected if c not in all_cases]
        if unknown:
            print(f"Unknown cases: {unknown}. Valid: {list(all_cases)}")
            sys.exit(1)
        cases = {k: all_cases[k] for k in selected}
    else:
        cases = all_cases

    run_baseline = args.mode in ("compare", "baseline-only")
    run_enhanced = args.mode in ("compare", "enhanced-only")

    print_run_header(log)
    log(f"  Cases:         {list(cases)}")
    log(f"  Mode:          {args.mode}")
    log(f"  Replicates:    {args.replicates}")
    log(f"  Quick:         {args.quick}")
    log(f"  Output dir:    {out_dir}")
    log(f"  Started:       {datetime.now().isoformat(timespec='seconds')}")
    log("")

    t_total_start = time.perf_counter()
    all_runs: list[SingleRun] = []
    comparisons: list[CaseComparison] = []

    total_jobs = len(cases) * ((run_baseline + run_enhanced)) * args.replicates
    job_idx = 0

    for case_name, base_cfg in cases.items():
        log("=" * 80)
        log(f"CASE: {case_name}")
        log("=" * 80)

        baseline_runs: list[SingleRun] = []
        enhanced_runs: list[SingleRun] = []

        # ── Baseline replicates ────────────────────────────────────────────
        if run_baseline:
            log(f"\n  [BASELINE]  {args.replicates} replicate(s)")
            for rep in range(args.replicates):
                job_idx += 1
                cfg = _apply_quick_mode(base_cfg, args.quick)
                if args.medium:
                    cfg = _apply_medium_mode(cfg)
                log(f"    Replicate {rep+1}/{args.replicates}  ({job_idx}/{total_jobs}) ...",
                    end="")
                t0 = time.perf_counter()
                run = run_baseline_single(case_name, cfg, out_dir, rep, quiet=args.quiet)
                elapsed = time.perf_counter() - t0
                status = "OK" if run.status == "success" else f"ERR:{run.error[:60]}"
                score_str = f"score={run.score:.4f}" if not np.isnan(run.score) else "score=N/A"
                log(f" {score_str}  [{elapsed:.0f}s]  {status}")
                baseline_runs.append(run)
                all_runs.append(run)

        # ── Enhanced replicates ────────────────────────────────────────────
        if run_enhanced:
            log(f"\n  [ENHANCED]  {args.replicates} replicate(s)")
            for rep in range(args.replicates):
                job_idx += 1
                cfg = _apply_quick_mode(base_cfg, args.quick)
                cfg = _apply_enhanced_mode(cfg)
                if args.medium:
                    # Apply after enhanced so medium's tighter FEA/bspline settings win
                    cfg = _apply_medium_mode(cfg)
                log(f"    Replicate {rep+1}/{args.replicates}  ({job_idx}/{total_jobs}) ...",
                    end="")
                t0 = time.perf_counter()
                run = run_enhanced_single(case_name, cfg, out_dir, rep, quiet=args.quiet)
                elapsed = time.perf_counter() - t0
                status = "OK" if run.status == "success" else f"ERR:{run.error[:60]}"
                score_str = f"score={run.score:.4f}" if not np.isnan(run.score) else "score=N/A"
                log(f" {score_str}  [{elapsed:.0f}s]  {status}")
                enhanced_runs.append(run)
                all_runs.append(run)

        # ── Per-case aggregate ─────────────────────────────────────────────
        log("")
        b_agg = _aggregate(baseline_runs) if baseline_runs else None
        e_agg = _aggregate(enhanced_runs) if enhanced_runs else None

        if b_agg:
            print_agg_row("BASELINE", b_agg, log)
        if e_agg:
            print_agg_row("ENHANCED", e_agg, log)

        cmp = _compare(b_agg, e_agg)
        comparisons.append(cmp)

        if b_agg and e_agg and b_agg.n_success > 0 and e_agg.n_success > 0:
            sign = "↑" if cmp.score_delta > 0 else "↓"
            sig_note = " ***" if cmp.significant else ""
            log(f"  Delta: {cmp.score_delta:+.4f} ({cmp.pct_improvement:+.1f}%){sign}{sig_note}"
                f"  p={cmp.p_value:.3f}" if not np.isnan(cmp.p_value) else
                f"  Delta: {cmp.score_delta:+.4f} ({cmp.pct_improvement:+.1f}%){sign}{sig_note}")
        log("")

    total_elapsed = time.perf_counter() - t_total_start

    # ── Final tables ───────────────────────────────────────────────────────
    print_comparison_table(comparisons, log)
    print_final_summary(comparisons, total_elapsed, log)

    # ── Save raw results ───────────────────────────────────────────────────
    def _make_serializable(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    import dataclasses

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            d = dataclasses.asdict(obj)
            return {k: _to_dict(v) for k, v in d.items()}
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_dict(v) for v in obj]
        return _make_serializable(obj)

    output = {
        "timestamp": ts,
        "args": vars(args),
        "total_elapsed_s": total_elapsed,
        "cases": list(cases.keys()),
        "runs": [_to_dict(r) for r in all_runs],
        "comparisons": [
            {
                "case_name": c.case_name,
                "score_delta": _make_serializable(c.score_delta),
                "pct_improvement": _make_serializable(c.pct_improvement),
                "p_value": _make_serializable(c.p_value),
                "significant": c.significant,
                "baseline": _to_dict(c.baseline) if c.baseline else None,
                "enhanced": _to_dict(c.enhanced) if c.enhanced else None,
            }
            for c in comparisons
        ],
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log(f"Results saved: {results_path}")
    log(f"Log saved:     {log_path}")
    log_fh.close()


if __name__ == "__main__":
    main()
