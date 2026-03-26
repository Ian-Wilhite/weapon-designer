#!/usr/bin/env python3
"""Evaluation script: runs multiple weapon design cases through full optimization.

Produces detailed logs with all 5 objective parameters and convergence history.
Expected runtime: ~3-4 hours for all cases.

Usage:
    python evaluate.py [--output-dir DIR] [--cases CASE1,CASE2,...] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weapon_designer.config import WeaponConfig, Material, Mounting, Envelope, OptimizationWeights, OptimizationParams, OutputParams


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

CASES: dict[str, WeaponConfig] = {}

# Case 1: Heavyweight disk spinner (classic BattleBots style)
CASES["heavyweight_disk"] = WeaponConfig(
    material=Material(name="AR500", density_kg_m3=7850, yield_strength_mpa=1400, hardness_hrc=50),
    weapon_style="disk",
    sheet_thickness_mm=12,
    weight_budget_kg=7.0,
    rpm=8000,
    mounting=Mounting(bore_diameter_mm=25.4, bolt_circle_diameter_mm=60, num_bolts=6, bolt_hole_diameter_mm=8.0),
    envelope=Envelope(max_radius_mm=180),
    optimization=OptimizationParams(
        weights=OptimizationWeights(moment_of_inertia=0.30, bite=0.15, structural_integrity=0.20, mass_utilization=0.10, balance=0.10, impact_zone=0.15),
        num_fourier_terms=5, num_cutout_pairs=3, max_iterations=200, population_size=60,
    ),
)

# Case 2: Lightweight disk spinner (featherweight)
CASES["featherweight_disk"] = WeaponConfig(
    material=Material(name="S7_Tool_Steel", density_kg_m3=7750, yield_strength_mpa=1600, hardness_hrc=56),
    weapon_style="disk",
    sheet_thickness_mm=6,
    weight_budget_kg=1.5,
    rpm=12000,
    mounting=Mounting(bore_diameter_mm=12.0, bolt_circle_diameter_mm=25, num_bolts=3, bolt_hole_diameter_mm=4.0),
    envelope=Envelope(max_radius_mm=80),
    optimization=OptimizationParams(
        weights=OptimizationWeights(moment_of_inertia=0.35, bite=0.10, structural_integrity=0.20, mass_utilization=0.10, balance=0.10, impact_zone=0.15),
        num_fourier_terms=4, num_cutout_pairs=2, max_iterations=200, population_size=60,
    ),
)

# Case 3: Max-energy disk (MOI-focused)
CASES["max_energy_disk"] = WeaponConfig(
    material=Material(name="AR500", density_kg_m3=7850, yield_strength_mpa=1400, hardness_hrc=50),
    weapon_style="disk",
    sheet_thickness_mm=15,
    weight_budget_kg=8.0,
    rpm=7000,
    mounting=Mounting(bore_diameter_mm=30.0, bolt_circle_diameter_mm=70, num_bolts=6, bolt_hole_diameter_mm=8.0),
    envelope=Envelope(max_radius_mm=200),
    optimization=OptimizationParams(
        weights=OptimizationWeights(moment_of_inertia=0.40, bite=0.10, structural_integrity=0.15, mass_utilization=0.10, balance=0.10, impact_zone=0.15),
        num_fourier_terms=5, num_cutout_pairs=3, max_iterations=200, population_size=60,
    ),
)



# ---------------------------------------------------------------------------
# Convergence-tracking optimizer wrapper
# ---------------------------------------------------------------------------

def run_case_with_logging(
    name: str,
    cfg: WeaponConfig,
    output_dir: Path,
    log_file,
) -> dict:
    """Run a single optimization case with convergence logging."""
    from weapon_designer.optimizer import (
        _get_profile_bounds, _get_cutout_bounds,
        _profile_objective, _cutout_objective,
    )
    from weapon_designer.archetypes import seed_population_from_archetypes
    from weapon_designer.parametric import build_weapon_polygon
    from weapon_designer.geometry import assemble_weapon
    from weapon_designer.objectives import compute_metrics, weighted_score
    from weapon_designer.constraints import validate_geometry, constraint_penalty
    from weapon_designer.exporter import export_dxf, export_snapshot
    from scipy.optimize import differential_evolution

    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    profile_bounds = _get_profile_bounds(cfg)
    cutout_bounds = _get_cutout_bounds(cfg)
    n_profile = len(profile_bounds)

    workers = max(1, os.cpu_count() or 1)

    def _log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{name}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    _log(f"Starting: style={cfg.weapon_style}, budget={cfg.weight_budget_kg}kg, "
         f"rpm={cfg.rpm}, thickness={cfg.sheet_thickness_mm}mm, "
         f"envelope_r={cfg.envelope.max_radius_mm}mm")
    _log(f"Weights: MOI={cfg.optimization.weights.moment_of_inertia}, "
         f"bite={cfg.optimization.weights.bite}, "
         f"struct={cfg.optimization.weights.structural_integrity}, "
         f"mass_util={cfg.optimization.weights.mass_utilization}, "
         f"balance={cfg.optimization.weights.balance}, "
         f"impact={cfg.optimization.weights.impact_zone}")
    _log(f"Profile params: {n_profile}, Cutout params: {len(cutout_bounds)}, Workers: {workers}")

    # --- Phase 1: Profile ---
    _log("=== Phase 1: Profile Optimization ===")
    phase1_iters = max(10, int(cfg.optimization.max_iterations * 0.7))
    pop_size = cfg.optimization.population_size

    archetype_pop = seed_population_from_archetypes(
        cfg, profile_bounds, pop_size * n_profile,
        rng=np.random.default_rng(42), profile_only=True,
    )

    # Create snapshots directory
    snapshot_dir = case_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_interval = cfg.optimization.snapshot_interval

    # Compute cutout stride
    cutout_stride = 3 + 2 * cfg.optimization.num_cutout_fourier_terms

    phase1_history = []

    class Phase1Callback:
        def __init__(self):
            self.step = 0
            self.start_time = time.time()

        def __call__(self, xk, convergence):
            self.step += 1
            elapsed = time.time() - self.start_time
            # Evaluate current best
            C = cfg.optimization.num_cutout_pairs
            x_full = np.concatenate([xk, np.zeros(C * cutout_stride)])
            try:
                outer, params, cutout_polys = build_weapon_polygon(x_full, cfg)
                weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
                weapon = validate_geometry(weapon)
                metrics = compute_metrics(weapon, cfg)
                score = weighted_score(metrics, cfg)

                entry = {
                    "phase": 1, "step": self.step, "elapsed_s": round(elapsed, 1),
                    "score": round(score, 6),
                    "moi_kg_mm2": round(metrics["moi_kg_mm2"], 2),
                    "bite_mm": round(metrics["bite_mm"], 2),
                    "structural": round(metrics["structural_integrity"], 4),
                    "mass_util": round(metrics["mass_utilization"], 4),
                    "com_offset_mm": round(metrics["com_offset_mm"], 3),
                    "mass_kg": round(metrics["mass_kg"], 4),
                    "energy_j": round(metrics["energy_joules"], 1),
                    "convergence": round(convergence, 6),
                }
                phase1_history.append(entry)

                if self.step % 10 == 0 or self.step <= 3:
                    _log(f"  P1 step {self.step:3d}: score={score:.4f} "
                         f"MOI={metrics['moi_kg_mm2']:.1f} bite={metrics['bite_mm']:.1f}mm "
                         f"struct={metrics['structural_integrity']:.3f} "
                         f"mass={metrics['mass_kg']:.3f}kg "
                         f"CoM={metrics['com_offset_mm']:.2f}mm "
                         f"impact={metrics.get('impact_zone', 0):.3f} "
                         f"conv={convergence:.4f} [{elapsed:.0f}s]")

                # Snapshot export
                if self.step % snapshot_interval == 0:
                    try:
                        export_snapshot(weapon, cfg, snapshot_dir, f"p1_step{self.step:03d}")
                    except Exception:
                        pass
            except Exception as e:
                _log(f"  P1 step {self.step}: callback error: {e}")

    p1_cb = Phase1Callback()

    result1 = differential_evolution(
        _profile_objective,
        bounds=profile_bounds,
        args=(cfg,),
        maxiter=phase1_iters,
        popsize=pop_size,
        seed=42,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.8,
        workers=workers,
        updating="deferred",
        disp=False,
        init=archetype_pop,
        callback=p1_cb,
    )

    best_profile = result1.x
    _log(f"Phase 1 complete: best_score={-result1.fun:.4f}, evals={result1.nfev}")

    # Snapshot at phase 1 boundary
    try:
        C = cfg.optimization.num_cutout_pairs
        x_snap = np.concatenate([best_profile, np.zeros(C * cutout_stride)])
        outer_snap, _, cutout_snap = build_weapon_polygon(x_snap, cfg)
        weapon_snap = assemble_weapon(outer_snap, cfg.mounting, cutout_snap)
        weapon_snap = validate_geometry(weapon_snap)
        export_snapshot(weapon_snap, cfg, snapshot_dir, "p1_final")
    except Exception:
        pass

    # --- Phase 2: Cutouts ---
    phase2_history = []
    if cfg.optimization.num_cutout_pairs > 0 and len(cutout_bounds) > 0:
        _log("=== Phase 2: Cutout Optimization ===")
        phase2_iters = max(10, int(cfg.optimization.max_iterations * 0.3))

        class Phase2Callback:
            def __init__(self):
                self.step = 0
                self.start_time = time.time()

            def __call__(self, xk, convergence):
                self.step += 1
                elapsed = time.time() - self.start_time
                try:
                    x_full = np.concatenate([best_profile, xk])
                    outer, params, cutout_polys = build_weapon_polygon(x_full, cfg)
                    weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
                    weapon = validate_geometry(weapon)
                    metrics = compute_metrics(weapon, cfg)
                    score = weighted_score(metrics, cfg)
                    penalty = constraint_penalty(weapon, cfg)

                    entry = {
                        "phase": 2, "step": self.step, "elapsed_s": round(elapsed, 1),
                        "score": round(score, 6),
                        "penalty": round(penalty, 6),
                        "effective_score": round(score * penalty, 6),
                        "moi_kg_mm2": round(metrics["moi_kg_mm2"], 2),
                        "bite_mm": round(metrics["bite_mm"], 2),
                        "structural": round(metrics["structural_integrity"], 4),
                        "mass_util": round(metrics["mass_utilization"], 4),
                        "com_offset_mm": round(metrics["com_offset_mm"], 3),
                        "mass_kg": round(metrics["mass_kg"], 4),
                        "energy_j": round(metrics["energy_joules"], 1),
                        "convergence": round(convergence, 6),
                    }
                    phase2_history.append(entry)

                    if self.step % 10 == 0 or self.step <= 3:
                        _log(f"  P2 step {self.step:3d}: score={score:.4f}*{penalty:.3f}={score*penalty:.4f} "
                             f"MOI={metrics['moi_kg_mm2']:.1f} bite={metrics['bite_mm']:.1f}mm "
                             f"struct={metrics['structural_integrity']:.3f} "
                             f"mass={metrics['mass_kg']:.3f}/{cfg.weight_budget_kg:.1f}kg "
                             f"CoM={metrics['com_offset_mm']:.2f}mm "
                             f"impact={metrics.get('impact_zone', 0):.3f} "
                             f"[{elapsed:.0f}s]")

                    # Snapshot export
                    if self.step % snapshot_interval == 0:
                        try:
                            export_snapshot(weapon, cfg, snapshot_dir, f"p2_step{self.step:03d}")
                        except Exception:
                            pass
                except Exception as e:
                    _log(f"  P2 step {self.step}: callback error: {e}")

        p2_cb = Phase2Callback()

        result2 = differential_evolution(
            _cutout_objective,
            bounds=cutout_bounds,
            args=(best_profile, cfg),
            maxiter=phase2_iters,
            popsize=pop_size,
            seed=42,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.8,
            workers=workers,
            updating="deferred",
            disp=False,
            callback=p2_cb,
        )

        best_cutouts = result2.x
        _log(f"Phase 2 complete: best_score={-result2.fun:.4f}, evals={result2.nfev}")
    else:
        C = cfg.optimization.num_cutout_pairs
        best_cutouts = np.zeros(C * cutout_stride)

    # --- Final evaluation with FEA ---
    _log("Running final FEA evaluation...")
    x_best = np.concatenate([best_profile, best_cutouts])
    outer, params, cutout_polys = build_weapon_polygon(x_best, cfg)
    weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
    weapon = validate_geometry(weapon)
    metrics = compute_metrics(weapon, cfg, use_fea=True)
    score = weighted_score(metrics, cfg)
    penalty = constraint_penalty(weapon, cfg, cutout_polys=cutout_polys)

    # Log final results
    _log("=== Final Results ===")
    _log(f"  Overall score: {score:.4f} (penalty: {penalty:.4f})")
    _log(f"  --- Objectives ---")
    _log(f"  MOI:           {metrics['moi_kg_mm2']:.2f} kg*mm^2")
    _log(f"  Bite:          {metrics['bite_mm']:.2f} mm")
    _log(f"  Structural:    {metrics['structural_integrity']:.4f}")
    _log(f"  Mass util:     {metrics['mass_utilization']:.4f} ({metrics['mass_kg']:.3f}/{cfg.weight_budget_kg:.1f} kg)")
    _log(f"  CoM offset:    {metrics['com_offset_mm']:.3f} mm")
    _log(f"  Impact zone:   {metrics.get('impact_zone', 0):.4f}")
    _log(f"  --- Derived ---")
    _log(f"  Energy:        {metrics['energy_joules']:.1f} J @ {cfg.rpm} RPM")
    if "fea_peak_stress_mpa" in metrics:
        _log(f"  FEA peak:      {metrics['fea_peak_stress_mpa']:.1f} MPa")
        _log(f"  FEA safety:    {metrics['fea_safety_factor']:.2f}")
        _log(f"  FEA elements:  {metrics['fea_n_elements']}")

    # Export DXF and final snapshot
    dxf_path = str(case_dir / f"{name}.dxf")
    export_dxf(weapon, dxf_path, cfg)
    try:
        export_snapshot(weapon, cfg, snapshot_dir, "final")
    except Exception:
        pass

    # Export stats + convergence
    stats = {
        "case_name": name,
        "weapon_style": cfg.weapon_style,
        "material": cfg.material.name,
        "sheet_thickness_mm": cfg.sheet_thickness_mm,
        "rpm": cfg.rpm,
        "weight_budget_kg": cfg.weight_budget_kg,
        "envelope": {
            "max_radius_mm": cfg.envelope.max_radius_mm,
            "max_length_mm": cfg.envelope.max_length_mm,
            "max_width_mm": cfg.envelope.max_width_mm,
        },
        "optimization_weights": {
            "moment_of_inertia": cfg.optimization.weights.moment_of_inertia,
            "bite": cfg.optimization.weights.bite,
            "structural_integrity": cfg.optimization.weights.structural_integrity,
            "mass_utilization": cfg.optimization.weights.mass_utilization,
            "balance": cfg.optimization.weights.balance,
            "impact_zone": cfg.optimization.weights.impact_zone,
        },
        "final_score": round(score, 6),
        "constraint_penalty": round(penalty, 6),
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        "convergence_phase1": phase1_history,
        "convergence_phase2": phase2_history,
        "dxf_path": dxf_path,
    }

    stats_path = case_dir / f"{name}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    _log(f"Exported: {dxf_path}")
    _log(f"Stats: {stats_path}")
    _log("")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_case_enhanced(
    name: str,
    cfg: WeaponConfig,
    output_dir: Path,
    log_file,
) -> dict:
    """Run the enhanced optimizer on a single case and return a stats dict.

    Mirrors the structure of run_case_with_logging() so the two can be
    compared column-by-column in the summary table.
    """
    from weapon_designer.optimizer_enhanced import optimize_enhanced
    from weapon_designer.exporter import export_dxf, export_snapshot

    case_dir = output_dir / (name + "_enhanced")
    case_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{name}+enh] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    _log(f"Starting enhanced: style={cfg.weapon_style}, "
         f"budget={cfg.weight_budget_kg}kg, rpm={cfg.rpm}")

    # Clone config and stamp as enhanced so optimiser knows which paths to use
    import copy
    ecfg = copy.deepcopy(cfg)
    ecfg.optimization.evaluation_mode = "enhanced"
    ecfg.optimization.cutout_type     = "superellipse"

    result = optimize_enhanced(ecfg, case_dir, verbose=False)
    for msg in result.get("logs", []):
        _log(msg)

    metrics = result["metrics"]
    score   = result["score"]
    penalty = result["penalty"]

    _log(f"Enhanced final score={score:.4f} (penalty={penalty:.4f})")

    # Export DXF and snapshot
    dxf_path = str(case_dir / f"{name}_enhanced.dxf")
    export_dxf(result["weapon_polygon"], dxf_path, ecfg)
    try:
        snap_dir = case_dir / "snapshots"
        snap_dir.mkdir(exist_ok=True)
        export_snapshot(result["weapon_polygon"], ecfg, snap_dir, "final_enhanced")
    except Exception:
        pass

    # Convergence history — flatten both phases
    convergence = result.get("convergence_p1", []) + result.get("convergence_p2", [])

    stats = {
        "case_name":          name + "_enhanced",
        "weapon_style":       cfg.weapon_style,
        "material":           cfg.material.name,
        "sheet_thickness_mm": cfg.sheet_thickness_mm,
        "rpm":                cfg.rpm,
        "weight_budget_kg":   cfg.weight_budget_kg,
        "evaluation_mode":    "enhanced",
        "final_score":        round(score, 6),
        "constraint_penalty": round(penalty, 6),
        "metrics":            {k: round(v, 6) if isinstance(v, float) else v
                               for k, v in metrics.items()
                               if not k.startswith("_")},
        "convergence":        convergence,
        "dxf_path":           dxf_path,
        "gif_phase1":         str(result["gif_phase1"]) if result["gif_phase1"] else None,
        "gif_phase2":         str(result["gif_phase2"]) if result["gif_phase2"] else None,
    }

    stats_path = case_dir / f"{name}_enhanced_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    _log(f"Exported DXF:  {dxf_path}")
    _log(f"Stats:         {stats_path}")
    if result["gif_phase1"]:
        _log(f"GIF Phase 1:   {result['gif_phase1']}")
    if result["gif_phase2"]:
        _log(f"GIF Phase 2:   {result['gif_phase2']}")
    _log("")

    return stats


def _print_comparison_table(baseline_results: list, enhanced_results: list, log_fn) -> None:
    """Print a side-by-side comparison of baseline vs enhanced results."""
    by_name: dict = {r["case_name"]: r for r in baseline_results if r["status"] == "success"}
    by_enh:  dict = {r["case_name"].replace("_enhanced", ""): r
                     for r in enhanced_results if r.get("status") == "success"}

    log_fn("")
    log_fn("=" * 130)
    log_fn("BASELINE vs ENHANCED COMPARISON")
    log_fn("=" * 130)
    hdr = (
        f"{'Case':<25} "
        f"{'B-Score':>8} {'E-Score':>8} {'ΔScore':>7}  "
        f"{'B-MOI':>8} {'E-MOI':>8} {'ΔMOI':>7}  "
        f"{'B-Bite':>7} {'E-Bite':>7} {'E-Teeth':>7}  "
        f"{'B-SF':>6} {'E-SF':>6}"
    )
    log_fn(hdr)
    log_fn("-" * 130)

    for name in sorted(by_name.keys()):
        br = by_name[name]
        er = by_enh.get(name)
        bm = br["metrics"]
        if er is None:
            log_fn(f"  {name:<25} (no enhanced result)")
            continue
        em = er["metrics"]
        bs, es = br["final_score"], er["final_score"]
        ds = (es - bs) / max(abs(bs), 1e-9) * 100
        bm_val, em_val = bm["moi_kg_mm2"], em["moi_kg_mm2"]
        dm = (em_val - bm_val) / max(abs(bm_val), 1e-9) * 100
        bb, eb = bm["bite_mm"], em["bite_mm"]
        bsf = bm.get("fea_safety_factor", 0)
        esf = em.get("fea_safety_factor", 0)
        et  = em.get("n_teeth", em.get("num_teeth", 0))
        log_fn(
            f"  {name:<25} "
            f"{bs:>8.4f} {es:>8.4f} {ds:>+6.1f}%  "
            f"{bm_val:>8.0f} {em_val:>8.0f} {dm:>+6.1f}%  "
            f"{bb:>6.1f}mm {eb:>6.1f}mm {int(et):>7}t  "
            f"{bsf:>6.2f} {esf:>6.2f}"
        )
    log_fn("=" * 130)


def main():
    parser = argparse.ArgumentParser(description="Weapon designer evaluation suite")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Directory for evaluation outputs (default: eval_results)")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated list of case names to run (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: reduced iterations for testing (~5 min total)")
    parser.add_argument(
        "--mode", type=str, default="baseline",
        choices=["baseline", "enhanced", "compare", "functional"],
        help=(
            "baseline    = original pipeline only (default); "
            "enhanced    = improved pipeline only (FEA-in-loop, superellipse cutouts, GIF export); "
            "compare     = run both and print side-by-side table; "
            "functional  = low-dimensional functional optimizer (Stage A + B)"
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select cases
    if args.cases:
        case_names = [c.strip() for c in args.cases.split(",")]
        cases = {k: v for k, v in CASES.items() if k in case_names}
        if not cases:
            print(f"No matching cases. Available: {', '.join(CASES.keys())}")
            sys.exit(1)
    else:
        cases = CASES

    # Quick mode: reduce iterations
    if args.quick:
        for cfg in cases.values():
            cfg.optimization.max_iterations = 15
            cfg.optimization.population_size = 15
            cfg.optimization.fea_interval = 3   # still produce a few frames

    run_baseline   = args.mode in ("baseline", "compare")
    run_enhanced   = args.mode in ("enhanced", "compare")
    run_functional = args.mode == "functional"

    # Open log file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = output_dir / f"eval_{timestamp}.log"
    log_file = open(log_path, "w")

    def _log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    _log(f"Evaluation started: {len(cases)} cases, mode={args.mode}")
    _log(f"Output directory: {output_dir}")
    _log(f"Log file: {log_path}")
    _log(f"Quick mode: {args.quick}")
    _log("")

    baseline_results: list[dict] = []
    enhanced_results: list[dict] = []
    total_start = time.time()

    for i, (name, cfg) in enumerate(cases.items(), 1):
        _log(f"{'='*60}")
        _log(f"Case {i}/{len(cases)}: {name}  [mode={args.mode}]")
        _log(f"{'='*60}")

        # ── Baseline ─────────────────────────────────────────────────────
        if run_baseline:
            case_start = time.time()
            try:
                result = run_case_with_logging(name, cfg, output_dir, log_file)
                result["status"] = "success"
                result["elapsed_s"] = round(time.time() - case_start, 1)
            except Exception as e:
                _log(f"BASELINE FAILED: {e}")
                import traceback
                _log(traceback.format_exc())
                result = {"case_name": name, "weapon_style": cfg.weapon_style,
                          "status": "failed", "error": str(e)}
                result["elapsed_s"] = round(time.time() - case_start, 1)
            baseline_results.append(result)

        # ── Enhanced ─────────────────────────────────────────────────────
        if run_enhanced:
            case_start = time.time()
            try:
                eresult = run_case_enhanced(name, cfg, output_dir, log_file)
                eresult["status"] = "success"
                eresult["elapsed_s"] = round(time.time() - case_start, 1)
            except Exception as e:
                _log(f"ENHANCED FAILED: {e}")
                import traceback
                _log(traceback.format_exc())
                eresult = {"case_name": name + "_enhanced", "weapon_style": cfg.weapon_style,
                           "status": "failed", "error": str(e)}
                eresult["elapsed_s"] = round(time.time() - case_start, 1)
            enhanced_results.append(eresult)

        # ── Functional ───────────────────────────────────────────────────
        if run_functional:
            from weapon_designer.optimizer_functional import optimize_functional
            from weapon_designer.exporter import export_dxf
            import copy
            case_start = time.time()
            try:
                fcfg = copy.deepcopy(cfg)
                fcfg.optimization.evaluation_mode = "enhanced"
                func_case_dir = output_dir / (name + "_functional")
                func_case_dir.mkdir(parents=True, exist_ok=True)

                def _flog(msg):
                    ts = datetime.now().strftime("%H:%M:%S")
                    line = f"[{ts}] [{name}+func] {msg}"
                    print(line); log_file.write(line + "\n"); log_file.flush()

                fresult = optimize_functional(fcfg, func_case_dir, stage_b=True, log_fn=_flog)
                dxf_path = str(func_case_dir / f"{name}_functional.dxf")
                export_dxf(fresult["weapon_polygon"], dxf_path, fcfg)

                stats = {
                    "case_name":         name + "_functional",
                    "weapon_style":      cfg.weapon_style,
                    "evaluation_mode":   "functional",
                    "final_score":       round(fresult["score"], 6),
                    "constraint_penalty": round(fresult["penalty"], 6),
                    "metrics":           {k: round(v, 6) if isinstance(v, float) else v
                                          for k, v in fresult["metrics"].items()
                                          if not k.startswith("_")},
                    "functional_params": fresult["functional_params"].tolist(),
                    "convergence_stage_a": fresult["convergence_stage_a"],
                    "convergence_stage_b": fresult["convergence_stage_b"],
                    "elapsed_s":         round(fresult["elapsed_s"], 1),
                    "status":            "success",
                }
                stats_path = func_case_dir / f"{name}_functional_stats.json"
                with open(stats_path, "w") as sf:
                    json.dump(stats, sf, indent=2)
                _flog(f"Functional done: score={fresult['score']:.4f}  DXF={dxf_path}")
            except Exception as e:
                _log(f"FUNCTIONAL FAILED: {e}")
                import traceback
                _log(traceback.format_exc())

        elapsed_total = time.time() - total_start
        remaining = len(cases) - i
        if i > 0 and remaining > 0:
            avg_time = elapsed_total / i
            _log(f"Progress: {i}/{len(cases)} done, "
                 f"elapsed={elapsed_total/60:.0f}min, "
                 f"est. remaining={avg_time*remaining/60:.0f}min")
        _log("")

    # ── Baseline summary table ────────────────────────────────────────────
    if run_baseline:
        _log(f"{'='*60}")
        _log("BASELINE SUMMARY")
        _log(f"{'='*60}")
        _log(f"Total time: {(time.time() - total_start)/60:.1f} minutes")
        _log("")
        _log(f"{'Case':<25} {'Style':<12} {'Score':>7} {'Mass':>8} {'MOI':>10} "
             f"{'Energy':>8} {'Bite':>7} {'Struct':>7} {'CoM':>7} {'FEA SF':>7} {'Status'}")
        _log("-" * 120)

        for r in baseline_results:
            if r["status"] == "success":
                m = r["metrics"]
                _log(f"{r['case_name']:<25} {r['weapon_style']:<12} "
                     f"{r['final_score']:>7.4f} "
                     f"{m['mass_kg']:>7.3f}kg "
                     f"{m['moi_kg_mm2']:>9.1f} "
                     f"{m['energy_joules']:>7.1f}J "
                     f"{m['bite_mm']:>6.1f}mm "
                     f"{m['structural_integrity']:>7.4f} "
                     f"{m['com_offset_mm']:>6.2f}mm "
                     f"{m.get('fea_safety_factor', 0):>7.1f} "
                     f"OK ({r['elapsed_s']:.0f}s)")
            else:
                _log(f"{r['case_name']:<25} {'':12} {'':>7} {'':>8} {'':>10} "
                     f"{'':>8} {'':>7} {'':>7} {'':>7} {'':>7} "
                     f"FAILED ({r['elapsed_s']:.0f}s)")

    # ── Enhanced summary table ────────────────────────────────────────────
    if run_enhanced:
        _log("")
        _log(f"{'='*60}")
        _log("ENHANCED SUMMARY")
        _log(f"{'='*60}")
        _log(f"{'Case':<32} {'Score':>7} {'MOI':>10} {'Energy':>8} "
             f"{'Bite':>7} {'Teeth':>6} {'SF':>6} {'GIF1':>5} {'GIF2':>5}")
        _log("-" * 100)

        for r in enhanced_results:
            if r.get("status") == "success":
                m = r["metrics"]
                g1 = "Y" if r.get("gif_phase1") else "N"
                g2 = "Y" if r.get("gif_phase2") else "N"
                _log(f"{r['case_name']:<32} "
                     f"{r['final_score']:>7.4f} "
                     f"{m['moi_kg_mm2']:>9.1f} "
                     f"{m['energy_joules']:>7.1f}J "
                     f"{m['bite_mm']:>6.1f}mm "
                     f"{int(m.get('n_teeth', m.get('num_teeth', 0))):>6} "
                     f"{m.get('fea_safety_factor', 0):>6.2f} "
                     f"{g1:>5} {g2:>5}  OK ({r['elapsed_s']:.0f}s)")
            else:
                _log(f"{r['case_name']:<32} FAILED ({r.get('elapsed_s', 0):.0f}s)")

    # ── Comparison table (mode=compare) ──────────────────────────────────
    if run_baseline and run_enhanced:
        _print_comparison_table(baseline_results, enhanced_results, _log)

    # Save summary JSON
    all_results = baseline_results + enhanced_results
    summary_path = output_dir / f"summary_{timestamp}.json"
    summary = {
        "timestamp":       timestamp,
        "mode":            args.mode,
        "total_elapsed_s": round(time.time() - total_start, 1),
        "n_cases":         len(cases),
        "baseline": [{
            "name":       r["case_name"],
            "status":     r["status"],
            "elapsed_s":  r.get("elapsed_s"),
            "score":      r.get("final_score"),
            "metrics":    r.get("metrics"),
        } for r in baseline_results],
        "enhanced": [{
            "name":       r["case_name"],
            "status":     r.get("status"),
            "elapsed_s":  r.get("elapsed_s"),
            "score":      r.get("final_score"),
            "metrics":    r.get("metrics"),
            "gif_phase1": r.get("gif_phase1"),
            "gif_phase2": r.get("gif_phase2"),
        } for r in enhanced_results],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _log(f"\nSummary saved to {summary_path}")
    _log(f"Log saved to {log_path}")

    log_file.close()


if __name__ == "__main__":
    main()
