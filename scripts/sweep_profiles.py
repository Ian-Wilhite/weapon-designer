#!/usr/bin/env python3
"""Profile-family sweep: compare bspline / bezier / catmull_rom convergence.

Runs the enhanced optimizer with each spline profile type — plus the original
Fourier baseline — across a curated set of weapon cases.  Saves per-run GIFs,
convergence JSON, and generates a suite of comparison charts.

Usage
─────
    .venv/bin/python sweep_profiles.py [options]

Options
    --cases   C1,C2,...   Case names to run  (default: featherweight_disk,
                          compact_bar,eggbeater_3blade,heavyweight_disk)
    --methods M1,M2,...   Methods to run     (default: all four)
                          Choices: fourier_baseline bspline bezier catmull_rom
    --iterations N        max_iterations per phase group  (default: 25)
    --popsize N           DE population size              (default: 15)
    --fea-interval N      FEA frame interval for GIFs     (default: 5)
    --output-dir DIR      Root output directory           (default: profile_sweep)
    --resume              Skip runs with existing run_result.json
    --charts-only         Skip all optimisation; regenerate charts from saved JSONs
    --no-baseline         Skip fourier_baseline runs (faster)

Output layout
─────────────
    profile_sweep/
        results.json                      updated after every run
        charts/
            convergence_<case>.png        score/MOI/bite vs step, one line/method
            final_scores.png              grouped bar chart
            score_heatmap.png             cases × methods heatmap
            metric_scatter.png            MOI vs bite scatter
        <case>_<method>/
            run_result.json               per-run summary + convergence history
            convergence_phase1.gif
            convergence_phase2.gif
            frames_p1/  frames_p2/
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weapon_designer.config import (
    WeaponConfig, Material, Mounting, Envelope,
    OptimizationParams, OptimizationWeights, OutputParams,
)


# ---------------------------------------------------------------------------
# Cases — curated for good mass/envelope matching
# ---------------------------------------------------------------------------

def _make_cases() -> dict[str, WeaponConfig]:
    return {
        "featherweight_disk": WeaponConfig(
            material=Material("S7_Tool_Steel", 7750, 1600, 56),
            weapon_style="disk",
            sheet_thickness_mm=6,
            # Solid disk at r=80mm, 6mm sheet ≈ 0.93 kg; budget set to 0.75 kg
            # so mass normalisation has room to shrink holes inward.
            weight_budget_kg=0.75,
            rpm=12000,
            mounting=Mounting(12.0, 25, 3, 4.0),
            envelope=Envelope(max_radius_mm=80),
            optimization=OptimizationParams(
                weights=OptimizationWeights(0.35, 0.10, 0.20, 0.10, 0.10, 0.15),
                num_fourier_terms=4, num_cutout_pairs=2,
            ),
        ),
        "compact_bar": WeaponConfig(
            material=Material("AR500", 7850, 1400, 50),
            weapon_style="bar",
            sheet_thickness_mm=10,
            # Solid bar 280×55mm ≈ 1.21 kg; budget set to 0.9 kg (~75% fill)
            weight_budget_kg=0.9,
            rpm=9000,
            mounting=Mounting(15.0, 30, 4, 5.0),
            envelope=Envelope(max_radius_mm=150, max_length_mm=280, max_width_mm=55),
            optimization=OptimizationParams(
                weights=OptimizationWeights(0.30, 0.15, 0.20, 0.10, 0.10, 0.15),
                num_fourier_terms=3, num_cutout_pairs=2,
            ),
        ),
        "eggbeater_3blade": WeaponConfig(
            material=Material("AR500", 7850, 1400, 50),
            weapon_style="eggbeater",
            sheet_thickness_mm=8,
            # Solid disk at r=130mm ≈ 3.33 kg; budget set to 2.5 kg (~75% fill)
            weight_budget_kg=2.5,
            rpm=12000,
            mounting=Mounting(20.0, 38, 3, 5.5),
            envelope=Envelope(max_radius_mm=130),
            optimization=OptimizationParams(
                weights=OptimizationWeights(0.25, 0.20, 0.20, 0.10, 0.10, 0.15),
                num_fourier_terms=4, num_cutout_pairs=2,
            ),
        ),
        "heavyweight_disk": WeaponConfig(
            material=Material("AR500", 7850, 1400, 50),
            weapon_style="disk",
            sheet_thickness_mm=12,
            weight_budget_kg=7.0,
            rpm=8000,
            mounting=Mounting(25.4, 60, 6, 8.0),
            envelope=Envelope(max_radius_mm=180),
            optimization=OptimizationParams(
                weights=OptimizationWeights(0.30, 0.15, 0.20, 0.10, 0.10, 0.15),
                num_fourier_terms=5, num_cutout_pairs=3,
            ),
        ),
    }


SWEEP_CASES = _make_cases()

# ---------------------------------------------------------------------------
# Method registry and visual style
# ---------------------------------------------------------------------------

METHODS = {
    "fourier_baseline": {"runner": "baseline",  "profile_type": None},
    "bspline":          {"runner": "enhanced",  "profile_type": "bspline"},
    "bezier":           {"runner": "enhanced",  "profile_type": "bezier"},
    "catmull_rom":      {"runner": "enhanced",  "profile_type": "catmull_rom"},
}

METHOD_STYLES: dict[str, dict] = {
    "fourier_baseline": {"color": "#888888", "ls": "--", "lw": 1.8, "label": "Fourier (baseline)", "marker": "s"},
    "bspline":          {"color": "#2196F3", "ls": "-",  "lw": 2.0, "label": "B-Spline",           "marker": "o"},
    "bezier":           {"color": "#FF5722", "ls": "-",  "lw": 2.0, "label": "Bézier",              "marker": "^"},
    "catmull_rom":      {"color": "#4CAF50", "ls": "-",  "lw": 2.0, "label": "Catmull-Rom",         "marker": "D"},
}

CASE_LABELS = {
    "featherweight_disk": "Featherweight Disk",
    "compact_bar":        "Compact Bar",
    "eggbeater_3blade":   "Eggbeater (3-blade)",
    "heavyweight_disk":   "Heavyweight Disk",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    case_name:     str
    method:        str
    profile_type:  str | None
    status:        str           # "success" | "failed" | "skipped"
    elapsed_s:     float
    final_score:   float
    final_metrics: dict
    convergence:   list[dict]    # normalised: [{cum_step, phase, score, moi, bite, sf, mass_kg}]
    output_dir:    str
    gif_phase1:    str | None
    gif_phase2:    str | None
    error:         str | None = None


def _to_json(r: RunResult) -> dict:
    d = asdict(r)
    return d


def save_run_result(r: RunResult, run_dir: Path) -> None:
    (run_dir / "run_result.json").write_text(json.dumps(_to_json(r), indent=2))


def load_run_result(run_dir: Path) -> RunResult | None:
    p = run_dir / "run_result.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return RunResult(**d)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Convergence normalisation
# ---------------------------------------------------------------------------

def _normalise_baseline_history(
    p1: list[dict], p2: list[dict]
) -> list[dict]:
    """Convert baseline phase1/phase2 history to common format."""
    out = []
    cum = 0
    for entry in p1:
        cum += 1
        out.append({
            "cum_step":  cum,
            "phase":     "P1",
            "step":      entry["step"],
            "score":     entry["score"],
            "moi":       entry.get("moi_kg_mm2", 0.0),
            "bite":      entry.get("bite_mm", 0.0),
            "sf":        None,   # no FEA in baseline
            "mass_kg":   entry.get("mass_kg", 0.0),
            "elapsed_s": entry.get("elapsed_s", 0.0),
        })
    for entry in p2:
        cum += 1
        out.append({
            "cum_step":  cum,
            "phase":     "P2",
            "step":      entry["step"],
            "score":     entry["score"],
            "moi":       entry.get("moi_kg_mm2", 0.0),
            "bite":      entry.get("bite_mm", 0.0),
            "sf":        None,
            "mass_kg":   entry.get("mass_kg", 0.0),
            "elapsed_s": entry.get("elapsed_s", 0.0),
        })
    return out


def _normalise_enhanced_history(
    p1: list[dict], p2: list[dict]
) -> list[dict]:
    """Convert enhanced phase1/phase2 history to common format."""
    out = []
    cum = 0
    for phase_label, entries in [("P1", p1), ("P2", p2)]:
        for entry in entries:
            cum += 1
            out.append({
                "cum_step":  cum,
                "phase":     phase_label,
                "step":      entry["step"],
                "score":     entry["score"],
                "moi":       entry.get("moi_kg_mm2", 0.0),
                "bite":      entry.get("bite_mm", 0.0),
                "sf":        entry.get("fea_sf"),
                "mass_kg":   entry.get("mass_kg", 0.0),
                "elapsed_s": entry.get("elapsed_s", 0.0),
            })
    return out


# ---------------------------------------------------------------------------
# Baseline runner  (Fourier profile + Fourier cutouts, no FEA in loop)
# ---------------------------------------------------------------------------

def run_baseline(
    case_name: str,
    cfg: WeaponConfig,
    run_dir: Path,
    iterations: int,
    popsize: int,
    log_fn,
) -> RunResult:
    from weapon_designer.optimizer import (
        _get_profile_bounds, _get_cutout_bounds,
        _profile_objective, _cutout_objective,
    )
    from weapon_designer.parametric import build_weapon_polygon, _cutout_stride
    from weapon_designer.geometry import assemble_weapon
    from weapon_designer.objectives import compute_metrics, weighted_score
    from weapon_designer.constraints import validate_geometry, constraint_penalty
    from scipy.optimize import differential_evolution

    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    profile_bounds = _get_profile_bounds(cfg)
    cutout_bounds  = _get_cutout_bounds(cfg)
    n_profile      = len(profile_bounds)
    cutout_stride  = _cutout_stride(cfg)
    workers        = max(1, os.cpu_count() or 1)

    phase1_history: list[dict] = []
    phase2_history: list[dict] = []

    # ── Phase 1 ──────────────────────────────────────────────────────────
    class _P1Cb:
        def __init__(self):
            self.step = 0
            self.t0   = time.time()

        def __call__(self, xk, conv):
            self.step += 1
            try:
                C = cfg.optimization.num_cutout_pairs
                x_full = np.concatenate([xk, np.zeros(C * cutout_stride)])
                outer, _, cutout_polys = build_weapon_polygon(x_full, cfg)
                weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
                weapon = validate_geometry(weapon)
                metrics = compute_metrics(weapon, cfg)
                score   = weighted_score(metrics, cfg)
                phase1_history.append({
                    "step":      self.step,
                    "elapsed_s": round(time.time() - self.t0, 1),
                    "score":     round(score, 6),
                    "moi_kg_mm2": round(metrics["moi_kg_mm2"], 2),
                    "bite_mm":   round(metrics["bite_mm"], 2),
                    "structural": round(metrics["structural_integrity"], 4),
                    "mass_kg":   round(metrics["mass_kg"], 4),
                })
                if self.step % 5 == 0 or self.step <= 2:
                    log_fn(f"  [baseline P1 {self.step:3d}] score={score:.4f} "
                           f"MOI={metrics['moi_kg_mm2']:.0f} bite={metrics['bite_mm']:.1f}mm")
            except Exception:
                pass

    p1_cb = _P1Cb()
    phase1_iters = max(5, int(iterations * 0.70))
    result1 = differential_evolution(
        _profile_objective, bounds=profile_bounds,
        args=(cfg,), maxiter=phase1_iters, popsize=popsize,
        seed=42, tol=1e-6, mutation=(0.5, 1.5), recombination=0.8,
        workers=workers, updating="deferred", disp=False,
        init="latinhypercube", callback=p1_cb,
    )
    best_profile = result1.x
    log_fn(f"  baseline P1 done: score={-result1.fun:.4f}, evals={result1.nfev}")

    # ── Phase 2 ──────────────────────────────────────────────────────────
    if cfg.optimization.num_cutout_pairs > 0 and len(cutout_bounds) > 0:
        class _P2Cb:
            def __init__(self):
                self.step = 0
                self.t0   = time.time()

            def __call__(self, xk, conv):
                self.step += 1
                try:
                    x_full = np.concatenate([best_profile, xk])
                    outer, _, cutout_polys = build_weapon_polygon(x_full, cfg)
                    weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
                    weapon = validate_geometry(weapon)
                    metrics = compute_metrics(weapon, cfg)
                    score   = weighted_score(metrics, cfg)
                    phase2_history.append({
                        "step":      self.step,
                        "elapsed_s": round(time.time() - self.t0, 1),
                        "score":     round(score, 6),
                        "moi_kg_mm2": round(metrics["moi_kg_mm2"], 2),
                        "bite_mm":   round(metrics["bite_mm"], 2),
                        "structural": round(metrics["structural_integrity"], 4),
                        "mass_kg":   round(metrics["mass_kg"], 4),
                    })
                    if self.step % 5 == 0 or self.step <= 2:
                        log_fn(f"  [baseline P2 {self.step:3d}] score={score:.4f}")
                except Exception:
                    pass

        p2_cb = _P2Cb()
        phase2_iters = max(3, int(iterations * 0.30))
        result2 = differential_evolution(
            _cutout_objective, bounds=cutout_bounds,
            args=(best_profile, cfg), maxiter=phase2_iters, popsize=popsize,
            seed=42, tol=1e-6, mutation=(0.5, 1.5), recombination=0.8,
            workers=workers, updating="deferred", disp=False,
            callback=p2_cb,
        )
        best_cutouts = result2.x
        log_fn(f"  baseline P2 done: score={-result2.fun:.4f}")
    else:
        best_cutouts = np.zeros(0)

    # ── Final eval ────────────────────────────────────────────────────────
    try:
        x_best = np.concatenate([best_profile, best_cutouts])
        outer, _, cutout_polys = build_weapon_polygon(x_best, cfg)
        weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
        weapon = validate_geometry(weapon)
        metrics = compute_metrics(weapon, cfg)
        score   = weighted_score(metrics, cfg)
    except Exception as e:
        return RunResult(
            case_name=case_name, method="fourier_baseline", profile_type=None,
            status="failed", elapsed_s=time.time()-t0,
            final_score=0.0, final_metrics={}, convergence=[],
            output_dir=str(run_dir), gif_phase1=None, gif_phase2=None,
            error=str(e),
        )

    convergence = _normalise_baseline_history(phase1_history, phase2_history)

    return RunResult(
        case_name=case_name, method="fourier_baseline", profile_type="fourier",
        status="success", elapsed_s=round(time.time()-t0, 1),
        final_score=round(score, 6),
        final_metrics={k: round(v, 6) if isinstance(v, float) else v
                       for k, v in metrics.items()},
        convergence=convergence,
        output_dir=str(run_dir), gif_phase1=None, gif_phase2=None,
    )


# ---------------------------------------------------------------------------
# Enhanced runner  (B-spline / Bézier / Catmull-Rom profile + FEA in loop)
# ---------------------------------------------------------------------------

def run_enhanced(
    case_name: str,
    cfg: WeaponConfig,
    profile_type: str,
    run_dir: Path,
    iterations: int,
    popsize: int,
    fea_interval: int,
    log_fn,
) -> RunResult:
    from weapon_designer.optimizer_enhanced import optimize_enhanced

    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ecfg = copy.deepcopy(cfg)
    ecfg.optimization.max_iterations   = iterations
    ecfg.optimization.population_size  = popsize
    ecfg.optimization.evaluation_mode  = "enhanced"
    ecfg.optimization.cutout_type      = "superellipse"
    ecfg.optimization.profile_type     = profile_type
    ecfg.optimization.fea_interval     = fea_interval

    try:
        result = optimize_enhanced(ecfg, run_dir, verbose=True)
    except Exception as e:
        return RunResult(
            case_name=case_name, method=profile_type, profile_type=profile_type,
            status="failed", elapsed_s=round(time.time()-t0, 1),
            final_score=0.0, final_metrics={}, convergence=[],
            output_dir=str(run_dir), gif_phase1=None, gif_phase2=None,
            error=traceback.format_exc(),
        )

    metrics   = result["metrics"]
    score     = result["score"]
    p1_hist   = result.get("convergence_p1", [])
    p2_hist   = result.get("convergence_p2", [])
    convergence = _normalise_enhanced_history(p1_hist, p2_hist)

    g1 = str(result["gif_phase1"]) if result["gif_phase1"] else None
    g2 = str(result["gif_phase2"]) if result["gif_phase2"] else None

    return RunResult(
        case_name=case_name, method=profile_type, profile_type=profile_type,
        status="success", elapsed_s=round(time.time()-t0, 1),
        final_score=round(score, 6),
        final_metrics={k: round(v, 6) if isinstance(v, float) else v
                       for k, v in metrics.items()
                       if not k.startswith("_")},
        convergence=convergence,
        output_dir=str(run_dir),
        gif_phase1=g1, gif_phase2=g2,
    )


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        10,
        "axes.titlesize":   11,
        "axes.labelsize":   10,
        "legend.fontsize":  9,
        "figure.dpi":       120,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "axes.spines.top":  False,
        "axes.spines.right": False,
    })
    return plt


def plot_case_convergence(
    case_name: str,
    results: list[RunResult],
    output_path: Path,
) -> None:
    """3-panel convergence chart (score, MOI, bite) per case, one line per method."""
    plt = _setup_matplotlib()
    import matplotlib.pyplot as plt2
    from matplotlib.lines import Line2D

    fig, axes = plt2.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        f"Convergence — {CASE_LABELS.get(case_name, case_name)}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    metrics_cfg = [
        ("score", "Score",      axes[0], None),
        ("moi",   "MOI (kg·mm²)", axes[1], None),
        ("bite",  "Bite (mm)",  axes[2], None),
    ]

    p1_boundary: dict[str, int] = {}  # method → last cum_step in P1

    for res in results:
        if res.status != "success" or not res.convergence:
            continue
        sty = METHOD_STYLES.get(res.method, {})
        color = sty.get("color", "black")
        ls    = sty.get("ls", "-")
        lw    = sty.get("lw", 1.5)

        conv = res.convergence
        steps = [c["cum_step"] for c in conv]

        # Track P1/P2 boundary
        p1_end = max((c["cum_step"] for c in conv if c["phase"] == "P1"), default=None)
        if p1_end:
            p1_boundary[res.method] = p1_end

        for key, ylabel, ax, _ in metrics_cfg:
            vals = [c[key] for c in conv]
            ax.plot(steps, vals, color=color, ls=ls, lw=lw, alpha=0.9)
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel(ylabel)

    # Phase boundary shading on score panel (use median of P1 ends)
    if p1_boundary:
        med_p1 = int(np.median(list(p1_boundary.values())))
        for _, _, ax, _ in metrics_cfg:
            ax.axvline(med_p1, color="#999999", ls=":", lw=1.2, alpha=0.7)
            ax.text(med_p1 + 0.3, ax.get_ylim()[1], "P2→", fontsize=7,
                    color="#666666", va="top")

    # Legend
    legend_handles = [
        Line2D([0], [0],
               color=METHOD_STYLES[m]["color"],
               ls=METHOD_STYLES[m]["ls"],
               lw=METHOD_STYLES[m]["lw"],
               label=METHOD_STYLES[m]["label"])
        for m in METHODS if any(r.method == m for r in results)
    ]
    axes[0].legend(handles=legend_handles, loc="lower right")

    plt2.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt2.close(fig)
    print(f"  chart: {output_path}")


def plot_final_scores(
    all_results: list[RunResult],
    output_path: Path,
) -> None:
    """Grouped bar chart: final score per (case × method)."""
    plt = _setup_matplotlib()
    import matplotlib.pyplot as plt2
    import matplotlib.patches as mpatches

    cases   = [c for c in SWEEP_CASES if any(r.case_name == c for r in all_results)]
    methods = [m for m in METHODS]
    n_cases = len(cases)
    n_meth  = len(methods)

    x      = np.arange(n_cases)
    width  = 0.18
    offsets = np.linspace(-(n_meth - 1) / 2, (n_meth - 1) / 2, n_meth) * width

    fig, ax = plt2.subplots(figsize=(max(10, n_cases * 2.5), 5))

    for j, method in enumerate(methods):
        scores = []
        for case in cases:
            match = [r for r in all_results if r.case_name == case and r.method == method
                     and r.status == "success"]
            scores.append(match[0].final_score if match else 0.0)

        sty = METHOD_STYLES[method]
        bars = ax.bar(
            x + offsets[j], scores, width,
            color=sty["color"], alpha=0.85,
            label=sty["label"], edgecolor="white", linewidth=0.5,
        )
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{score:.3f}", ha="center", va="bottom",
                        fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS.get(c, c) for c in cases], rotation=15, ha="right")
    ax.set_ylabel("Final weighted score")
    ax.set_title("Final Score by Case and Profile Family", fontweight="bold")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.legend(loc="upper right")

    plt2.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt2.close(fig)
    print(f"  chart: {output_path}")


def plot_score_heatmap(
    all_results: list[RunResult],
    output_path: Path,
) -> None:
    """Heatmap of final scores: rows=cases, cols=methods."""
    plt = _setup_matplotlib()
    import matplotlib.pyplot as plt2

    cases   = [c for c in SWEEP_CASES if any(r.case_name == c for r in all_results)]
    methods = list(METHODS.keys())

    data = np.zeros((len(cases), len(methods)))
    mask = np.zeros_like(data, dtype=bool)

    for i, case in enumerate(cases):
        for j, method in enumerate(methods):
            match = [r for r in all_results
                     if r.case_name == case and r.method == method and r.status == "success"]
            if match:
                data[i, j] = match[0].final_score
            else:
                mask[i, j] = True

    fig, ax = plt2.subplots(figsize=(len(methods) * 2 + 1.5, len(cases) * 0.9 + 1.5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd",
                   vmin=max(0, data[~mask].min() - 0.05) if (~mask).any() else 0,
                   vmax=data[~mask].max() + 0.02 if (~mask).any() else 1)

    # Annotate cells
    for i in range(len(cases)):
        for j in range(len(methods)):
            txt = f"{data[i,j]:.4f}" if not mask[i, j] else "—"
            color = "white" if data[i, j] > data[~mask].mean() * 1.05 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_STYLES[m]["label"] for m in methods], rotation=20, ha="right")
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([CASE_LABELS.get(c, c) for c in cases])
    ax.set_title("Final Score Heatmap (Cases × Profile Family)", fontweight="bold", pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Score", fontsize=9)

    plt2.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt2.close(fig)
    print(f"  chart: {output_path}")


def plot_metric_scatter(
    all_results: list[RunResult],
    output_path: Path,
) -> None:
    """Scatter plot: final MOI vs final bite, colored by method, marker by case."""
    plt = _setup_matplotlib()
    import matplotlib.pyplot as plt2
    from matplotlib.lines import Line2D

    CASE_MARKERS = ["o", "s", "^", "D", "v", "P", "*"]
    cases = sorted({r.case_name for r in all_results if r.status == "success"})
    case_marker = {c: CASE_MARKERS[i % len(CASE_MARKERS)] for i, c in enumerate(cases)}

    fig, ax = plt2.subplots(figsize=(9, 6))

    for res in all_results:
        if res.status != "success":
            continue
        sty  = METHOD_STYLES.get(res.method, {})
        moi  = res.final_metrics.get("moi_kg_mm2", 0.0)
        bite = res.final_metrics.get("bite_mm",    0.0)
        ax.scatter(moi, bite,
                   color=sty.get("color", "gray"),
                   marker=case_marker.get(res.case_name, "o"),
                   s=80, alpha=0.85, edgecolors="white", linewidths=0.5,
                   zorder=5)

    ax.set_xlabel("Final MOI (kg·mm²)")
    ax.set_ylabel("Final bite (mm)")
    ax.set_title("MOI vs Bite — All Runs", fontweight="bold")

    # Legend: method colors
    method_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_STYLES[m]["color"],
               markersize=10, label=METHOD_STYLES[m]["label"])
        for m in METHODS
    ]
    # Legend: case markers
    case_handles = [
        Line2D([0], [0], marker=case_marker[c], color="w", markerfacecolor="#555555",
               markersize=9, label=CASE_LABELS.get(c, c))
        for c in cases
    ]
    leg1 = ax.legend(handles=method_handles, title="Profile family",
                     loc="upper left",  fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=case_handles, title="Case",
              loc="lower right", fontsize=8)

    plt2.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt2.close(fig)
    print(f"  chart: {output_path}")


def plot_phase_comparison(
    all_results: list[RunResult],
    output_path: Path,
) -> None:
    """Multi-row convergence grid: rows=cases, cols=metrics, lines=methods."""
    plt = _setup_matplotlib()
    import matplotlib.pyplot as plt2
    from matplotlib.lines import Line2D

    cases   = [c for c in SWEEP_CASES if any(r.case_name == c and r.status == "success"
                                              for r in all_results)]
    metrics = [("score", "Score"), ("moi", "MOI (kg·mm²)"), ("bite", "Bite (mm)")]
    n_rows  = len(cases)
    n_cols  = len(metrics)

    fig, axes = plt2.subplots(n_rows, n_cols,
                               figsize=(n_cols * 4.5, n_rows * 3.2),
                               sharex=False)
    if n_rows == 1:
        axes = [axes]

    for i, case in enumerate(cases):
        row = axes[i]
        case_results = [r for r in all_results if r.case_name == case and r.status == "success"]

        for j, (key, ylabel) in enumerate(metrics):
            ax = row[j]
            for res in case_results:
                if not res.convergence:
                    continue
                sty   = METHOD_STYLES.get(res.method, {})
                steps = [c["cum_step"] for c in res.convergence]
                vals  = [c[key] for c in res.convergence]
                ax.plot(steps, vals,
                        color=sty["color"], ls=sty["ls"], lw=sty["lw"], alpha=0.9)

            if j == 0:
                ax.set_ylabel(ylabel + f"\n{CASE_LABELS.get(case, case)}", fontsize=8)
            else:
                ax.set_ylabel(ylabel, fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel("Optimizer step", fontsize=8)
            if i == 0:
                ax.set_title(ylabel, fontweight="bold", fontsize=9)

    # Shared legend at bottom
    handles = [
        Line2D([0], [0], color=METHOD_STYLES[m]["color"],
               ls=METHOD_STYLES[m]["ls"], lw=2.0, label=METHOD_STYLES[m]["label"])
        for m in METHODS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(METHODS),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Convergence Grid — All Cases × All Metrics", fontsize=13,
                 fontweight="bold", y=1.01)
    plt2.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt2.close(fig)
    print(f"  chart: {output_path}")


def generate_all_charts(all_results: list[RunResult], charts_dir: Path) -> None:
    """Generate full chart suite from completed results."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    successful = [r for r in all_results if r.status == "success"]
    if not successful:
        print("[charts] No successful results to plot.")
        return

    print(f"\n[charts] Generating charts in {charts_dir}/ ...")

    # Per-case convergence
    cases = {r.case_name for r in successful}
    for case in sorted(cases):
        case_results = [r for r in successful if r.case_name == case]
        plot_case_convergence(case, case_results,
                              charts_dir / f"convergence_{case}.png")

    # Cross-case charts
    plot_final_scores(successful,    charts_dir / "final_scores.png")
    plot_score_heatmap(successful,   charts_dir / "score_heatmap.png")
    plot_metric_scatter(successful,  charts_dir / "metric_scatter.png")
    plot_phase_comparison(successful, charts_dir / "convergence_grid.png")

    print(f"[charts] Done — {len(list(charts_dir.glob('*.png')))} charts written.")


# ---------------------------------------------------------------------------
# Summary table (stdout)
# ---------------------------------------------------------------------------

def print_summary_table(all_results: list[RunResult]) -> None:
    print("\n" + "=" * 110)
    print("SWEEP SUMMARY")
    print("=" * 110)
    hdr = (f"{'Case':<25} {'Method':<18} {'Score':>8} {'MOI':>10} "
           f"{'Bite':>7} {'SF':>7} {'Mass':>8} {'Time':>8} {'Status'}")
    print(hdr)
    print("-" * 110)
    for r in sorted(all_results, key=lambda x: (x.case_name, list(METHODS).index(x.method)
                                                 if x.method in METHODS else 99)):
        if r.status == "success":
            moi  = r.final_metrics.get("moi_kg_mm2", 0)
            bite = r.final_metrics.get("bite_mm", 0)
            sf   = r.final_metrics.get("fea_safety_factor", r.final_metrics.get("structural_integrity", 0))
            mass = r.final_metrics.get("mass_kg", 0)
            print(f"  {r.case_name:<25} {r.method:<18} {r.final_score:>8.4f} "
                  f"{moi:>9.0f} {bite:>6.1f}mm {sf:>7.2f} "
                  f"{mass:>6.3f}kg {r.elapsed_s:>7.0f}s  OK")
        else:
            print(f"  {r.case_name:<25} {r.method:<18} {'':>8} {'':>10} "
                  f"{'':>7} {'':>7} {'':>8} {r.elapsed_s:>7.0f}s  FAILED")
    print("=" * 110)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_all_results(all_results: list[RunResult], output_dir: Path) -> None:
    p = output_dir / "results.json"
    p.write_text(json.dumps([_to_json(r) for r in all_results], indent=2))


def load_all_results(output_dir: Path) -> list[RunResult]:
    p = output_dir / "results.json"
    if not p.exists():
        return []
    try:
        items = json.loads(p.read_text())
        return [RunResult(**d) for d in items]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep all profile families across weapon cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cases",       default=",".join(SWEEP_CASES.keys()),
                        help="Comma-separated case names")
    parser.add_argument("--methods",     default=",".join(METHODS.keys()),
                        help="Comma-separated method names")
    parser.add_argument("--iterations",  type=int, default=25,
                        help="max_iterations per run (default: 25)")
    parser.add_argument("--phase1-iters", type=int, default=0,
                        help="Override Phase-1 iteration count (default: 50%% of --iterations)")
    parser.add_argument("--phase2-iters", type=int, default=0,
                        help="Override Phase-2 iteration count (default: 25%% of --iterations)")
    parser.add_argument("--popsize",     type=int, default=15,
                        help="DE population size multiplier (default: 15). "
                             "Actual population = popsize × n_params.")
    parser.add_argument("--fea-interval", type=int, default=5,
                        help="FEA frame interval for GIFs (default: 5)")
    parser.add_argument("--medium",      action="store_true",
                        help="Medium-strength preset: <5 min per run "
                             "(iterations=20, popsize=6, n_bspline=8, fea_coarse=18mm, "
                             "phase1=12, phase2=6, no GIFs). "
                             "Overrides --iterations/--popsize/--fea-interval.")
    parser.add_argument("--output-dir",  default="profile_sweep",
                        help="Root output directory (default: profile_sweep)")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip runs with existing run_result.json")
    parser.add_argument("--charts-only", action="store_true",
                        help="Skip optimisation; regenerate charts from saved JSONs")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip fourier_baseline runs")
    args = parser.parse_args()

    # Apply medium preset (overrides explicit flags)
    # popsize=6, n_bspline=8, fea_coarse=18mm → actual_pop ≈ 6×8=48 per gen
    # phase1=12 gens × 48 = 576 FEA calls; phase2=6 gens × ~60 = 360 → ~<3 min
    if args.medium:
        args.iterations  = 20
        args.popsize     = 6
        args.fea_interval = 0   # disable GIF frames (saves significant time)
        if args.phase1_iters == 0:
            args.phase1_iters = 12
        if args.phase2_iters == 0:
            args.phase2_iters = 6

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"

    # Parse selections
    selected_cases   = [c.strip() for c in args.cases.split(",") if c.strip() in SWEEP_CASES]
    selected_methods = [m.strip() for m in args.methods.split(",") if m.strip() in METHODS]
    if args.no_baseline:
        selected_methods = [m for m in selected_methods if m != "fourier_baseline"]

    if not selected_cases:
        print(f"No valid cases.  Choices: {', '.join(SWEEP_CASES)}")
        sys.exit(1)
    if not selected_methods:
        print(f"No valid methods.  Choices: {', '.join(METHODS)}")
        sys.exit(1)

    preset_str = " [MEDIUM]" if args.medium else ""
    print(f"Sweep{preset_str}: {len(selected_cases)} case(s) × {len(selected_methods)} method(s) "
          f"= {len(selected_cases) * len(selected_methods)} run(s)")
    print(f"  Cases:   {', '.join(selected_cases)}")
    print(f"  Methods: {', '.join(selected_methods)}")
    print(f"  Iterations: {args.iterations}  Popsize: {args.popsize}  "
          f"Phase1: {args.phase1_iters or 'auto'}  Phase2: {args.phase2_iters or 'auto'}  "
          f"FEA interval: {args.fea_interval}")
    print(f"  Output:  {output_dir}/\n")

    # Load any existing results
    all_results: list[RunResult] = load_all_results(output_dir)

    if args.charts_only:
        if not all_results:
            print("[charts-only] No results.json found — nothing to chart.")
            sys.exit(1)
        generate_all_charts(all_results, charts_dir)
        print_summary_table(all_results)
        return

    total_runs = len(selected_cases) * len(selected_methods)
    run_idx = 0

    for case_name in selected_cases:
        cfg_base = SWEEP_CASES[case_name]

        for method in selected_methods:
            run_idx += 1
            method_info = METHODS[method]
            run_dir = output_dir / f"{case_name}_{method}"

            print(f"\n{'='*70}")
            print(f"Run {run_idx}/{total_runs}: {case_name} + {method}")
            print(f"{'='*70}")

            # Resume: skip if already done
            if args.resume and run_dir.exists():
                existing = load_run_result(run_dir)
                if existing and existing.status == "success":
                    print(f"  [resume] found successful result — skipping")
                    # Ensure it's in all_results
                    key = (case_name, method)
                    if not any(r.case_name == case_name and r.method == method
                               for r in all_results):
                        all_results.append(existing)
                    continue

            # Prepare per-run cfg copy
            cfg = copy.deepcopy(cfg_base)
            cfg.optimization.max_iterations = args.iterations
            cfg.optimization.population_size = args.popsize
            cfg.optimization.phase1_iters = args.phase1_iters
            cfg.optimization.phase2_iters = args.phase2_iters
            # Medium preset: tighter FEA mesh and fewer bspline params
            if args.medium:
                cfg.optimization.n_bspline_points = 8
                cfg.optimization.fea_coarse_spacing_mm = 18.0
                cfg.optimization.fea_fine_spacing_mm = 9.0
                cfg.optimization.convergence_patience = 8
                cfg.optimization.convergence_min_delta = 0.003

            def _log(msg: str):
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] {msg}")

            t_run_start = time.time()
            try:
                if method_info["runner"] == "baseline":
                    result = run_baseline(
                        case_name, cfg, run_dir,
                        args.iterations, args.popsize, _log,
                    )
                else:
                    result = run_enhanced(
                        case_name, cfg, method_info["profile_type"],
                        run_dir, args.iterations, args.popsize,
                        args.fea_interval, _log,
                    )
            except Exception as e:
                print(f"  UNEXPECTED ERROR: {e}")
                result = RunResult(
                    case_name=case_name, method=method,
                    profile_type=method_info.get("profile_type"),
                    status="failed", elapsed_s=round(time.time()-t_run_start, 1),
                    final_score=0.0, final_metrics={}, convergence=[],
                    output_dir=str(run_dir), gif_phase1=None, gif_phase2=None,
                    error=traceback.format_exc(),
                )

            # Save per-run and aggregate
            save_run_result(result, run_dir)
            # Replace or append in all_results
            all_results = [r for r in all_results
                           if not (r.case_name == case_name and r.method == method)]
            all_results.append(result)
            save_all_results(all_results, output_dir)

            status_str = ("OK" if result.status == "success"
                          else f"FAILED: {result.error or '?'}")
            print(f"\n  {case_name}/{method}: {status_str} "
                  f"score={result.final_score:.4f} elapsed={result.elapsed_s:.0f}s")

    # ── Generate charts after all runs ────────────────────────────────────
    print(f"\n{'='*70}")
    print("All runs complete — generating charts")
    print(f"{'='*70}")
    generate_all_charts(all_results, charts_dir)
    print_summary_table(all_results)

    successful = [r for r in all_results if r.status == "success"]
    failed     = [r for r in all_results if r.status == "failed"]
    print(f"\nDone: {len(successful)} succeeded, {len(failed)} failed.")
    if failed:
        for r in failed:
            print(f"  FAILED: {r.case_name}/{r.method}")
    print(f"Results: {output_dir}/results.json")
    print(f"Charts:  {charts_dir}/")


if __name__ == "__main__":
    main()
