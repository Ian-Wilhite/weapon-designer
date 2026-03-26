#!/usr/bin/env python3
"""Pareto runtime-vs-optimality scatter and bar-chart from compare_baselines results.

Usage
-----
    python3 scripts/plot_pareto.py [--input-dir DIR] [--output PATH]

Loads all ``*_results.json`` files from ``eval_results/compare/`` (or the
directory specified by ``--input-dir``), then produces a two-panel figure:

Panel 1 — Scatter:  x = elapsed_s, y = score
           colour = config_fingerprint (algorithm variant),
           marker  = case_name

Panel 2 — Bar chart: time_to_07 across configs, grouped by case
           (score ≥ 0.7 is a common "good design" threshold)

Saved to ``eval_results/pareto_runtime_vs_score.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_results(input_dir: Path) -> list[dict]:
    """Load all *_results.json files from the directory."""
    results = []
    for path in sorted(input_dir.glob("*_results.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            # Flatten each run entry
            for run in data.get("runs", []):
                run["_source_file"] = path.name
                results.append(run)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}", file=sys.stderr)
    return results


def _unique_values(runs: list[dict], key: str) -> list:
    seen: list = []
    for r in runs:
        v = r.get(key, "")
        if v not in seen:
            seen.append(v)
    return seen


def plot_pareto(input_dir: Path, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not available — skipping plot_pareto", file=sys.stderr)
        return

    runs = _load_results(input_dir)
    if not runs:
        print(f"No *_results.json files found in {input_dir}", file=sys.stderr)
        return

    # Filter to successful enhanced runs with valid scores
    enhanced = [
        r for r in runs
        if r.get("mode") == "enhanced"
        and r.get("status") == "success"
        and isinstance(r.get("score"), (int, float))
        and not np.isnan(float(r["score"]))
    ]
    if not enhanced:
        print("No successful enhanced runs found — skipping plot", file=sys.stderr)
        return

    fingerprints = _unique_values(enhanced, "config_fingerprint")
    case_names   = _unique_values(enhanced, "case_name")

    cmap = plt.cm.get_cmap("tab10", max(len(fingerprints), 1))
    marker_cycle = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "+"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel 1: Scatter — runtime vs score ──────────────────────────────
    ax1 = axes[0]
    for i, fp in enumerate(fingerprints):
        fp_runs = [r for r in enhanced if r.get("config_fingerprint") == fp]
        for j, case in enumerate(case_names):
            pts = [r for r in fp_runs if r.get("case_name") == case]
            if not pts:
                continue
            xs = [float(r.get("elapsed_s", 0.0)) for r in pts]
            ys = [float(r.get("score", 0.0)) for r in pts]
            mk = marker_cycle[j % len(marker_cycle)]
            ax1.scatter(xs, ys, color=cmap(i), marker=mk, s=60, alpha=0.8,
                        label=f"{fp}/{case[:10]}")

    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("Score")
    ax1.set_title("Runtime vs. Score by Config & Case")
    ax1.grid(True, alpha=0.3)

    # Legend: colour = fingerprint, marker = case
    legend_fp   = [Line2D([0], [0], color=cmap(i), marker="o", lw=0,
                           label=f"config {fp[:6]}")
                   for i, fp in enumerate(fingerprints)]
    legend_case = [Line2D([0], [0], color="gray",
                           marker=marker_cycle[j % len(marker_cycle)], lw=0,
                           label=case[:12])
                   for j, case in enumerate(case_names)]
    ax1.legend(handles=legend_fp + legend_case, fontsize=7, ncol=2,
               loc="lower right")

    # ── Panel 2: Bar — time_to_07 by config×case ─────────────────────────
    ax2 = axes[1]
    n_fp   = len(fingerprints)
    n_case = len(case_names)
    bar_width = 0.8 / max(n_fp, 1)
    x_base = np.arange(n_case)

    for i, fp in enumerate(fingerprints):
        fp_runs = [r for r in enhanced if r.get("config_fingerprint") == fp]
        t07_vals: list[float] = []
        for case in case_names:
            pts = [r for r in fp_runs if r.get("case_name") == case]
            if not pts:
                t07_vals.append(float("nan"))
                continue
            t07s = [float(r.get("time_to_07", float("nan"))) for r in pts]
            t07s = [v for v in t07s if not np.isnan(v)]
            t07_vals.append(float(np.mean(t07s)) if t07s else float("nan"))

        offset = (i - n_fp / 2.0 + 0.5) * bar_width
        bar_heights = [v if not np.isnan(v) else 0.0 for v in t07_vals]
        bar_alphas  = [0.85 if not np.isnan(v) else 0.15 for v in t07_vals]
        bars = ax2.bar(x_base + offset, bar_heights, width=bar_width * 0.9,
                       color=cmap(i), alpha=0.75,
                       label=f"config {fp[:6]}")

    ax2.set_xticks(x_base)
    ax2.set_xticklabels([c[:12] for c in case_names], rotation=20, ha="right",
                        fontsize=8)
    ax2.set_ylabel("Time to score ≥ 0.7 (s)  [nan = never reached]")
    ax2.set_title("Time-to-Threshold (score ≥ 0.7) by Config & Case")
    ax2.legend(fontsize=7)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Pareto plot saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Pareto runtime-vs-score from compare results")
    parser.add_argument("--input-dir", default="eval_results/compare",
                        help="Directory containing *_results.json files")
    parser.add_argument("--output", default="eval_results/pareto_runtime_vs_score.png",
                        help="Output PNG path")
    args = parser.parse_args()

    input_dir   = Path(args.input_dir)
    output_path = Path(args.output)
    plot_pareto(input_dir, output_path)


if __name__ == "__main__":
    main()
