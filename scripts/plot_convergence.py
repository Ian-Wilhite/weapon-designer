"""Plot score vs. wall-clock time (and step) from one or more run directories.

Reads ``output_stats.json`` (or any JSON with ``convergence_p1``/``convergence_p2``
arrays) from each run directory, then produces:

  • score_vs_time.png  — best-so-far score vs. elapsed seconds
  • score_vs_step.png  — best-so-far score vs. DE step number

Usage
-----
    python3 scripts/plot_convergence.py runs/my_run/
    python3 scripts/plot_convergence.py runs/run_a/ runs/run_b/ --labels "A,B"
    python3 scripts/plot_convergence.py runs/sweep/ --glob "*/output_stats.json"
    python3 scripts/plot_convergence.py runs/my_run/ --out plots/

All phases (P1, P2, P2-topo) are plotted on a shared time axis where P2
continues from where P1 ends.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_history(stats_path: Path) -> tuple[list[dict], list[dict]]:
    """Return (p1_history, p2_history) from a stats JSON file."""
    with open(stats_path) as f:
        data = json.load(f)
    p1 = data.get("convergence_p1") or []
    p2 = data.get("convergence_p2") or []
    return p1, p2


def _best_so_far(history: list[dict], key: str = "score") -> list[float]:
    """Cumulative maximum of `key` across the history list."""
    best = -float("inf")
    result = []
    for entry in history:
        v = entry.get(key, -float("inf"))
        best = max(best, v)
        result.append(best)
    return result


def _find_stats_files(roots: list[Path]) -> list[Path]:
    """Recursively find output_stats.json in given root directories."""
    found = []
    for root in roots:
        if root.is_file():
            found.append(root)
        else:
            found.extend(sorted(root.rglob("output_stats.json")))
    return found


def plot_all(
    stats_files: list[Path],
    labels: list[str] | None,
    out_dir: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    if not stats_files:
        print("No stats files found.", file=sys.stderr)
        sys.exit(1)

    if labels is None:
        labels = [p.parent.name or str(p) for p in stats_files]
    while len(labels) < len(stats_files):
        labels.append(str(stats_files[len(labels)]))

    fig_t, ax_t = plt.subplots(figsize=(10, 5))
    fig_s, ax_s = plt.subplots(figsize=(10, 5))

    for sf, label in zip(stats_files, labels):
        try:
            p1, p2 = _load_history(sf)
        except Exception as exc:
            print(f"Skipping {sf}: {exc}", file=sys.stderr)
            continue

        # ── Combine phases, offsetting time/step for P2 ──────────────────
        all_entries: list[dict] = []
        t_offset = 0.0
        s_offset = 0

        for phase_entries in (p1, p2):
            if not phase_entries:
                continue
            for entry in phase_entries:
                all_entries.append({
                    "step":      entry.get("step", 0) + s_offset,
                    "elapsed_s": entry.get("elapsed_s", 0.0) + t_offset,
                    "score":     entry.get("score", 0.0),
                })
            if phase_entries:
                t_offset += phase_entries[-1].get("elapsed_s", 0.0)
                s_offset += phase_entries[-1].get("step", 0) + 1

        if not all_entries:
            print(f"No history entries in {sf}", file=sys.stderr)
            continue

        steps     = [e["step"] for e in all_entries]
        elapsed   = [e["elapsed_s"] for e in all_entries]
        scores    = [e["score"] for e in all_entries]
        best_t    = _best_so_far([{"score": s} for s in scores])

        # Plot P1/P2 boundary as a vertical line
        p2_start_step = 0
        p2_start_t    = 0.0
        if p1 and p2:
            p2_start_step = p1[-1].get("step", 0) + 1
            p2_start_t    = p1[-1].get("elapsed_s", 0.0)

        color = next(iter(plt.rcParams["axes.prop_cycle"]))["color"]
        ax_t.plot(elapsed, best_t, label=label)
        ax_s.plot(steps,   best_t, label=label)

        if p1 and p2:
            ax_t.axvline(p2_start_t, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
            ax_s.axvline(p2_start_step, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

    for ax, xlabel, fig, fname in [
        (ax_t, "Elapsed time (s)", fig_t, "score_vs_time.png"),
        (ax_s, "DE step",          fig_s, "score_vs_step.png"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Best-so-far score")
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Optimizer Convergence")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
        if show:
            plt.show()
        plt.close(fig)
        del fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path,
                        help="Run directory/directories or direct paths to output_stats.json")
    parser.add_argument("--labels", default=None,
                        help="Comma-separated labels for each run (default: directory names)")
    parser.add_argument("--out", type=Path, default=Path("."),
                        help="Output directory for PNGs (default: .)")
    parser.add_argument("--show", action="store_true", help="Show interactive plot")
    args = parser.parse_args()

    stats_files = _find_stats_files(args.paths)
    if not stats_files:
        print(f"No output_stats.json found under: {args.paths}", file=sys.stderr)
        sys.exit(1)

    labels = [s.strip() for s in args.labels.split(",")] if args.labels else None
    args.out.mkdir(parents=True, exist_ok=True)
    plot_all(stats_files, labels, args.out, args.show)


if __name__ == "__main__":
    main()
