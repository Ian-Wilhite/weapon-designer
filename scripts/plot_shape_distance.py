"""Plot shape distance vs. step from sidecar meta JSONs in a frames directory.

Shape distance (RMS mm difference between consecutive best radii vectors) is a
proxy for optimizer thrashing vs. converging.  A large distance means the best
solution jumped significantly; near-zero means the optimizer has stalled.

Reads ``*_meta.json`` sidecars produced by the enhanced optimizer alongside FEA
frame PNGs.  Works with both P1 and P2 frame directories.

Usage
-----
    python3 scripts/plot_shape_distance.py runs/my_run/frames_p1/
    python3 scripts/plot_shape_distance.py runs/my_run/frames_p1/ runs/my_run/frames_p2/
    python3 scripts/plot_shape_distance.py runs/my_run/ --glob "frames_*/frame_*_meta.json"
    python3 scripts/plot_shape_distance.py runs/my_run/ --from-stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_meta_files(dirs: list[Path]) -> list[dict]:
    """Load all *_meta.json files from the given directories, sorted by step."""
    records: list[dict] = []
    for d in dirs:
        if d.is_file():
            try:
                records.append(json.loads(d.read_text()))
            except Exception:
                pass
        elif d.is_dir():
            for f in sorted(d.glob("*_meta.json")):
                try:
                    records.append(json.loads(f.read_text()))
                except Exception:
                    continue
    # Sort by (phase, step) so P2 follows P1
    def sort_key(r: dict) -> tuple[int, int]:
        phase_ord = 0 if "P1" in str(r.get("phase", "")) else 1
        return (phase_ord, int(r.get("step", 0)))
    records.sort(key=sort_key)
    return records


def _load_from_stats(stats_path: Path) -> list[dict]:
    """Load convergence history from output_stats.json (inline format)."""
    with open(stats_path) as f:
        data = json.load(f)
    records = []
    for entry in (data.get("convergence_p1") or []):
        records.append(entry)
    for entry in (data.get("convergence_p2") or []):
        records.append(entry)
    return records


def plot_shape_distance(
    records: list[dict],
    out_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    steps    = [r.get("step", i)         for i, r in enumerate(records)]
    scores   = [r.get("score", 0.0)      for r in records]
    dists    = [r.get("shape_distance", None) for r in records]

    has_dist = any(d is not None for d in dists)
    dists_clean = [d if d is not None else 0.0 for d in dists]

    fig, axes = plt.subplots(
        2 if has_dist else 1, 1,
        figsize=(10, 7 if has_dist else 4),
        sharex=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # ── Score panel ───────────────────────────────────────────────────────
    ax_score = axes[0]
    ax_score.plot(steps, scores, color="steelblue", linewidth=1.0, label="score")
    # Running best
    best_scores = []
    best = -float("inf")
    for s in scores:
        best = max(best, s)
        best_scores.append(best)
    ax_score.plot(steps, best_scores, color="navy", linewidth=1.5,
                  linestyle="--", label="best-so-far")
    ax_score.set_ylabel("Score")
    ax_score.set_ylim(bottom=0.0)
    ax_score.legend(fontsize=8)
    ax_score.grid(True, alpha=0.3)
    ax_score.set_title("Optimizer Convergence & Shape Distance")

    # ── Shape distance panel ──────────────────────────────────────────────
    if has_dist:
        ax_dist = axes[1]
        ax_dist.semilogy(steps, [max(d, 1e-4) for d in dists_clean],
                         color="darkorange", linewidth=1.0)
        ax_dist.set_ylabel("Shape distance (mm, log)")
        ax_dist.set_xlabel("DE step")
        ax_dist.grid(True, alpha=0.3, which="both")

        # Mark the step where shape distance dropped below 0.5mm (converging)
        for i, (s, d) in enumerate(zip(steps, dists_clean)):
            if d < 0.5 and i > 5:
                ax_dist.axvline(s, color="green", linestyle=":", alpha=0.5)
                ax_dist.annotate("d<0.5mm", xy=(s, 0.5),
                                 fontsize=7, color="green", rotation=90,
                                 va="top", ha="right")
                break
    else:
        axes[0].set_xlabel("DE step")

    fig.tight_layout()
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
                        help="Frame directories (frames_p1/, frames_p2/) or paths to *_meta.json")
    parser.add_argument("--from-stats", action="store_true",
                        help="Read from output_stats.json instead of sidecar meta files")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: shape_distance.png alongside first path)")
    parser.add_argument("--show", action="store_true", help="Show interactive plot")
    args = parser.parse_args()

    if args.from_stats:
        # Find output_stats.json in the given paths
        records: list[dict] = []
        for p in args.paths:
            candidates = [p] if p.is_file() else list(p.rglob("output_stats.json"))
            for c in candidates:
                records.extend(_load_from_stats(c))
    else:
        records = _load_meta_files(args.paths)

    out_path = args.out
    if out_path is None:
        base = args.paths[0]
        out_path = (base.parent if base.is_file() else base) / "shape_distance.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_shape_distance(records, out_path, args.show)


if __name__ == "__main__":
    main()
