#!/usr/bin/env python3
"""Fine-mesh SF spot-check: quantify coarse (10mm) vs fine (4mm) FEA error.

Reads profile_sweep/results.json and compares:
  - last-step SF from convergence history  (coarse mesh, during optimisation)
  - final_metrics SF                       (fine mesh, committed design eval)

Reports per-run correction factors and flags any design where the coarse mesh
suggested SF >= 1.5 but fine mesh disagrees by > 20%.

Usage
-----
    python scripts/fine_mesh_spotcheck.py
    python scripts/fine_mesh_spotcheck.py --sweep-dir profile_sweep --out spotcheck.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default="profile_sweep")
    parser.add_argument("--out",       default="profile_sweep/fine_mesh_spotcheck.json")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    results_path = sweep_dir / "results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    enhanced = [r for r in results
                if r.get("method") != "fourier_baseline"
                and r.get("status") == "success"
                and r.get("final_metrics", {}).get("fea_safety_factor") is not None
                and r.get("convergence")]

    if not enhanced:
        print("No enhanced runs with FEA data found.")
        sys.exit(0)

    rows = []
    print(f"\n{'Case':<28} {'Method':<14} {'SF_coarse':>10} {'SF_fine':>10} "
          f"{'Ratio':>8} {'ΔSF%':>7} {'Flag'}")
    print("-" * 82)

    for r in sorted(enhanced, key=lambda x: (x["case_name"], x["method"])):
        conv = r["convergence"]
        # Last step SF from the optimisation loop (coarse mesh)
        last_sf = None
        for step in reversed(conv):
            v = step.get("sf")
            if v is not None and v > 0:
                last_sf = float(v)
                break

        fine_sf = float(r["final_metrics"]["fea_safety_factor"])

        if last_sf is None or last_sf <= 0:
            continue

        ratio   = fine_sf / last_sf
        delta_pct = (fine_sf - last_sf) / last_sf * 100

        # Flag: coarse said safe (>=1.5) but fine disagrees by >20%
        flag = ""
        if last_sf >= 1.5 and abs(delta_pct) > 20:
            flag = "⚠ LARGE SHIFT"
        elif last_sf < 1.5 and fine_sf >= 1.5:
            flag = "↑ recovered"
        elif last_sf >= 1.5 and fine_sf < 1.5:
            flag = "↓ UNSAFE at fine"

        print(f"{r['case_name']:<28} {r['method']:<14} "
              f"{last_sf:>10.3f} {fine_sf:>10.3f} "
              f"{ratio:>8.3f} {delta_pct:>+7.1f}% {flag}")

        rows.append({
            "case_name":    r["case_name"],
            "method":       r["method"],
            "sf_coarse":    last_sf,
            "sf_fine":      fine_sf,
            "ratio":        ratio,
            "delta_pct":    delta_pct,
            "flag":         flag,
            "final_score":  r.get("final_score"),
        })

    # Summary statistics
    ratios = [row["ratio"] for row in rows]
    deltas = [row["delta_pct"] for row in rows]
    if ratios:
        import statistics
        print(f"\nSummary ({len(rows)} enhanced runs):")
        print(f"  SF_fine / SF_coarse:  mean={statistics.mean(ratios):.3f}  "
              f"std={statistics.stdev(ratios) if len(ratios)>1 else 0:.3f}  "
              f"range=[{min(ratios):.3f}, {max(ratios):.3f}]")
        print(f"  ΔSF%:                 mean={statistics.mean(deltas):+.1f}%  "
              f"max_abs={max(abs(d) for d in deltas):.1f}%")
        large = [r for r in rows if abs(r["delta_pct"]) > 20]
        if large:
            print(f"  ⚠  {len(large)} run(s) with |ΔSF| > 20% (coarse mesh materially misleading):")
            for r in large:
                print(f"       {r['case_name']} / {r['method']}: "
                      f"coarse={r['sf_coarse']:.2f} → fine={r['sf_fine']:.2f}")
        else:
            print(f"  ✓  All runs within 20% — coarse mesh is a reliable proxy.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"runs": rows, "n": len(rows)}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
