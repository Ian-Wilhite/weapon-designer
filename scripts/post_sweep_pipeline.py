#!/usr/bin/env python3
"""Post-sweep pipeline: waits for heavyweight sweep, then chains:
  1. Charts regen + PDF recompile
  2. Fine-mesh SF spot-check
  3. Multi-replicate compare study (featherweight_disk, heavyweight_disk, max_energy_disk)

Run this in a terminal and leave it; it will block until the sweep is done,
then execute each step automatically.

Usage
-----
    python scripts/post_sweep_pipeline.py
    python scripts/post_sweep_pipeline.py --replicates 5 --no-pdf
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

LOG_PATH = ROOT / "post_sweep_pipeline.log"
PYTHON   = sys.executable


def _log(msg: str):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _run(cmd: list[str], cwd: Path | None = None, label: str = "") -> int:
    label = label or " ".join(cmd[:3])
    _log(f"RUN  {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=cwd or ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    proc.wait()
    _log(f"EXIT {proc.returncode}  ({label})")
    return proc.returncode


# ---------------------------------------------------------------------------
# Completion detection
# ---------------------------------------------------------------------------

HEAVY_CASES   = {"bspline", "bezier", "catmull_rom"}
SWEEP_RESULTS = ROOT / "profile_sweep" / "results.json"


def _heavyweight_done() -> bool:
    """True when profile_sweep/results.json has all 3 heavyweight enhanced variants."""
    if not SWEEP_RESULTS.exists():
        return False
    try:
        with open(SWEEP_RESULTS) as f:
            data = json.load(f)
        done = {
            r["method"]
            for r in data
            if r.get("case_name") == "heavyweight_disk"
            and r.get("method") in HEAVY_CASES
            and r.get("status") == "success"
        }
        return done >= HEAVY_CASES
    except Exception:
        return False


def _wait_for_heavyweight(poll_s: int = 30):
    _log("Waiting for heavyweight disk sweep to finish "
         "(polling results.json every 30 s)…")
    while not _heavyweight_done():
        time.sleep(poll_s)
    _log("Heavyweight sweep complete — all 3 variants in results.json.")


# ---------------------------------------------------------------------------
# Step 1: charts regen + PDF
# ---------------------------------------------------------------------------

def step_charts(no_pdf: bool):
    _log("=" * 60)
    _log("STEP 1: Charts regen + PDF recompile")
    _log("=" * 60)

    rc = _run([
        PYTHON, "scripts/sweep_profiles.py",
        "--charts-only",
        "--output-dir", "profile_sweep",
    ], label="charts-only")
    if rc != 0:
        _log(f"WARNING: charts regen exited {rc} — continuing anyway")

    if no_pdf:
        _log("--no-pdf set — skipping LaTeX compile")
        return

    pdflatex = subprocess.run(["which", "pdflatex"], capture_output=True, text=True)
    if pdflatex.returncode != 0:
        _log("pdflatex not found — skipping PDF compile")
        return

    rc = _run(
        ["pdflatex", "-interaction=nonstopmode", "report.tex"],
        cwd=ROOT / "docs",
        label="pdflatex",
    )
    if rc == 0:
        _log("PDF compiled successfully → docs/report.pdf")
    else:
        _log(f"WARNING: pdflatex exited {rc}")


# ---------------------------------------------------------------------------
# Step 2: fine-mesh SF spot-check
# ---------------------------------------------------------------------------

def step_spotcheck():
    _log("=" * 60)
    _log("STEP 2: Fine-mesh SF spot-check")
    _log("=" * 60)

    rc = _run([
        PYTHON, "scripts/fine_mesh_spotcheck.py",
        "--sweep-dir", "profile_sweep",
        "--out", "profile_sweep/fine_mesh_spotcheck.json",
    ], label="spotcheck")
    if rc != 0:
        _log(f"WARNING: spot-check exited {rc} — continuing")


# ---------------------------------------------------------------------------
# Step 3: multi-replicate compare study
# ---------------------------------------------------------------------------

REPLICATE_CASES = "featherweight_disk,heavyweight_disk,max_energy_disk"


def step_replicates(n_replicates: int, out_dir: str):
    _log("=" * 60)
    _log(f"STEP 3: Multi-replicate compare ({n_replicates} reps × 3 cases)")
    _log("=" * 60)

    rc = _run([
        PYTHON, "scripts/compare_baselines.py",
        "--cases",      REPLICATE_CASES,
        "--replicates", str(n_replicates),
        "--output-dir", out_dir,
    ], label="compare_baselines")

    if rc == 0:
        _log(f"Replicate study complete → {out_dir}/")
    else:
        _log(f"WARNING: compare_baselines exited {rc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-sweep automation pipeline")
    parser.add_argument("--replicates", type=int, default=3,
                        help="Replicates per case/mode for compare study (default: 3)")
    parser.add_argument("--replicate-dir", default="eval_results/replicates_fixed",
                        help="Output dir for replicate study")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip pdflatex compile")
    parser.add_argument("--skip-wait", action="store_true",
                        help="Skip waiting (assume sweep already done)")
    args = parser.parse_args()

    _log("Post-sweep pipeline started")
    _log(f"Log: {LOG_PATH}")

    t0 = time.perf_counter()

    if not args.skip_wait:
        _wait_for_heavyweight()
    else:
        _log("--skip-wait: assuming heavyweight sweep already done")

    step_charts(args.no_pdf)
    step_spotcheck()
    step_replicates(args.replicates, args.replicate_dir)

    elapsed = (time.perf_counter() - t0) / 3600
    _log(f"All steps complete — total wall time: {elapsed:.2f}h")
    _log(f"Full log: {LOG_PATH}")


if __name__ == "__main__":
    main()
