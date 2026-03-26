#!/usr/bin/env python3
"""Resumable surrogate pipeline: FEA database → ROM build → UCB optimizer.

Each stage is independently resumable:
  - FEA database: uses --resume to skip already-computed designs
  - ROM build:    always re-runs (fast, ~5 min); reads completed designs only
  - Surrogate:    checkpoints every 50 evals to surrogate_checkpoint.json

Usage
-----
    # First run (or after crash — stages auto-detect existing work):
    python scripts/run_surrogate_pipeline.py

    # Override defaults:
    python scripts/run_surrogate_pipeline.py \
        --n-designs 500 \
        --variance-threshold 0.90 \
        --case featherweight_disk \
        --db-dir fea_database_surrogate \
        --rom-dir rom_surrogate \
        --out-dir surrogate_run

Crash recovery
--------------
    If the process dies at any stage, re-run the same command.
    Completed .npz designs are never re-computed; the ROM is rebuilt from
    whatever designs exist; the optimizer restarts fresh (GP warm-start from
    surrogate_checkpoint.json if present — best_x and surr_data are logged).
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

PIPELINE_LOG = None   # set in main()


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if PIPELINE_LOG is not None:
        with open(PIPELINE_LOG, "a") as f:
            f.write(line + "\n")


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a subprocess, tee-ing stdout/stderr to the pipeline log."""
    _log(f"CMD: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd or ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        if PIPELINE_LOG is not None:
            with open(PIPELINE_LOG, "a") as f:
                f.write(line + "\n")
    proc.wait()
    return proc.returncode


def stage_build_database(db_dir: Path, n_designs: int, mesh_spacing: float, seed: int) -> bool:
    """Stage 1: build / resume FEA database."""
    _log(f"=== STAGE 1: FEA Database (target={n_designs}, db={db_dir}) ===")

    manifest = db_dir / "manifest.json"
    n_done = 0
    if manifest.exists():
        with open(manifest) as f:
            m = json.load(f)
        n_done = m.get("n_designs", 0)
        _log(f"  Existing database: {n_done} designs already computed")

    if n_done >= n_designs:
        _log(f"  Database complete ({n_done}/{n_designs}) — skipping build")
        return True

    _log(f"  Need {n_designs - n_done} more designs — running with --resume")
    rc = _run([
        sys.executable, "scripts/build_fea_database.py",
        "--n", str(n_designs),
        "--out-dir", str(db_dir),
        "--mesh-spacing", str(mesh_spacing),
        "--seed", str(seed),
        "--resume",
    ])
    if rc != 0:
        _log(f"  ERROR: build_fea_database.py exited {rc}")
        return False

    with open(manifest) as f:
        m = json.load(f)
    _log(f"  Database complete: {m.get('n_designs', 0)} designs saved")
    return True


def stage_build_rom(db_dir: Path, rom_dir: Path, variance_threshold: float) -> bool:
    """Stage 2: build POD/GP ROM from database."""
    _log(f"=== STAGE 2: ROM Build (threshold={variance_threshold:.0%}, rom={rom_dir}) ===")

    # Count available designs
    manifest = db_dir / "manifest.json"
    if not manifest.exists():
        _log("  ERROR: no manifest.json — run Stage 1 first")
        return False
    with open(manifest) as f:
        m = json.load(f)
    n_avail = m.get("n_designs", 0)
    _log(f"  Using {n_avail} designs from {db_dir}")

    rc = _run([
        sys.executable, "scripts/build_rom.py",
        "--db-dir", str(db_dir),
        "--variance-threshold", str(variance_threshold),
        "--out-dir", str(rom_dir),
    ])
    if rc != 0:
        _log(f"  ERROR: build_rom.py exited {rc}")
        return False

    pkl = rom_dir / "fea_surrogate.pkl"
    if not pkl.exists():
        _log("  ERROR: fea_surrogate.pkl not found after ROM build")
        return False
    _log(f"  ROM saved → {pkl}")
    return True


def stage_run_surrogate(cfg_case: str, rom_dir: Path, out_dir: Path,
                        max_iter: int, beta_ucb: float) -> bool:
    """Stage 3: run UCB surrogate optimizer."""
    _log(f"=== STAGE 3: Surrogate Optimizer (case={cfg_case}, out={out_dir}) ===")

    ckpt = out_dir / "surrogate_checkpoint.json"
    if ckpt.exists():
        with open(ckpt) as f:
            c = json.load(f)
        _log(f"  Checkpoint found: {c.get('n_total',0)} evals, "
             f"best_score={c.get('best_score',0):.4f} — will start fresh DE but log is preserved")

    # Inline script avoids subprocess pickling issues with surrogate object
    inline = f"""
import sys
sys.path.insert(0, "{ROOT / 'src'}")
import pickle
from pathlib import Path
from weapon_designer.config import WeaponConfig, Material, Mounting, Envelope, OptimizationParams
from weapon_designer.optimizer_surrogate import optimize_surrogate

# Load surrogate
pkl_path = Path("{rom_dir / 'fea_surrogate.pkl'}")
with open(pkl_path, "rb") as f:
    surrogate = pickle.load(f)
print(f"Loaded surrogate: k={{surrogate.k_}} modes", flush=True)

# Case config: featherweight disk (known-good, fast mesh)
cfg = WeaponConfig(
    material=Material(name="S7 Tool Steel", density_kg_m3=7750,
                      yield_strength_mpa=1600, hardness_hrc=58),
    weapon_style="disk",
    sheet_thickness_mm=6.35,
    weight_budget_kg=0.75,
    rpm=12000,
    mounting=Mounting(bore_diameter_mm=8.0, bolt_circle_diameter_mm=20,
                      num_bolts=3, bolt_hole_diameter_mm=3.5),
    envelope=Envelope(max_radius_mm=80.0),
    optimization=OptimizationParams(
        n_bspline_points=8,
        profile_type="bspline",
        fea_coarse_spacing_mm=10.0,
        evaluation_mode="physical",
        max_iterations={max_iter},
        population_size=15,
    ),
)

result = optimize_surrogate(
    cfg, surrogate,
    case_dir=Path("{out_dir}"),
    max_iterations={max_iter},
    beta_ucb={beta_ucb},
    checkpoint_interval=50,
    verbose=True,
)
print(f"\\nFinal: score={{result['score']:.4f}}  "
      f"FEA calls={{result['n_fea_calls']}}/{{result['n_total_evals']}} "
      f"({{100*result['fea_rate']:.1f}})  wall={{result['wall_time_s']/3600:.2f}}h",
      flush=True)
"""
    script_path = out_dir / "_run_surrogate.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(inline)

    rc = _run([sys.executable, str(script_path)])
    if rc != 0:
        _log(f"  ERROR: surrogate optimizer exited {rc}")
        return False
    _log("  Surrogate run complete")
    return True


def main():
    global PIPELINE_LOG

    parser = argparse.ArgumentParser(description="Resumable surrogate pipeline")
    parser.add_argument("--n-designs",          type=int,   default=500)
    parser.add_argument("--variance-threshold",  type=float, default=0.90,
                        help="POD variance threshold (0.90 = best calibration)")
    parser.add_argument("--case",               default="featherweight_disk")
    parser.add_argument("--db-dir",             default="fea_database_surrogate")
    parser.add_argument("--rom-dir",            default="rom_surrogate")
    parser.add_argument("--out-dir",            default="surrogate_run")
    parser.add_argument("--mesh-spacing",       type=float, default=8.0)
    parser.add_argument("--db-seed",            type=int,   default=42)
    parser.add_argument("--max-iter",           type=int,   default=100)
    parser.add_argument("--beta-ucb",           type=float, default=2.0)
    parser.add_argument("--skip-db",            action="store_true", help="Skip Stage 1")
    parser.add_argument("--skip-rom",           action="store_true", help="Skip Stage 2")
    args = parser.parse_args()

    db_dir  = Path(args.db_dir)
    rom_dir = Path(args.rom_dir)
    out_dir = Path(args.out_dir)

    log_path = out_dir / "pipeline.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    PIPELINE_LOG = log_path
    _log(f"Pipeline started — log: {log_path}")
    _log(f"Args: {vars(args)}")

    t0 = time.perf_counter()

    if not args.skip_db:
        ok = stage_build_database(db_dir, args.n_designs, args.mesh_spacing, args.db_seed)
        if not ok:
            _log("STAGE 1 FAILED — aborting")
            sys.exit(1)

    if not args.skip_rom:
        ok = stage_build_rom(db_dir, rom_dir, args.variance_threshold)
        if not ok:
            _log("STAGE 2 FAILED — aborting")
            sys.exit(1)

    ok = stage_run_surrogate(args.case, rom_dir, out_dir, args.max_iter, args.beta_ucb)
    if not ok:
        _log("STAGE 3 FAILED")
        sys.exit(1)

    total = time.perf_counter() - t0
    _log(f"Pipeline complete — total wall time: {total/3600:.2f}h")


if __name__ == "__main__":
    main()
