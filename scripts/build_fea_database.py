#!/usr/bin/env python3
"""Build the FEA stress-field database for surrogate model training.

Samples N weapon designs via a Sobol quasi-random sequence over the
B-spline profile parameter space, runs coarse-mesh FEA on each, and
interpolates the resulting von-Mises stress field onto a fixed reference
mesh (solid disk, 8 mm grid).

Outputs (in --out-dir, default fea_database/)
-------------------------------------------
  fea_database/manifest.json        — design registry + metadata
  fea_database/design_NNNN.npz      — per-design: params, vm_stresses (ref mesh)
  fea_database/ref_mesh.npz         — shared reference mesh (nodes, elements)
  fea_database/progress.png         — live progress chart (updated every 10 designs)
  fea_database/sample_designs.png   — thumbnail grid of first 16 designs
  fea_database/database_stats.png   — distribution of key metrics

Usage
-----
    python scripts/build_fea_database.py --n 50
    python scripts/build_fea_database.py --n 500 --mesh-spacing 8 --jobs 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from weapon_designer.config import (
    WeaponConfig, Material, Mounting, Envelope,
    OptimizationParams, OutputParams,
)


# ---------------------------------------------------------------------------
# Reference configuration (standardised for all database designs)
# ---------------------------------------------------------------------------

def _make_ref_cfg() -> WeaponConfig:
    """Canonical configuration for the FEA database."""
    return WeaponConfig(
        material=Material(name="AR500", density_kg_m3=7850, yield_strength_mpa=1400, hardness_hrc=50),
        weapon_style="disk",
        sheet_thickness_mm=10.0,
        weight_budget_kg=3.0,
        rpm=8000,
        mounting=Mounting(bore_diameter_mm=25.4, bolt_circle_diameter_mm=50, num_bolts=4, bolt_hole_diameter_mm=6.5),
        envelope=Envelope(max_radius_mm=100.0),
        optimization=OptimizationParams(
            n_bspline_points=8,
            profile_type="bspline",
            fea_coarse_spacing_mm=8.0,
            evaluation_mode="physical",
        ),
        output=OutputParams(run_dir="fea_database"),
    )


# ---------------------------------------------------------------------------
# Reference mesh: fixed solid disk
# ---------------------------------------------------------------------------

def _build_reference_mesh(R_mm: float = 100.0, spacing: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed reference mesh (solid disk) for field interpolation.

    Returns (nodes (N,2), elements (M,3)).
    """
    import math
    from shapely.geometry import Polygon as ShapelyPolygon
    from weapon_designer.fea import fea_stress_analysis_with_mesh

    n_pts = 120
    theta = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    disk  = ShapelyPolygon([(R_mm * np.cos(t), R_mm * np.sin(t)) for t in theta])

    # Run FEA with minimal RPM just to get the mesh
    result = fea_stress_analysis_with_mesh(
        disk, rpm=1.0, density_kg_m3=7850, thickness_mm=10.0,
        yield_strength_mpa=1400, bore_diameter_mm=0.0, mesh_spacing=spacing,
    )
    return result["nodes"], result["elements"]


# ---------------------------------------------------------------------------
# Sobol sampling
# ---------------------------------------------------------------------------

def _sobol_samples(n: int, d: int, lo: np.ndarray, hi: np.ndarray, seed: int = 0) -> np.ndarray:
    """Sample n points in [lo, hi]^d using a Sobol quasi-random sequence."""
    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=d, scramble=True, seed=seed)
        unit = sampler.random(n)
    except Exception:
        # Fallback: LHS if Sobol unavailable
        unit = np.zeros((n, d))
        for j in range(d):
            perm = np.random.default_rng(seed + j).permutation(n)
            unit[:, j] = (perm + np.random.default_rng(seed + j + 1000).random(n)) / n

    return lo + unit * (hi - lo)


# ---------------------------------------------------------------------------
# Per-design FEA runner
# ---------------------------------------------------------------------------

def _run_one_design(
    idx: int,
    params: np.ndarray,
    ref_nodes: np.ndarray,
    ref_elements: np.ndarray,
    cfg: WeaponConfig,
    mesh_spacing: float,
) -> dict | None:
    """Build weapon from params, run FEA, interpolate onto reference mesh.

    Returns a dict with all results, or None on failure.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    from weapon_designer.profile_builder import build_profile
    from weapon_designer.geometry import assemble_weapon
    from weapon_designer.constraints import validate_geometry
    from weapon_designer.fea import fea_stress_analysis_with_mesh
    from weapon_designer.physics import polygon_mass_kg, mass_moi_kg_mm2, com_offset_mm
    from weapon_designer.objectives_enhanced import kinematic_spiral_bite

    try:
        outer = build_profile("bspline", params, cfg)
        if outer is None or outer.is_empty:
            return None

        weapon = assemble_weapon(outer, cfg.mounting, [])
        weapon = validate_geometry(weapon)
        if weapon.is_empty or weapon.area < 100:
            return None

        t = cfg.sheet_thickness_mm
        rho = cfg.material.density_kg_m3
        mass = polygon_mass_kg(weapon, t, rho)
        moi  = mass_moi_kg_mm2(weapon, t, rho)
        com  = com_offset_mm(weapon)

        v = float(getattr(cfg.optimization, "drive_speed_mps", 6.0))
        bite = kinematic_spiral_bite(weapon, cfg.rpm, drive_speed_mps=v)

        result = fea_stress_analysis_with_mesh(
            weapon,
            rpm=cfg.rpm,
            density_kg_m3=rho,
            thickness_mm=t,
            yield_strength_mpa=cfg.material.yield_strength_mpa,
            bore_diameter_mm=cfg.mounting.bore_diameter_mm,
            mesh_spacing=mesh_spacing,
        )

        # ── Interpolate stress field onto reference mesh ──────────────────
        # Source: per-element centroids + vm stresses
        elements = result["elements"]
        nodes    = result["nodes"]
        vm_src   = result["vm_stresses"]   # (M_src,)

        # Compute element centroids for source
        src_centroids = nodes[elements].mean(axis=1)   # (M_src, 2)

        # Target: reference mesh element centroids
        ref_centroids = ref_nodes[ref_elements].mean(axis=1)   # (M_ref, 2)

        try:
            interp = LinearNDInterpolator(src_centroids, vm_src, fill_value=np.nan)
            vm_ref = interp(ref_centroids)
            nan_mask = np.isnan(vm_ref)
            if nan_mask.any():
                # Fallback to nearest-neighbour for out-of-hull points
                nn_interp = NearestNDInterpolator(src_centroids, vm_src)
                vm_ref[nan_mask] = nn_interp(ref_centroids[nan_mask])
        except Exception:
            # Full nearest-neighbour fallback
            from scipy.spatial import cKDTree
            tree = cKDTree(src_centroids)
            _, nn_idx = tree.query(ref_centroids)
            vm_ref = vm_src[nn_idx]

        return {
            "idx":               idx,
            "params":            params,
            "vm_stresses_ref":   vm_ref,   # (M_ref,) on reference mesh
            "mass_kg":           float(mass),
            "moi_kg_mm2":        float(moi),
            "com_offset_mm":     float(com),
            "bite_mm":           float(bite["bite_mm"]),
            "n_contacts":        int(bite["n_contacts"]),
            "peak_stress_mpa":   float(result["peak_stress_mpa"]),
            "safety_factor":     float(result["safety_factor"]),
            "fea_score":         float(result["fea_score"]),
            "n_elements_src":    int(result["n_elements"]),
            "exterior_coords":   np.array(list(weapon.exterior.coords)),
        }
    except Exception as e:
        print(f"  [design {idx:04d}] FAILED: {e}")
        return None


# ---------------------------------------------------------------------------
# Public loader (used by build_rom.py and validate_rom.py)
# ---------------------------------------------------------------------------

def load_database(db_dir: Path, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load all design npz files from db_dir. Returns (params (N,d), stresses (N,M_ref))."""
    manifest_path = Path(db_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in {db_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = manifest.get("designs", [])
    if not entries:
        raise RuntimeError("manifest.json has no designs — run build_fea_database.py first")

    params_list:   list[np.ndarray] = []
    stresses_list: list[np.ndarray] = []
    skipped = 0

    for entry in entries:
        fpath = Path(db_dir) / entry["file"]
        if not fpath.exists():
            skipped += 1
            continue
        try:
            data = np.load(fpath, allow_pickle=True)
            params_list.append(data["params"])
            stresses_list.append(data["vm_stresses_ref"])
        except Exception as e:
            if verbose:
                print(f"  [skip] {fpath.name}: {e}")
            skipped += 1

    if not params_list:
        raise RuntimeError("No valid designs loaded from database")

    params   = np.stack(params_list,   axis=0)
    stresses = np.stack(stresses_list, axis=0)

    if verbose:
        print(f"Loaded {len(params_list)} designs ({skipped} skipped)")
        print(f"  params shape:   {params.shape}")
        print(f"  stresses shape: {stresses.shape}")

    return params, stresses


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_progress(completed: list[dict], out_dir: Path, n_total: int):
    """Live progress chart: FEA score, mass, safety factor distributions."""
    if len(completed) < 2:
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    scores  = [d["fea_score"]       for d in completed]
    masses  = [d["mass_kg"]         for d in completed]
    sfs     = [d["safety_factor"]   for d in completed]
    bites   = [d["bite_mm"]         for d in completed]

    for ax, data, label, color in [
        (axes[0], scores, "FEA Score",       "#2196F3"),
        (axes[1], masses, "Mass (kg)",        "#FF9800"),
        (axes[2], sfs,    "Safety Factor",    "#4CAF50"),
        (axes[3], bites,  "Bite Depth (mm)",  "#9C27B0"),
    ]:
        data_clean = [v for v in data if np.isfinite(v)]
        if not data_clean:
            continue
        ax.hist(data_clean, bins=min(20, len(data_clean)), color=color, alpha=0.75, edgecolor="k", linewidth=0.5)
        mean_val = float(np.mean(data_clean))
        ax.axvline(mean_val, color="r", ls="--", lw=1.5, label=f"mean={mean_val:.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"FEA Database Progress ({len(completed)}/{n_total} designs)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    path = out_dir / "progress.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_sample_designs(completed: list[dict], out_dir: Path, n_show: int = 16):
    """Thumbnail grid of weapon outer profiles, coloured by FEA score."""
    n_show  = min(n_show, len(completed))
    if n_show < 1:
        return

    import math as _math
    ncols = 4
    nrows = _math.ceil(n_show / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    scores_all = [d["fea_score"] for d in completed[:n_show]]
    vmin, vmax = min(scores_all), max(scores_all)

    cmap = plt.cm.viridis
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    for i, (d, ax) in enumerate(zip(completed[:n_show], axes)):
        coords = d["exterior_coords"]
        ax.fill(coords[:, 0], coords[:, 1], alpha=0.5,
                color=cmap((d["fea_score"] - vmin) / max(vmax - vmin, 1e-6)))
        ax.plot(coords[:, 0], coords[:, 1], "k-", lw=0.7)
        ax.set_aspect("equal")
        ax.set_title(f"#{d['idx']:04d}  score={d['fea_score']:.3f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n_show:]:
        ax.set_visible(False)

    fig.colorbar(sm, ax=axes[:n_show], label="FEA score", shrink=0.6)
    fig.suptitle(f"FEA Database — First {n_show} Designs", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "sample_designs.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_matrix(completed: list[dict], out_dir: Path):
    """Scatter matrix of key metrics to reveal structure in the database."""
    if len(completed) < 10:
        return

    keys   = ["mass_kg", "moi_kg_mm2", "bite_mm", "safety_factor", "fea_score"]
    labels = ["Mass (kg)", "MOI (kg·mm²)", "Bite (mm)", "Safety F.", "Score"]
    data   = np.array([[d[k] for k in keys] for d in completed])

    n = len(keys)
    fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n))

    scores = data[:, -1]
    sc_norm = (scores - scores.min()) / max(float(np.ptp(scores)), 1e-6)
    colours = plt.cm.viridis(sc_norm)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                ax.hist(data[:, i], bins=15, color="#607D8B", alpha=0.7)
                ax.set_title(labels[i], fontsize=8)
            else:
                ax.scatter(data[:, j], data[:, i], c=colours, s=10, alpha=0.6)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=7)
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle("FEA Database — Metric Correlation Matrix", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "database_stats.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import math

    parser = argparse.ArgumentParser(description="Build FEA stress-field database")
    parser.add_argument("--n",             type=int,   default=50,   help="Number of designs to sample")
    parser.add_argument("--out-dir",       default="fea_database",   help="Output directory")
    parser.add_argument("--mesh-spacing",  type=float, default=8.0,  help="FEA mesh spacing (mm)")
    parser.add_argument("--ref-spacing",   type=float, default=8.0,  help="Reference mesh spacing (mm)")
    parser.add_argument("--seed",          type=int,   default=42,   help="Random seed for Sobol")
    parser.add_argument("--resume",        action="store_true",      help="Skip already-computed designs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _make_ref_cfg()

    print("=" * 60)
    print(f"  FEA DATABASE — sampling {args.n} designs")
    print(f"  Output: {out_dir}/")
    print("=" * 60)

    # ── Build / load reference mesh ───────────────────────────────────────
    ref_mesh_path = out_dir / "ref_mesh.npz"
    if ref_mesh_path.exists():
        print("Loading existing reference mesh...")
        ref_data = np.load(ref_mesh_path)
        ref_nodes, ref_elements = ref_data["nodes"], ref_data["elements"]
    else:
        print(f"Building reference mesh (spacing={args.ref_spacing}mm)...")
        ref_nodes, ref_elements = _build_reference_mesh(
            R_mm=cfg.envelope.max_radius_mm, spacing=args.ref_spacing
        )
        np.savez_compressed(ref_mesh_path, nodes=ref_nodes, elements=ref_elements)
        print(f"  Reference mesh: {len(ref_nodes)} nodes, {len(ref_elements)} elements → {ref_mesh_path}")

    # ── Sobol parameter samples ───────────────────────────────────────────
    from weapon_designer.profile_builder import get_profile_bounds
    bounds = get_profile_bounds("bspline", cfg)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    d  = len(bounds)

    all_params = _sobol_samples(args.n, d, lo, hi, seed=args.seed)
    print(f"  Parameter space: {d} dims  [{lo.min():.1f}, {hi.max():.1f}] mm")

    # ── Load manifest (for resume) ────────────────────────────────────────
    manifest_path = out_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        done_ids = {e["idx"] for e in manifest["designs"]}
    else:
        manifest = {"n_designs": 0, "designs": []}
        done_ids = set()

    # ── Main loop ─────────────────────────────────────────────────────────
    completed: list[dict] = []
    n_failed = 0
    t_start = time.perf_counter()

    for i, params in enumerate(all_params):
        if args.resume and i in done_ids:
            existing_path = out_dir / f"design_{i:04d}.npz"
            if existing_path.exists():
                d_data = np.load(existing_path, allow_pickle=True)
                completed.append({
                    "idx":             i,
                    "fea_score":       float(d_data.get("fea_score", 0)),
                    "mass_kg":         float(d_data.get("mass_kg", 0)),
                    "moi_kg_mm2":      float(d_data.get("moi_kg_mm2", 0)),
                    "bite_mm":         float(d_data.get("bite_mm", 0)),
                    "safety_factor":   float(d_data.get("safety_factor", 0)),
                    "exterior_coords": d_data.get("exterior_coords", np.zeros((2, 2))),
                })
            continue

        t0 = time.perf_counter()
        result = _run_one_design(i, params, ref_nodes, ref_elements, cfg, args.mesh_spacing)
        dt = time.perf_counter() - t0

        if result is None:
            n_failed += 1
            print(f"  [{i+1:4d}/{args.n}] FAILED  ({dt*1e3:.0f} ms)")
            continue

        # Save design npz
        save_path = out_dir / f"design_{i:04d}.npz"
        np.savez_compressed(
            save_path,
            params=result["params"],
            vm_stresses_ref=result["vm_stresses_ref"],
            exterior_coords=result["exterior_coords"],
            mass_kg=result["mass_kg"],
            moi_kg_mm2=result["moi_kg_mm2"],
            com_offset_mm=result["com_offset_mm"],
            bite_mm=result["bite_mm"],
            n_contacts=result["n_contacts"],
            peak_stress_mpa=result["peak_stress_mpa"],
            safety_factor=result["safety_factor"],
            fea_score=result["fea_score"],
        )

        completed.append(result)
        manifest["designs"].append({
            "idx":           i,
            "file":          f"design_{i:04d}.npz",
            "fea_score":     result["fea_score"],
            "safety_factor": result["safety_factor"],
            "mass_kg":       result["mass_kg"],
            "bite_mm":       result["bite_mm"],
        })
        manifest["n_designs"] = len(manifest["designs"])

        elapsed_total = time.perf_counter() - t_start
        n_done = len(completed)
        eta_s = elapsed_total / n_done * (args.n - n_done) if n_done > 0 else 0
        print(
            f"  [{i+1:4d}/{args.n}]  score={result['fea_score']:.3f}  "
            f"SF={result['safety_factor']:.2f}  bite={result['bite_mm']:.1f}mm  "
            f"{dt*1e3:.0f}ms  ETA:{eta_s/60:.1f}min",
            flush=True,
        )

        # Write manifest after every design so --resume can recover from any crash
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Regenerate plots every 10 designs (heavier I/O)
        if n_done % 10 == 0:
            _plot_progress(completed, out_dir, args.n)
            if n_done <= 64:
                _plot_sample_designs(completed, out_dir)

    # ── Final outputs ─────────────────────────────────────────────────────
    manifest["n_failed"]         = n_failed
    manifest["n_ref_nodes"]      = len(ref_nodes)
    manifest["n_ref_elements"]   = len(ref_elements)
    manifest["mesh_spacing_mm"]  = args.mesh_spacing
    manifest["ref_spacing_mm"]   = args.ref_spacing
    manifest["n_bspline_points"] = getattr(cfg.optimization, "n_bspline_points", 8)
    manifest["param_lo"]         = lo.tolist()
    manifest["param_hi"]         = hi.tolist()

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    _plot_progress(completed, out_dir, args.n)
    _plot_sample_designs(completed, out_dir)
    _plot_correlation_matrix(completed, out_dir)

    total_time = time.perf_counter() - t_start
    print(f"\nDone: {len(completed)} designs saved, {n_failed} failed")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    import math
    main()
