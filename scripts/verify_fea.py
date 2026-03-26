#!/usr/bin/env python3
"""FEA verification script — analytical vs. numerical stress comparison.

Tests
-----
1. Centrifugal stress in a solid disk
   Analytical: σ_r(0) = ρ·ω²·R²·(3+ν)/8  (peak at centre)
   Expected FEA error: < 15%

2. von-Mises stress at r = 0.7R in a solid disk
   Analytical: σ_vm = √(σ_r² + σ_θ² − σ_r·σ_θ)  (Lamé biaxial)
   Expected FEA error: < 15%

3. Smoke test — short optimizer run
   Checks that bite and safety factor are in physically plausible ranges.

Produces
--------
  verify_fea/analytical_vs_fea.png  — bar chart comparing all test results
  verify_fea/stress_fields.png      — FEA von-Mises stress field plots
  verify_fea/results.json           — machine-readable pass/fail table

Usage
-----
    python scripts/verify_fea.py [--no-smoke-test]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from shapely.geometry import Point, Polygon

from weapon_designer.config import (
    WeaponConfig, Material, Mounting, Envelope,
    OptimizationParams, OptimizationWeights, OutputParams,
)
from weapon_designer.fea import fea_stress_analysis_with_mesh


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RHO    = 7850.0    # kg/m³ (steel)
NU     = 0.3       # Poisson's ratio
RPM    = 8000.0
OMEGA  = 2 * math.pi * RPM / 60.0     # rad/s

PASS_THRESHOLD = 0.15   # 15% max allowable error


# ---------------------------------------------------------------------------
# Test 1 — solid disk centrifugal stress
# ---------------------------------------------------------------------------

def _solid_disk_polygon(R_mm: float, n_pts: int = 120) -> Polygon:
    theta = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    pts   = [(R_mm * math.cos(t), R_mm * math.sin(t)) for t in theta]
    return Polygon(pts)


def test_solid_disk(R_mm: float = 100.0, mesh_spacing: float = 5.0) -> dict:
    """Compare FEA peak centre stress against the Lamé analytical solution.

    Analytical peak (at r=0):
        σ_r(0) = σ_θ(0) = ρ·ω²·R²·(3+ν)/8

    FEA approximates this as the mean stress in the central region
    (within 0.15·R of the origin).
    """
    poly = _solid_disk_polygon(R_mm)

    # Analytical (MPa)
    R_m = R_mm * 1e-3
    omega = OMEGA
    sigma_analytical_pa = RHO * omega**2 * R_m**2 * (3 + NU) / 8.0
    sigma_analytical    = sigma_analytical_pa * 1e-6   # Pa → MPa

    # FEA
    t0 = time.perf_counter()
    result = fea_stress_analysis_with_mesh(
        poly,
        rpm=RPM,
        density_kg_m3=RHO,
        thickness_mm=10.0,
        yield_strength_mpa=1400.0,
        bore_diameter_mm=0.0,          # no bore — solid disk
        mesh_spacing=mesh_spacing,
    )
    fea_dt = time.perf_counter() - t0

    nodes    = result["nodes"]       # (N, 2)
    elements = result["elements"]    # (M, 3)
    vm_stresses = result["vm_stresses"]   # (M,)  per-element

    # Estimate central stress: mean of elements whose centroid is within 0.15·R
    centroids_r = np.sqrt(
        (nodes[elements, 0].mean(axis=1))**2 +
        (nodes[elements, 1].mean(axis=1))**2
    )
    central_mask = centroids_r < 0.15 * R_mm
    if central_mask.any():
        sigma_fea = float(vm_stresses[central_mask].mean())
    else:
        sigma_fea = float(vm_stresses.mean())

    error = abs(sigma_fea - sigma_analytical) / max(sigma_analytical, 1e-6)
    passed = error <= PASS_THRESHOLD

    print(f"\n[Test 1] Solid disk  R={R_mm:.0f}mm  RPM={RPM:.0f}")
    print(f"  Analytical σ_centre = {sigma_analytical:.1f} MPa")
    print(f"  FEA        σ_centre = {sigma_fea:.1f} MPa  (mean of central elements)")
    print(f"  Error      = {100*error:.1f}%   {'PASS' if passed else 'FAIL'}")
    print(f"  FEA time   = {fea_dt*1e3:.0f} ms  ({result['n_elements']} elements)")

    return {
        "name": "solid_disk_centrifugal",
        "sigma_analytical_mpa": sigma_analytical,
        "sigma_fea_mpa": sigma_fea,
        "error_frac": error,
        "passed": passed,
        "n_elements": result["n_elements"],
        "fea_dt_s": fea_dt,
        # For plotting
        "_nodes": nodes,
        "_elements": elements,
        "_vm": vm_stresses,
        "_R_mm": R_mm,
    }


# ---------------------------------------------------------------------------
# Test 2 — solid disk mid-radius stress profile
# ---------------------------------------------------------------------------

def test_mid_radius(R_mm: float = 100.0, r_frac: float = 0.70, mesh_spacing: float = 5.0) -> dict:
    """Verify FEA reproduces the Lamé solid-disk stress at r = r_frac·R.

    Analytical (Timoshenko, solid disk, free outer rim, zero-displacement centre):
        σ_r(r) = ρω²(3+ν)/8 × (R² − r²)
        σ_θ(r) = ρω²(3+ν)/8 × [R² − (1+3ν)/(3+ν) × r²]
        σ_vm   = √(σ_r² + σ_θ² − σ_r·σ_θ)    [plane-stress von-Mises]

    At r = 0.7R the radial and hoop stresses are both non-zero (unlike the centre
    where σ_r = σ_θ, or the rim where σ_r = 0), making this a sensitive test of
    the biaxial stress state.

    FEA: mean von-Mises over elements whose centroid lies in [0.65R, 0.75R].
    Threshold: 15%.
    """
    poly = _solid_disk_polygon(R_mm)

    r_check = r_frac * R_mm   # mm
    R_m     = R_mm * 1e-3
    r_m     = r_check * 1e-3

    factor = RHO * OMEGA**2 * (3 + NU) / 8.0   # Pa/m²
    sigma_r_pa = factor * (R_m**2 - r_m**2)
    sigma_t_pa = factor * (R_m**2 - (1 + 3*NU)/(3 + NU) * r_m**2)
    sigma_vm_analytical = math.sqrt(
        sigma_r_pa**2 + sigma_t_pa**2 - sigma_r_pa * sigma_t_pa
    ) * 1e-6   # Pa → MPa

    t0 = time.perf_counter()
    result = fea_stress_analysis_with_mesh(
        poly,
        rpm=RPM,
        density_kg_m3=RHO,
        thickness_mm=10.0,
        yield_strength_mpa=1400.0,
        bore_diameter_mm=0.0,
        mesh_spacing=mesh_spacing,
    )
    fea_dt = time.perf_counter() - t0

    nodes       = result["nodes"]
    elements    = result["elements"]
    vm_stresses = result["vm_stresses"]

    # Mean of elements whose centroid is in [0.65R, 0.75R]
    centroids_r = np.sqrt(
        (nodes[elements, 0].mean(axis=1))**2 +
        (nodes[elements, 1].mean(axis=1))**2
    )
    band_mask = (centroids_r >= 0.65 * R_mm) & (centroids_r <= 0.75 * R_mm)
    if band_mask.any():
        sigma_fea = float(vm_stresses[band_mask].mean())
    else:
        sigma_fea = float(vm_stresses.mean())

    error  = abs(sigma_fea - sigma_vm_analytical) / max(sigma_vm_analytical, 1e-6)
    passed = error <= PASS_THRESHOLD

    print(f"\n[Test 2] Solid disk mid-radius  r={r_check:.0f}mm (0.7R)  RPM={RPM:.0f}")
    print(f"  Analytical σ_vm(0.7R) = {sigma_vm_analytical:.1f} MPa  (σ_r={sigma_r_pa*1e-6:.1f}, σ_θ={sigma_t_pa*1e-6:.1f})")
    print(f"  FEA mean σ_vm(0.7R)   = {sigma_fea:.1f} MPa  ({band_mask.sum()} elements in band)")
    print(f"  Error      = {100*error:.1f}%   {'PASS' if passed else 'FAIL'}")
    print(f"  FEA time   = {fea_dt*1e3:.0f} ms  ({result['n_elements']} elements)")

    return {
        "name": "solid_disk_mid_radius",
        "sigma_analytical_mpa": sigma_vm_analytical,
        "sigma_fea_mpa": sigma_fea,
        "error_frac": error,
        "passed": passed,
        "n_elements": result["n_elements"],
        "fea_dt_s": fea_dt,
        "_nodes": nodes,
        "_elements": elements,
        "_vm": vm_stresses,
        "_R_mm": R_mm,
        "_r_frac": r_frac,
    }


# ---------------------------------------------------------------------------
# Test 3 — smoke test (short optimizer run)
# ---------------------------------------------------------------------------

def test_smoke() -> dict:
    """Run a 30-step enhanced optimizer and check physical plausibility.

    Checks:
    - bite_mm in [1, 50]
    - safety_factor in [0.5, 10]
    - score in [0, 1]
    - no Python exception
    """
    print("\n[Test 3] Smoke test — 30-step enhanced optimizer run")

    from weapon_designer.optimizer_enhanced import optimize_enhanced

    cfg = WeaponConfig(
        material=Material(name="AR500", density_kg_m3=7850, yield_strength_mpa=1400, hardness_hrc=50),
        weapon_style="disk",
        sheet_thickness_mm=10.0,
        weight_budget_kg=3.0,
        rpm=8000,
        mounting=Mounting(bore_diameter_mm=25.4, bolt_circle_diameter_mm=50, num_bolts=4, bolt_hole_diameter_mm=6.5),
        envelope=Envelope(max_radius_mm=100.0),
        optimization=OptimizationParams(
            max_iterations=30,
            population_size=20,
            evaluation_mode="enhanced",
            cutout_type="topology",
            fea_interval=0,
            n_bspline_points=6,
            profile_type="bspline",
            topo_n_iter=10,
        ),
        output=OutputParams(run_dir="verify_fea/smoke_run"),
    )

    out_dir = Path("verify_fea/smoke_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    passed = False
    details: dict = {}

    try:
        t0 = time.perf_counter()
        result = optimize_enhanced(cfg, case_dir=out_dir, verbose=False)
        dt = time.perf_counter() - t0

        score  = result.get("score", -1.0)
        m      = result.get("metrics", {})
        bite   = m.get("bite_mm", -1.0)
        sf     = m.get("fea_safety_factor", m.get("structural_integrity", -1.0))

        bite_ok = 1.0 <= bite <= 50.0
        sf_ok   = 0.5 <= sf   <= 10.0
        sc_ok   = 0.0 <= score <= 1.0

        passed = bite_ok and sf_ok and sc_ok

        print(f"  Score         = {score:.4f}  {'OK' if sc_ok else 'OUT OF RANGE'}")
        print(f"  Bite          = {bite:.2f} mm  {'OK' if bite_ok else 'OUT OF RANGE [1,50]'}")
        print(f"  Safety factor = {sf:.2f}  {'OK' if sf_ok else 'OUT OF RANGE [0.5,10]'}")
        print(f"  Wall time     = {dt:.1f} s")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")

        details = {
            "score": score, "bite_mm": bite,
            "safety_factor": sf, "wall_time_s": dt,
            "bite_ok": bite_ok, "sf_ok": sf_ok, "score_ok": sc_ok,
        }

    except Exception as exc:
        print(f"  EXCEPTION: {exc}")
        details = {"exception": str(exc)}

    return {
        "name": "smoke_test_30step",
        "passed": passed,
        **details,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot_stress_field(ax, nodes, elements, vm_stresses, title: str, R_mm: float):
    """Render a von-Mises stress heatmap on ax using matplotlib tricontourf."""
    if len(elements) == 0:
        ax.set_title(f"{title}\n(no elements)")
        return

    # Element centroids → per-element colour
    x_c = nodes[elements, 0].mean(axis=1)
    y_c = nodes[elements, 1].mean(axis=1)
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Interpolate element stress to nodes (simple nearest-element averaging)
    node_stress = np.zeros(len(nodes))
    node_count  = np.zeros(len(nodes))
    for el_idx, el in enumerate(elements):
        for n in el:
            node_stress[n] += vm_stresses[el_idx]
            node_count[n]  += 1
    mask = node_count > 0
    node_stress[mask] /= node_count[mask]

    vmax = np.percentile(node_stress, 98)
    cf = ax.tricontourf(triang, node_stress, levels=20, cmap="hot_r", vmin=0, vmax=vmax)
    plt.colorbar(cf, ax=ax, label="von Mises (MPa)")

    # Draw weapon outline circle
    theta_c = np.linspace(0, 2 * math.pi, 200)
    ax.plot(R_mm * np.cos(theta_c), R_mm * np.sin(theta_c), "w--", lw=1, alpha=0.5)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")


def make_plots(results: list[dict], out_dir: Path):
    """Produce two figures: stress-field plots and analytical-vs-FEA bar chart."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: stress field plots ─────────────────────────────────────
    fea_results = [r for r in results if "_nodes" in r]
    if fea_results:
        fig, axes = plt.subplots(1, len(fea_results), figsize=(7 * len(fea_results), 6))
        if len(fea_results) == 1:
            axes = [axes]

        for ax, r in zip(axes, fea_results):
            _plot_stress_field(
                ax,
                r["_nodes"], r["_elements"], r["_vm"],
                title=r["name"].replace("_", "\n"),
                R_mm=r["_R_mm"],
            )

        fig.suptitle("FEA von-Mises Stress Fields (centrifugal loading)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path1 = out_dir / "stress_fields.png"
        fig.savefig(path1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[verify_fea] stress field plot → {path1}")

    # ── Figure 2: analytical vs FEA comparison ───────────────────────────
    cmp_results = [r for r in results if "sigma_analytical_mpa" in r]
    if cmp_results:
        names       = [r["name"] for r in cmp_results]
        analytical  = [r["sigma_analytical_mpa"] for r in cmp_results]
        fea_vals    = [r["sigma_fea_mpa"] for r in cmp_results]
        errors      = [r["error_frac"] * 100 for r in cmp_results]
        passed_list = [r["passed"] for r in cmp_results]

        x = np.arange(len(names))
        w = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        bars1 = ax1.bar(x - w/2, analytical, w, label="Analytical", color="#2196F3", alpha=0.85)
        bars2 = ax1.bar(x + w/2, fea_vals,   w, label="FEA",        color="#FF9800", alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
        ax1.set_ylabel("Stress (MPa)")
        ax1.set_title("Analytical vs FEA Stress")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        colours = ["#4CAF50" if p else "#F44336" for p in passed_list]
        ax2.bar(x, errors, color=colours, edgecolor="k", linewidth=0.8)
        ax2.axhline(PASS_THRESHOLD * 100, color="r", ls="--", label=f"Threshold {PASS_THRESHOLD*100:.0f}%")
        ax2.set_xticks(x)
        ax2.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
        ax2.set_ylabel("Relative error (%)")
        ax2.set_title("FEA Error vs. Analytical")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        # Annotate pass/fail
        for xi, (err, p) in enumerate(zip(errors, passed_list)):
            ax2.text(xi, err + 0.5, "PASS" if p else "FAIL", ha="center", fontsize=10,
                     color="darkgreen" if p else "darkred", fontweight="bold")

        fig.suptitle("FEA Verification — Analytical Solutions", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path2 = out_dir / "analytical_vs_fea.png"
        fig.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[verify_fea] comparison plot → {path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FEA verification vs. analytical solutions")
    parser.add_argument("--no-smoke-test", action="store_true", help="Skip the 30-step optimizer smoke test")
    parser.add_argument("--out-dir", default="verify_fea", help="Output directory for plots and JSON")
    parser.add_argument("--mesh-spacing", type=float, default=5.0, help="FEA mesh spacing in mm")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FEA VERIFICATION — Analytical Solution Comparison")
    print("=" * 60)

    results = []

    r1 = test_solid_disk(R_mm=100.0, mesh_spacing=args.mesh_spacing)
    results.append(r1)

    r2 = test_mid_radius(R_mm=100.0, r_frac=0.70, mesh_spacing=args.mesh_spacing)
    results.append(r2)

    if not args.no_smoke_test:
        r3 = test_smoke()
        results.append(r3)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}]  {r['name']}")
        if not r["passed"]:
            all_passed = False

    print("=" * 60)
    print(f"  Overall: {'ALL PASS' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)

    # ── Plots ─────────────────────────────────────────────────────────────
    make_plots(results, out_dir)

    # ── JSON output ───────────────────────────────────────────────────────
    def _to_py(v):
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    json_results = [
        {k: _to_py(v) for k, v in r.items() if not k.startswith("_")}
        for r in results
    ]
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump({"overall_pass": bool(all_passed), "tests": json_results}, f, indent=2)
    print(f"\nResults saved to {json_path}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
