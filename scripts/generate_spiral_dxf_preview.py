#!/usr/bin/env python3
"""Generate ~20 Sobol-separated spiral_weapon designs as DXF + PNG previews.

Samples the 6-parameter spiral_weapon space using a Sobol sequence (well-
spread low-discrepancy), builds each design, exports DXF and a PNG thumbnail,
and prints a summary table.

Usage
-----
    python scripts/generate_spiral_dxf_preview.py
    python scripts/generate_spiral_dxf_preview.py --n 20 --out spiral_preview
    python scripts/generate_spiral_dxf_preview.py --n 20 --case featherweight_disk
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from weapon_designer.config import (
    WeaponConfig, Material, Mounting, Envelope,
    OptimizationWeights, OptimizationParams,
)
from weapon_designer.spiral_weapon import build_spiral_weapon, get_spiral_weapon_bounds
from weapon_designer.geometry import assemble_weapon
from weapon_designer.exporter import export_weapon_dxf


# ---------------------------------------------------------------------------
# Pre-defined cases (mirror evaluate.py)
# ---------------------------------------------------------------------------

def _make_featherweight() -> WeaponConfig:
    return WeaponConfig(
        material=Material(name="S7_Tool_Steel", density_kg_m3=7750,
                          yield_strength_mpa=1600, hardness_hrc=56),
        weapon_style="disk",
        sheet_thickness_mm=6,
        weight_budget_kg=1.5,
        rpm=12000,
        mounting=Mounting(bore_diameter_mm=12.0, bolt_circle_diameter_mm=25,
                          num_bolts=3, bolt_hole_diameter_mm=4.0),
        envelope=Envelope(max_radius_mm=80),
        optimization=OptimizationParams(
            weights=OptimizationWeights(
                moment_of_inertia=0.35, bite=0.10, structural_integrity=0.20,
                mass_utilization=0.10, balance=0.10, impact_zone=0.15,
            ),
            num_fourier_terms=4, num_cutout_pairs=2,
            max_iterations=200, population_size=60,
        ),
    )


def _make_heavyweight() -> WeaponConfig:
    return WeaponConfig(
        material=Material(name="AR500", density_kg_m3=7850,
                          yield_strength_mpa=1400, hardness_hrc=50),
        weapon_style="disk",
        sheet_thickness_mm=12,
        weight_budget_kg=4.5,
        rpm=8000,
        mounting=Mounting(bore_diameter_mm=20.0, bolt_circle_diameter_mm=40,
                          num_bolts=4, bolt_hole_diameter_mm=6.0),
        envelope=Envelope(max_radius_mm=110),
        optimization=OptimizationParams(
            weights=OptimizationWeights(
                moment_of_inertia=0.35, bite=0.10, structural_integrity=0.20,
                mass_utilization=0.10, balance=0.10, impact_zone=0.15,
            ),
            num_fourier_terms=4, num_cutout_pairs=2,
            max_iterations=200, population_size=60,
        ),
    )


CASES = {
    "featherweight_disk": _make_featherweight,
    "heavyweight_disk":   _make_heavyweight,
}


# ---------------------------------------------------------------------------
# Sobol sampling
# ---------------------------------------------------------------------------

def sobol_samples(n: int, d: int) -> np.ndarray:
    """Return (n, d) array of Sobol samples in [0, 1]^d.

    Uses scipy.stats.qmc.Sobol if available, else falls back to
    plain random (still prints a warning).
    """
    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=d, scramble=True, seed=0)
        m = 1
        while 2**m < n:
            m += 1
        raw = sampler.random_base2(m)[:n]
        return raw
    except Exception:
        print("WARNING: scipy.stats.qmc unavailable — using pseudo-random fallback")
        rng = np.random.default_rng(0)
        return rng.random((n, d))


# ---------------------------------------------------------------------------
# PNG thumbnail
# ---------------------------------------------------------------------------

def _save_png(weapon, assem, cfg, out_path: Path, design_id: int, params: np.ndarray):
    """Save a simple matplotlib thumbnail of the weapon polygon."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly
        from matplotlib.collections import PatchCollection

        fig, ax = plt.subplots(figsize=(5, 5))

        # Draw assembled weapon (with bore + bolt holes)
        poly = assem
        if poly.geom_type == "Polygon":
            polys_to_draw = [poly]
        else:
            polys_to_draw = list(poly.geoms)

        for p in polys_to_draw:
            ext_xy = np.array(p.exterior.coords)
            ax.fill(ext_xy[:, 0], ext_xy[:, 1], color="#aabbcc", alpha=0.9)
            for ring in p.interiors:
                int_xy = np.array(ring.coords)
                ax.fill(int_xy[:, 0], int_xy[:, 1], color="white")

        R = float(cfg.envelope.max_radius_mm)
        ax.set_xlim(-R * 1.1, R * 1.1)
        ax.set_ylim(-R * 1.1, R * 1.1)
        ax.set_aspect("equal")
        ax.axis("off")

        param_names = ["pitch", "n_st", "t_rim", "t_hub", "n_sup", "t_sup", "r_fil"]
        param_str = "  ".join(f"{n}={v:.1f}" for n, v in zip(param_names, params))
        ax.set_title(f"Design {design_id:02d}\n{param_str}", fontsize=7)

        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  [png] failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate spiral weapon DXF previews")
    parser.add_argument("--n",    type=int, default=20,
                        help="Number of Sobol designs (default: 20)")
    parser.add_argument("--out",  default="spiral_preview",
                        help="Output directory (default: spiral_preview)")
    parser.add_argument("--case", default="featherweight_disk",
                        choices=list(CASES.keys()),
                        help="Weapon config case (default: featherweight_disk)")
    args = parser.parse_args()

    cfg = CASES[args.case]()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = get_spiral_weapon_bounds(cfg)
    d = len(bounds)  # 6
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    raw = sobol_samples(args.n, d)                  # (n, 6) in [0,1]
    params_matrix = lo + raw * (hi - lo)             # scale to physical bounds

    print(f"\nSpiral weapon preview — {args.case} — {args.n} Sobol designs")
    print(f"Output: {out_dir.resolve()}")
    print(f"R_outer = {cfg.envelope.max_radius_mm} mm  "
          f"R_bore = {cfg.mounting.bore_diameter_mm/2:.1f} mm\n")

    hdr = f"{'#':>3}  {'pitch':>7}  {'n_st':>5}  {'t_rim':>6}  {'t_hub':>6}  {'n_sup':>6}  "
    hdr += f"{'t_sup':>6}  {'r_fil':>6}  {'area_mm2':>10}  {'mass_frac':>10}  {'status'}"
    print(hdr)
    print("-" * len(hdr))

    n_ok = 0
    for i, params in enumerate(params_matrix):
        design_id = i + 1
        # New order: [pitch, n_starts, t_rim, t_hub, n_supports, t_support, r_fillet]
        spiral_pitch, n_starts_f, t_rim, t_hub, n_sup_f, t_support, r_fillet = params

        weapon = build_spiral_weapon(params, cfg)
        if weapon is None:
            print(f"{design_id:3d}  {spiral_pitch:7.2f}  {n_starts_f:5.1f}  {t_rim:6.2f}  "
                  f"{t_hub:6.2f}  {n_sup_f:6.1f}  {t_support:6.2f}  {r_fillet:6.2f}  "
                  f"{'N/A':>10}  {'N/A':>10}  DEGENERATE")
            continue

        # Assemble: subtract bore + bolt holes
        assem = assemble_weapon(weapon, cfg.mounting)

        solid_area = math.pi * cfg.envelope.max_radius_mm ** 2
        mass_frac  = weapon.area / solid_area

        stem = f"spiral_{design_id:02d}"

        # DXF export
        dxf_path = export_weapon_dxf(assem, cfg, out_dir, stem)
        dxf_ok = "DXF" if dxf_path else "dxf-FAIL"

        # PNG thumbnail
        png_path = out_dir / f"{stem}.png"
        _save_png(weapon, assem, cfg, png_path, design_id, params)

        print(f"{design_id:3d}  {spiral_pitch:7.2f}  {n_starts_f:5.1f}  {t_rim:6.2f}  "
              f"{t_hub:6.2f}  {n_sup_f:6.1f}  {t_support:6.2f}  {r_fillet:6.2f}  "
              f"{weapon.area:10.0f}  {mass_frac:10.3f}  {dxf_ok}")
        n_ok += 1

    print(f"\n{n_ok}/{args.n} designs built successfully → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
