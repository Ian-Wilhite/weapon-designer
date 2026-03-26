#!/usr/bin/env python3
"""FEA speed sweep for Archimedean spiral weapon archetypes.

Selects the best-SF design from each n_starts group (1, 2, 3, 4) stored in
the spiral POD cache, then sweeps RPM from rpm_min to an estimated failure
point and records:

  - peak_stress_mpa, safety_factor   at each RPM
  - impact force (centripetal model: F = ½·m·ω²·r_contact)
  - von-Mises stress snapshots at 5 representative RPMs

No B-spline, no control points.  Weapons are built with build_spiral_weapon().

⚠  MAJOR COMPUTE: 4 archetypes × ~200 RPM steps × FEA ≈ 800+ calls (~30 min).
   Prints the run plan and waits for confirmation unless --yes is passed.

Outputs (in --out-dir)
----------------------
  n{k}_sweep.npz  (k = 1, 2, 3, 4)
      rpms, sf_vals, peak_stress_mpa, f_impact_n,
      snap_indices, snap_rpms, snap_stresses (n_snaps × M_elem),
      snap_nodes, snap_elements,
      params (7,), exterior_coords, mass_kg, I_kg_m2, m_opponent_kg,
      t_contact_s, v_approach_ms, n_starts
  archetypes.json  — metadata for all selected archetypes

Usage
-----
    python scripts/fea_archetype_sweep.py
    python scripts/fea_archetype_sweep.py --rpm-step 100 --yes
    python scripts/fea_archetype_sweep.py \\
        --cache-dir outputs/spiral_pod_cache_v1 \\
        --out-dir   outputs/spiral_speed_sweep  \\
        --rpm-step 50 --sf-min 1.5 --yes
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

from weapon_designer.config import (
    WeaponConfig, Material, Mounting, Envelope, OptimizationParams,
)
from weapon_designer.geometry import assemble_weapon
from weapon_designer.fea import fea_stress_analysis, fea_stress_analysis_with_mesh
from weapon_designer.physics import polygon_mass_kg, mass_moi_kg_mm2
from weapon_designer.spiral_weapon import build_spiral_weapon
from weapon_designer.spiral_contact import analyse_contacts, contact_forces as spiral_contact_forces

CACHE_DIR = ROOT / "outputs" / "spiral_pod_cache_v2"
OUT_DIR   = ROOT / "outputs" / "spiral_speed_sweep"

# Opponent mass used in the reduced-mass impact model (kg).
# Typical featherweight class opponent.  Override with --m-opponent.
M_OPPONENT_KG_DEFAULT = 0.68   # ~1.5 lb featherweight

# Contact time for impulse→force conversion (seconds).
# ~1 ms is typical for hardened steel-on-steel impact in combat robotics.
T_CONTACT_S_DEFAULT = 1e-3


# ---------------------------------------------------------------------------
# Config — must match figure_functional_pod_table.py exactly
# ---------------------------------------------------------------------------

def _make_cfg(rpm: float = 10000) -> WeaponConfig:
    return WeaponConfig(
        material=Material(
            name="S7_Tool_Steel", density_kg_m3=7750,
            yield_strength_mpa=1600, hardness_hrc=56,
        ),
        weapon_style="disk",
        sheet_thickness_mm=6.0,
        weight_budget_kg=0.5,
        rpm=rpm,
        mounting=Mounting(
            bore_diameter_mm=12.0, bolt_circle_diameter_mm=25,
            num_bolts=3, bolt_hole_diameter_mm=4.0,
        ),
        envelope=Envelope(max_radius_mm=100.0),
        optimization=OptimizationParams(
            fea_coarse_spacing_mm=10.0,
        ),
    )


# ---------------------------------------------------------------------------
# Archetype selection from spiral POD cache
# ---------------------------------------------------------------------------

def select_archetypes(
    cache_dir: Path,
    sf_min: float = 1.5,
    verbose: bool = True,
) -> list[dict]:
    """Pick the highest-SF valid design from each n_starts group in the cache.

    Cache files: n{k}_N*.npz  (k = 1, 2, 3, 4)
    Each file contains: stresses, sf_vals, params (N, 7), n_starts_fixed.

    Returns a list of dicts (one per n_starts group), sorted by n_starts.
    """
    cache_dir = Path(cache_dir)
    archetypes = []

    for n_starts in [1, 2, 3, 4]:
        # Find cache file for this group (n{k}_N*.npz)
        matches = sorted(cache_dir.glob(f"n{n_starts}_N*.npz"))
        if not matches:
            if verbose:
                print(f"  [warn] no cache file found for n_starts={n_starts}, skipping")
            continue

        data = np.load(matches[-1])   # take largest N if multiple
        sf_vals = data["sf_vals"]
        params  = data["params"]      # (N, 7)

        # Best finite SF above threshold
        finite = np.isfinite(sf_vals) & (sf_vals >= sf_min)
        if not finite.any():
            # Try inf SF (structurally trivial but still valid)
            finite = (sf_vals >= sf_min) | ~np.isfinite(sf_vals)
        if not finite.any():
            if verbose:
                print(f"  [warn] n_starts={n_starts}: no design meets SF≥{sf_min}")
            continue

        best_i  = int(np.where(finite)[0][np.argmax(
            np.where(np.isfinite(sf_vals[finite]), sf_vals[finite], 1e6)
        )])
        best_sf = float(sf_vals[best_i]) if np.isfinite(sf_vals[best_i]) else 1e6
        p       = params[best_i]

        # Rebuild weapon to get mass and exterior coords
        cfg_ref = _make_cfg()
        weapon  = build_spiral_weapon(p, cfg_ref)
        if weapon is None:
            if verbose:
                print(f"  [warn] n_starts={n_starts}: best design rebuilt as None, skipping")
            continue
        assembled = assemble_weapon(weapon, cfg_ref.mounting)
        mass_kg   = polygon_mass_kg(
            assembled,
            cfg_ref.sheet_thickness_mm,
            cfg_ref.material.density_kg_m3,
        )

        archetypes.append({
            "n_starts":        n_starts,
            "params":          p,
            "safety_factor":   best_sf,
            "mass_kg":         mass_kg,
            "exterior_coords": np.array(weapon.exterior.coords),
            "cache_file":      str(matches[-1].name),
            "cache_idx":       best_i,
        })

    if verbose:
        print(f"  Selected {len(archetypes)} archetypes:")
        for a in archetypes:
            sf_str = f"{a['safety_factor']:.3f}" if a["safety_factor"] < 1e5 else "∞"
            print(f"    [n_starts={a['n_starts']}]  "
                  f"SF={sf_str}  mass={a['mass_kg']:.3f}kg  "
                  f"pitch={a['params'][0]:.1f}mm  "
                  f"(cache idx {a['cache_idx']})")

    return archetypes


# ---------------------------------------------------------------------------
# Single archetype RPM sweep
# ---------------------------------------------------------------------------

def sweep_archetype(
    archetype: dict,
    rpm_step: int    = 50,
    rpm_min:  int    = 200,
    rpm_max_cap: int = 25000,
    m_opponent_kg: float = M_OPPONENT_KG_DEFAULT,
    t_contact_s:   float = T_CONTACT_S_DEFAULT,
    v_approach_ms: float = 3.0,
    verbose: bool = True,
) -> dict:
    """Run FEA at a range of RPM values for one spiral archetype.

    FEA at each RPM step includes TWO loading sources:
      1. Centrifugal body force  — from rpm parameter (scales as ω²)
      2. Impact contact forces   — from opponent collision (scales as ω via v_tip)
         F_impact = μ × v_tip / t_contact
         where μ = reduced_mass = m_eff × m_opp / (m_eff + m_opp)
               m_eff = I / r_contact²   (effective mass at contact point)

    RPM upper bound: min(rpm_ref × sqrt(SF_ref) × 1.15, rpm_max_cap).
    """
    rpm_ref  = 10000
    cfg_ref  = _make_cfg(rpm=rpm_ref)
    params   = np.array(archetype["params"])
    n_starts = int(archetype["n_starts"])

    # Build weapon (geometry fixed — only loading changes with RPM)
    weapon = build_spiral_weapon(params, cfg_ref)
    if weapon is None or not weapon.is_valid:
        raise ValueError(f"Invalid polygon for n_starts={n_starts}")
    assembled = assemble_weapon(weapon, cfg_ref.mounting)

    mat    = cfg_ref.material
    bore_d = cfg_ref.mounting.bore_diameter_mm
    mesh_sp = cfg_ref.optimization.fea_coarse_spacing_mm

    # MOI — geometry-only, RPM-independent — used for effective mass
    I_kg_m2 = mass_moi_kg_mm2(
        assembled, cfg_ref.sheet_thickness_mm, mat.density_kg_m3,
    ) * 1e-6   # kg·mm² → kg·m²

    # Estimate RPM at failure and build RPM array
    sf_ref       = min(archetype["safety_factor"], 100.0)
    rpm_fail_est = int(rpm_ref * np.sqrt(max(sf_ref, 0.01)))
    rpm_upper    = min(int(rpm_fail_est * 1.15), rpm_max_cap)
    rpms = np.arange(max(rpm_min, 1), rpm_upper + rpm_step, rpm_step, dtype=int)

    if verbose:
        print(f"    RPM range: {rpm_min}–{rpm_upper} (step {rpm_step})  "
              f"→ {len(rpms)} FEA calls  (I={I_kg_m2*1e6:.1f} g·m²)")

    sf_vals       = np.zeros(len(rpms))
    peak_stress   = np.zeros(len(rpms))
    f_impact_vals = np.zeros(len(rpms))

    t0 = time.perf_counter()
    for i, rpm_i in enumerate(rpms):
        omega = 2.0 * np.pi * float(rpm_i) / 60.0

        # ── Contact analysis at this RPM ──────────────────────────────────
        try:
            contacts_i, _ = analyse_contacts(
                weapon, n_spirals=6, v_ms=v_approach_ms,
                rpm=float(rpm_i), n_eval=360,
            )
        except Exception:
            contacts_i = []

        if contacts_i:
            r_contact_m = float(np.mean([c.r_contact for c in contacts_i])) * 1e-3
        else:
            r_contact_m = float(weapon.bounds[2]) * 0.45e-3   # fallback: 45% of max_r

        # ── Impact force magnitude (reduced-mass impulse model) ───────────
        m_eff  = I_kg_m2 / max(r_contact_m ** 2, 1e-6)
        mu     = m_eff * m_opponent_kg / (m_eff + m_opponent_kg)
        v_tip  = omega * r_contact_m
        f_mag  = mu * v_tip / t_contact_s    # Newtons
        f_impact_vals[i] = f_mag

        # ── FEA: centrifugal + contact forces combined ────────────────────
        fea_forces = (spiral_contact_forces(contacts_i, f_mag, scale_by_angle=True)
                      if contacts_i else None)
        try:
            res = fea_stress_analysis(
                assembled,
                rpm=float(rpm_i),
                density_kg_m3=mat.density_kg_m3,
                thickness_mm=cfg_ref.sheet_thickness_mm,
                yield_strength_mpa=mat.yield_strength_mpa,
                bore_diameter_mm=bore_d,
                mesh_spacing=mesh_sp,
                contact_forces=fea_forces,
            )
            sf_vals[i]     = res["safety_factor"]
            peak_stress[i] = res["peak_stress_mpa"]
        except Exception as exc:
            if verbose:
                print(f"    [warn] RPM={rpm_i}: {exc}")
            sf_vals[i]     = 0.0
            peak_stress[i] = float("inf")

        if verbose and (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    [{i+1:>3}/{len(rpms)}]  RPM={rpm_i:>6}  "
                  f"SF={sf_vals[i]:.3f}  F_impact={f_impact_vals[i]:.0f}N  "
                  f"({elapsed:.0f}s)")

    # ── Select 5 snapshot RPM indices ─────────────────────────────────────
    def _nearest_sf(target: float) -> int:
        valid = np.where((sf_vals > 0) & np.isfinite(sf_vals))[0]
        if len(valid) == 0:
            return len(rpms) // 2
        return int(valid[np.argmin(np.abs(sf_vals[valid] - target))])

    snap_indices = np.array(sorted(set([
        0,
        len(rpms) // 4,
        len(rpms) // 2,
        _nearest_sf(2.0),
        _nearest_sf(1.0),
    ])), dtype=int)
    snap_rpms = rpms[snap_indices]

    # Stress snapshots with full mesh (combined loading at each snapshot RPM)
    if verbose:
        print(f"    Snapshot solves at RPMs {snap_rpms.tolist()} ...")
    snap_stresses_list = []
    snap_nodes = snap_elements = None

    for rpm_i in snap_rpms.tolist():
        omega_i = 2.0 * np.pi * float(rpm_i) / 60.0
        try:
            contacts_s, _ = analyse_contacts(
                weapon, n_spirals=6, v_ms=v_approach_ms,
                rpm=float(rpm_i), n_eval=360,
            )
        except Exception:
            contacts_s = []

        if contacts_s:
            rc_m   = float(np.mean([c.r_contact for c in contacts_s])) * 1e-3
            m_eff_s = I_kg_m2 / max(rc_m**2, 1e-6)
            mu_s    = m_eff_s * m_opponent_kg / (m_eff_s + m_opponent_kg)
            f_s     = mu_s * omega_i * rc_m / t_contact_s
            snap_forces = spiral_contact_forces(contacts_s, f_s, scale_by_angle=True)
        else:
            snap_forces = None

        try:
            res = fea_stress_analysis_with_mesh(
                assembled,
                rpm=float(rpm_i),
                density_kg_m3=mat.density_kg_m3,
                thickness_mm=cfg_ref.sheet_thickness_mm,
                yield_strength_mpa=mat.yield_strength_mpa,
                bore_diameter_mm=bore_d,
                mesh_spacing=mesh_sp,
                contact_forces=snap_forces,
            )
            snap_stresses_list.append(np.array(res["vm_stresses"]))
            if snap_nodes is None:
                snap_nodes    = res["nodes"]
                snap_elements = res["elements"]
        except Exception as exc:
            if verbose:
                print(f"    [warn] snapshot RPM={rpm_i}: {exc}")
            snap_stresses_list.append(np.array([]))

    max_len = max((len(s) for s in snap_stresses_list if len(s) > 0), default=1)
    snap_stresses = np.full((len(snap_indices), max_len), np.nan)
    for k, sv in enumerate(snap_stresses_list):
        if len(sv) > 0:
            m = min(len(sv), max_len)
            snap_stresses[k, :m] = sv[:m]

    return {
        "n_starts":        n_starts,
        "params":          params,
        "exterior_coords": np.array(archetype["exterior_coords"]),
        "mass_kg":         archetype["mass_kg"],
        "I_kg_m2":         I_kg_m2,
        "m_opponent_kg":   m_opponent_kg,
        "t_contact_s":     t_contact_s,
        "v_approach_ms":   v_approach_ms,
        "rpm_ref":         rpm_ref,
        "rpm_fail_est":    rpm_fail_est,
        "rpms":            rpms,
        "sf_vals":         sf_vals,
        "peak_stress_mpa": peak_stress,
        "f_impact_n":      f_impact_vals,
        "snap_indices":    snap_indices,
        "snap_rpms":       snap_rpms,
        "snap_stresses":   snap_stresses,
        "snap_nodes":      snap_nodes if snap_nodes is not None else np.zeros((0, 2)),
        "snap_elements":   (snap_elements.astype(np.int32)
                            if snap_elements is not None else np.zeros((0, 3), dtype=np.int32)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FEA RPM sweep for Archimedean spiral weapon archetypes")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR),
                        help="Spiral POD cache directory")
    parser.add_argument("--out-dir",   default=str(OUT_DIR),
                        help="Output directory for sweep NPZ files")
    parser.add_argument("--rpm-step",   type=int,   default=50)
    parser.add_argument("--rpm-max",    type=int,   default=25000)
    parser.add_argument("--sf-min",     type=float, default=1.5)
    parser.add_argument("--m-opponent", type=float, default=M_OPPONENT_KG_DEFAULT,
                        help="Opponent mass in kg (default: 0.68 kg = ~1.5 lb featherweight)")
    parser.add_argument("--t-contact",  type=float, default=T_CONTACT_S_DEFAULT,
                        help="Contact duration in seconds for impulse→force conversion (default: 1e-3)")
    parser.add_argument("--v-approach", type=float, default=3.0,
                        help="Opponent approach speed in m/s for contact analysis (default: 3.0)")
    parser.add_argument("--yes",        action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  SPIRAL WEAPON FEA SWEEP — RPM vs. impact force")
    print(f"  Cache:    {cache_dir}")
    print(f"  Output:   {out_dir}")
    print(f"  RPM step: {args.rpm_step}  |  SF min: {args.sf_min}")
    print("=" * 65)

    print("\n[1/3] Selecting archetypes from spiral cache ...")
    archetypes = select_archetypes(cache_dir, sf_min=args.sf_min)
    if not archetypes:
        print("[ERROR] No archetypes found. Run figure_functional_pod_table.py first.")
        sys.exit(1)

    # Estimate compute load
    total_fea = 0
    for a in archetypes:
        sf_r      = min(a["safety_factor"], 100.0)
        rpm_fail  = int(10000 * np.sqrt(max(sf_r, 0.01)))
        rpm_upper = min(int(rpm_fail * 1.15), args.rpm_max)
        n_rpms    = len(np.arange(200, rpm_upper + args.rpm_step, args.rpm_step))
        total_fea += n_rpms + 5   # +5 snapshot solves
        print(f"  n_starts={a['n_starts']}: RPM 500–{rpm_upper} "
              f"→ ~{n_rpms}+5 FEA calls")

    est_min = total_fea * 2.5 / 60
    print(f"\n  Total estimated FEA calls: ~{total_fea}")
    print(f"  Estimated wall time:       ~{est_min:.0f} min (10mm mesh)")

    if not args.yes:
        ans = input("\n  Proceed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("  Aborted.")
            sys.exit(0)

    # Save archetype metadata
    meta = []
    for a in archetypes:
        meta.append({
            "n_starts":      a["n_starts"],
            "safety_factor": a["safety_factor"] if a["safety_factor"] < 1e5 else None,
            "mass_kg":       a["mass_kg"],
            "cache_file":    a.get("cache_file", ""),
            "cache_idx":     int(a.get("cache_idx", -1)),
            "params":        a["params"].tolist(),
        })
    with open(out_dir / "archetypes.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[2/3] Running FEA sweeps ...")
    for i, a in enumerate(archetypes):
        tag = f"n{a['n_starts']}"
        print(f"\n  [{i+1}/{len(archetypes)}] n_starts={a['n_starts']}  "
              f"mass={a['mass_kg']:.3f}kg  "
              f"pitch={a['params'][0]:.1f}mm")
        t0 = time.perf_counter()
        try:
            sw = sweep_archetype(
                a,
                rpm_step=args.rpm_step,
                rpm_max_cap=args.rpm_max,
                m_opponent_kg=args.m_opponent,
                t_contact_s=args.t_contact,
                v_approach_ms=args.v_approach,
                verbose=True,
            )
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            continue

        npz_path = out_dir / f"{tag}_sweep.npz"
        np.savez_compressed(
            npz_path,
            rpms            = sw["rpms"],
            sf_vals         = sw["sf_vals"],
            peak_stress_mpa = sw["peak_stress_mpa"],
            f_impact_n      = sw["f_impact_n"],
            snap_indices    = sw["snap_indices"],
            snap_rpms       = sw["snap_rpms"],
            snap_stresses   = sw["snap_stresses"],
            snap_nodes      = sw["snap_nodes"],
            snap_elements   = sw["snap_elements"],
            params          = sw["params"],
            exterior_coords = sw["exterior_coords"],
            mass_kg         = np.array([sw["mass_kg"]]),
            I_kg_m2         = np.array([sw["I_kg_m2"]]),
            m_opponent_kg   = np.array([sw["m_opponent_kg"]]),
            t_contact_s     = np.array([sw["t_contact_s"]]),
            v_approach_ms   = np.array([sw["v_approach_ms"]]),
            n_starts        = np.array([sw["n_starts"]]),
            rpm_ref         = np.array([sw["rpm_ref"]]),
            rpm_fail_est    = np.array([sw["rpm_fail_est"]]),
        )
        dt    = time.perf_counter() - t0
        valid = sw["sf_vals"][sw["sf_vals"] > 0]
        sf_rng = f"{valid.min():.2f}–{valid.max():.2f}" if len(valid) else "N/A"
        print(f"  Done in {dt/60:.1f}min  SF range {sf_rng}  → {npz_path.name}")

    print(f"\n[3/3] All outputs saved to {out_dir}")
    print("  Next: python docs/figures/figure_speed_pareto.py "
          f"--sweep-dir {out_dir}")


if __name__ == "__main__":
    main()
