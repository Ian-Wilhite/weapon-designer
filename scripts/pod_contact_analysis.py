#!/usr/bin/env python3
"""Stratified POD analysis of FEA database, grouped by n_contacts.

For each contact-count group (nc = 1, 2, 3, …) in the FEA database:
  1. Assemble stress-field matrix  X  (N_group × N_ref_elements)
  2. Mean-centre and compute truncated SVD → POD basis
  3. Record variance explained, projection RMSE, group stats
  4. Compute pairwise subspace similarity (Grassmannian geodesic distance)
  5. Save per-group basis + summary JSON

Outputs (in --out-dir)
----------------------
  nc{n}_basis.npz   — POD basis, singular values, mean stress, params
  summary.json      — per-group stats + pairwise similarity matrix

Usage
-----
    python scripts/pod_contact_analysis.py
    python scripts/pod_contact_analysis.py --db-dir outputs/fea_database \\
        --out-dir outputs/pod_stratified --nc 1,2,3 --variance-threshold 0.99
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GroupData:
    nc: int
    stresses: np.ndarray          # (N, M_ref) float64
    params: np.ndarray            # (N, 8) float64
    safety_factors: np.ndarray    # (N,) float64 — may contain inf
    exterior_coords: list         # list of (M_i, 2) arrays
    design_files: list            # NPZ filenames for traceability
    best_idx: int = 0             # index of highest-SF design (finite SF only)


@dataclass
class PODResult:
    nc: int
    N: int
    basis: np.ndarray             # (M_ref, k_99) — right singular vectors
    singular_values: np.ndarray   # (k_99,)
    mean_stress: np.ndarray       # (M_ref,)
    variance_explained: np.ndarray  # cumulative, shape (k_99,), values in [0, 1]
    k_99: int
    k_95: int
    k_80: int
    var_3modes: float             # fraction of variance from first 3 modes
    proj_rmse_k3: float           # projection RMSE (MPa) with k=3 modes
    proj_rmse_k20: float          # projection RMSE with k=20 modes
    group_data: GroupData = field(repr=False)


@dataclass
class SimilarityMatrix:
    nc_labels: list
    principal_angles: dict        # (nc_a, nc_b) -> np.ndarray of angles (rad)
    geodesic_distance: np.ndarray # (n_groups, n_groups)
    chordal_distance: np.ndarray  # (n_groups, n_groups)
    k_used: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _proj_rmse(stresses: np.ndarray, basis_k: np.ndarray, mean_stress: np.ndarray) -> float:
    """Projection RMSE: reconstruct X with k modes, return RMSE in MPa.

    This is the *projection* RMSE (lower bound on true LOO error) — fast and
    consistent across groups.  It equals zero when k equals the rank of X.
    """
    X_c = stresses - mean_stress
    # Project onto basis_k: coefficients = X_c @ basis_k, reconstruct = coeff @ basis_k.T
    coeffs = X_c @ basis_k          # (N, k)
    recon  = coeffs @ basis_k.T     # (N, M_ref)
    residual = X_c - recon
    return float(np.sqrt(np.mean(residual ** 2)))


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_group_data(
    db_dir: Path,
    nc_values: list[int],
    min_group_size: int = 50,
    verbose: bool = True,
) -> dict[int, GroupData]:
    """Scan all design NPZ files and group by n_contacts.

    n_contacts is stored in individual NPZ files only (not in manifest.json).
    """
    db_dir = Path(db_dir)
    manifest_path = db_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in {db_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = manifest.get("designs", [])
    if not entries:
        raise RuntimeError("manifest.json has no designs")

    # Accumulators per nc value
    buckets: dict[int, dict] = {
        nc: {"stresses": [], "params": [], "sf": [], "ext": [], "files": []}
        for nc in nc_values
    }
    other_nc_counts: dict[int, int] = {}

    total = len(entries)
    n_loaded = 0
    t0 = time.perf_counter()
    for i, entry in enumerate(entries):
        fpath = db_dir / entry["file"]
        if not fpath.exists():
            continue
        try:
            data = np.load(fpath, allow_pickle=True)
            nc = int(data["n_contacts"])
            if nc in buckets:
                buckets[nc]["stresses"].append(data["vm_stresses_ref"])
                buckets[nc]["params"].append(data["params"])
                sf_val = float(data["safety_factor"])
                buckets[nc]["sf"].append(sf_val)
                buckets[nc]["ext"].append(np.array(data["exterior_coords"]))
                buckets[nc]["files"].append(entry["file"])
            else:
                other_nc_counts[nc] = other_nc_counts.get(nc, 0) + 1
            n_loaded += 1
        except Exception:
            continue

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [load] {i+1}/{total} designs scanned ({elapsed:.1f}s)")

    if verbose:
        print(f"  [load] done — {n_loaded}/{total} designs loaded in {time.perf_counter()-t0:.1f}s")
        for nc, cnt in sorted(other_nc_counts.items()):
            print(f"  [load] nc={nc}: {cnt} designs (not in requested groups)")

    # Build GroupData objects
    result: dict[int, GroupData] = {}
    for nc in nc_values:
        b = buckets[nc]
        N = len(b["stresses"])
        if N == 0:
            if verbose:
                print(f"  [warn] nc={nc}: no designs found — skipping")
            continue
        if N < min_group_size:
            if verbose:
                print(f"  [warn] nc={nc}: N={N} < min_group_size={min_group_size} — "
                      f"POD basis may be unreliable")

        stresses = np.array(b["stresses"], dtype=np.float64)   # (N, M_ref)
        params   = np.array(b["params"],   dtype=np.float64)   # (N, 8)
        sfs      = np.array(b["sf"],       dtype=np.float64)   # (N,)

        # Best design by safety factor (finite values only)
        finite_mask = np.isfinite(sfs)
        if finite_mask.any():
            best_idx = int(np.argmax(np.where(finite_mask, sfs, -np.inf)))
        else:
            best_idx = 0

        result[nc] = GroupData(
            nc=nc,
            stresses=stresses,
            params=params,
            safety_factors=sfs,
            exterior_coords=b["ext"],
            design_files=b["files"],
            best_idx=best_idx,
        )
        if verbose:
            print(f"  [group] nc={nc}: N={N}, best SF={sfs[best_idx]:.3f} "
                  f"({b['files'][best_idx]})")

    return result


def compute_group_pod(
    group_data: GroupData,
    variance_threshold: float = 0.99,
) -> PODResult:
    """Compute truncated SVD POD basis for one n_contacts group."""
    stresses    = group_data.stresses       # (N, M_ref)
    mean_stress = stresses.mean(axis=0)    # (M_ref,)
    X_c         = stresses - mean_stress   # centred (N, M_ref)

    # SVD — full_matrices=False is critical to avoid (N×N) U matrix
    _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    # Vt shape: (min(N, M_ref), M_ref); basis columns = Vt.T rows

    # Cumulative variance
    s2      = S ** 2
    cumvar  = np.cumsum(s2) / s2.sum()   # shape: (min(N, M_ref),)

    k_80 = int(np.searchsorted(cumvar, 0.80)) + 1
    k_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    k_99 = int(np.searchsorted(cumvar, variance_threshold)) + 1
    k_99 = min(k_99, len(S))

    # Basis: first k_99 right singular vectors, shape (M_ref, k_99)
    basis = Vt[:k_99].T   # (M_ref, k_99)

    var_3modes = float(cumvar[min(2, len(cumvar) - 1)])

    # Projection RMSE
    proj_rmse_k3  = _proj_rmse(stresses, basis[:, :min(3,  k_99)], mean_stress)
    proj_rmse_k20 = _proj_rmse(stresses, basis[:, :min(20, k_99)], mean_stress)

    return PODResult(
        nc=group_data.nc,
        N=len(stresses),
        basis=basis,
        singular_values=S[:k_99],
        mean_stress=mean_stress,
        variance_explained=cumvar[:k_99],
        k_99=k_99,
        k_95=k_95,
        k_80=k_80,
        var_3modes=var_3modes,
        proj_rmse_k3=proj_rmse_k3,
        proj_rmse_k20=proj_rmse_k20,
        group_data=group_data,
    )


def compute_subspace_similarity(
    pod_results: dict[int, PODResult],
    k_compare: int = 10,
) -> SimilarityMatrix:
    """Compute pairwise Grassmannian distances between POD subspaces."""
    labels = sorted(pod_results.keys())
    n      = len(labels)
    geo    = np.zeros((n, n))
    chord  = np.zeros((n, n))
    angles_dict: dict[tuple, np.ndarray] = {}

    for i, nc_a in enumerate(labels):
        for j, nc_b in enumerate(labels):
            if j <= i:
                continue
            ka = min(k_compare, pod_results[nc_a].basis.shape[1])
            kb = min(k_compare, pod_results[nc_b].basis.shape[1])
            k  = min(ka, kb)

            A = pod_results[nc_a].basis[:, :k]  # (M_ref, k)
            B = pod_results[nc_b].basis[:, :k]  # (M_ref, k)

            angs = np.clip(subspace_angles(A, B), 0.0, np.pi / 2)
            geo[i, j] = geo[j, i] = float(np.sqrt((angs ** 2).sum()))
            chord[i, j] = chord[j, i] = float(np.sqrt(np.sum(np.sin(angs) ** 2)))
            angles_dict[(nc_a, nc_b)] = angs

    return SimilarityMatrix(
        nc_labels=labels,
        principal_angles=angles_dict,
        geodesic_distance=geo,
        chordal_distance=chord,
        k_used=k_compare,
    )


def save_pod_basis(pod_result: PODResult, out_dir: Path) -> Path:
    """Save POD basis + metadata to nc{n}_basis.npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"nc{pod_result.nc}_basis.npz"
    np.savez_compressed(
        path,
        basis=pod_result.basis,
        singular_values=pod_result.singular_values,
        mean_stress=pod_result.mean_stress,
        params=pod_result.group_data.params,
        variance_explained=pod_result.variance_explained,
        safety_factors=pod_result.group_data.safety_factors,
        best_design_idx=np.array([pod_result.group_data.best_idx]),
        best_exterior_coords=pod_result.group_data.exterior_coords[pod_result.group_data.best_idx],
        best_vm_stresses=pod_result.group_data.stresses[pod_result.group_data.best_idx],
        k_99=np.array([pod_result.k_99]),
        k_95=np.array([pod_result.k_95]),
        k_80=np.array([pod_result.k_80]),
        N=np.array([pod_result.N]),
    )
    return path


def save_summary_json(
    pod_results: dict[int, PODResult],
    similarity: SimilarityMatrix,
    out_dir: Path,
) -> Path:
    """Save all group stats + similarity matrix to summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for nc, pr in pod_results.items():
        groups[str(nc)] = {
            "nc": nc,
            "N": pr.N,
            "k_99": pr.k_99,
            "k_95": pr.k_95,
            "k_80": pr.k_80,
            "var_3modes": pr.var_3modes,
            "proj_rmse_k3_mpa": pr.proj_rmse_k3,
            "proj_rmse_k20_mpa": pr.proj_rmse_k20,
            "variance_explained_first20": pr.variance_explained[:20].tolist(),
            "singular_values_first20": pr.singular_values[:20].tolist(),
            "best_design_file": pr.group_data.design_files[pr.group_data.best_idx],
            "best_sf": float(pr.group_data.safety_factors[pr.group_data.best_idx]),
        }

    # Similarity matrix
    labels = similarity.nc_labels
    sim_block = {
        "k_used": similarity.k_used,
        "nc_labels": labels,
        "geodesic_distance": similarity.geodesic_distance.tolist(),
        "chordal_distance": similarity.chordal_distance.tolist(),
        "pairs": {},
    }
    for (nc_a, nc_b), angs in similarity.principal_angles.items():
        sim_block["pairs"][f"nc{nc_a}_vs_nc{nc_b}"] = {
            "principal_angles_rad": angs.tolist(),
            "mean_angle_deg": float(np.degrees(angs.mean())),
            "geodesic_distance": float(np.sqrt((angs ** 2).sum())),
        }

    summary = {"groups": groups, "similarity": sim_block}
    path = out_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stratified POD analysis of FEA database")
    parser.add_argument("--db-dir",  default=str(ROOT / "outputs" / "fea_database"),
                        help="FEA database directory")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "pod_stratified"),
                        help="Output directory")
    parser.add_argument("--nc",      default="1,2,3,4",
                        help="Comma-separated n_contacts groups to analyse")
    parser.add_argument("--variance-threshold", type=float, default=0.99,
                        help="Variance threshold for k_99 selection")
    parser.add_argument("--k-compare", type=int, default=10,
                        help="Number of modes for subspace similarity comparison")
    parser.add_argument("--min-group-size", type=int, default=50,
                        help="Minimum group size to warn on (still computed)")
    args = parser.parse_args()

    db_dir  = Path(args.db_dir)
    out_dir = Path(args.out_dir)
    nc_values = [int(x.strip()) for x in args.nc.split(",")]

    print("=" * 60)
    print("  STRATIFIED POD ANALYSIS")
    print(f"  Database:  {db_dir}")
    print(f"  Output:    {out_dir}")
    print(f"  Groups:    nc ∈ {nc_values}")
    print(f"  Threshold: {args.variance_threshold*100:.0f}% variance")
    print("=" * 60)

    # ── Load & group ──────────────────────────────────────────────────────
    print("\n[1/3] Loading database and grouping by n_contacts ...")
    groups = load_group_data(db_dir, nc_values, min_group_size=args.min_group_size)

    if not groups:
        print("[ERROR] No groups found — check database path and --nc values")
        sys.exit(1)

    # ── SVD per group ─────────────────────────────────────────────────────
    print("\n[2/3] Computing POD bases ...")
    pod_results: dict[int, PODResult] = {}
    for nc, gdata in sorted(groups.items()):
        t0 = time.perf_counter()
        print(f"  nc={nc}: N={len(gdata.stresses)}, M_ref={gdata.stresses.shape[1]} ... ", end="", flush=True)
        pr = compute_group_pod(gdata, variance_threshold=args.variance_threshold)
        dt = time.perf_counter() - t0
        pod_results[nc] = pr
        path = save_pod_basis(pr, out_dir)
        print(f"k_99={pr.k_99}, var_3={pr.var_3modes*100:.1f}%, "
              f"RMSE_k3={pr.proj_rmse_k3:.2f}MPa  ({dt:.1f}s) → {path.name}")

    # ── Subspace similarity ────────────────────────────────────────────────
    print("\n[3/3] Computing subspace similarity ...")
    # Use only groups with enough modes for meaningful comparison
    sim_groups = {nc: pr for nc, pr in pod_results.items() if pr.N >= args.min_group_size}
    similarity = compute_subspace_similarity(sim_groups, k_compare=args.k_compare)
    labels = similarity.nc_labels
    print(f"  Grassmannian geodesic distances (k={args.k_compare} modes):")
    header = "      " + "".join(f"  nc={nc:d}" for nc in labels)
    print(header)
    for i, nc_a in enumerate(labels):
        row = f"  nc={nc_a}: "
        for j, _ in enumerate(labels):
            row += f"  {similarity.geodesic_distance[i,j]:5.3f}"
        print(row)

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary_path = save_summary_json(pod_results, similarity, out_dir)
    print(f"\n  summary → {summary_path}")

    # ── Print table ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'nc':>3}  {'N':>5}  {'k@99%':>6}  {'k@95%':>6}  "
          f"{'var_3modes%':>11}  {'RMSE_k3 MPa':>11}  {'RMSE_k20 MPa':>12}")
    print("  " + "-" * 66)
    for nc, pr in sorted(pod_results.items()):
        print(f"  {nc:>3}  {pr.N:>5}  {pr.k_99:>6}  {pr.k_95:>6}  "
              f"{pr.var_3modes*100:>11.1f}  {pr.proj_rmse_k3:>11.2f}  {pr.proj_rmse_k20:>12.2f}")
    print("=" * 70)
    print(f"\nOutputs in {out_dir}")


if __name__ == "__main__":
    main()
