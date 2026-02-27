#!/usr/bin/env python3
"""
Diversity Explorer — generate maximally diverse weapon designs.

Uses three exploration strategies (Voronoi farthest-point, RRT*-style
expansion, MAP-Elites targeted generation) feeding a shared archive with
MAP-Elites grid management.  Produces a family of meaningfully different
designs that are still reasonably performant.

Usage:
    python explore_diverse.py configs/example_disk.json [options]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from weapon_designer.archetypes import archetype_to_params, get_archetypes
from weapon_designer.config import WeaponConfig, load_config
from weapon_designer.constraints import (
    check_envelope,
    constraint_penalty,
    validate_geometry,
)
from weapon_designer.exporter import export_dxf
from weapon_designer.geometry import assemble_weapon
from weapon_designer.objectives import compute_metrics, weighted_score
from weapon_designer.optimizer import _get_profile_bounds
from weapon_designer.parametric import build_weapon_polygon, _cutout_stride

log = logging.getLogger("explore_diverse")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ArchiveMember:
    """A single design in the diversity archive."""
    params: np.ndarray
    score: float
    penalty: float
    metrics: dict[str, Any]
    behavior: np.ndarray          # 4-D normalized behavior vector
    cell: tuple[int, ...]         # MAP-Elites grid cell
    generation_method: str
    iteration_found: int
    design_id: int = 0


# ---------------------------------------------------------------------------
# Evaluation (top-level for pickle compatibility)
# ---------------------------------------------------------------------------

def evaluate_candidate(
    args: tuple[np.ndarray, dict],
) -> dict[str, Any] | None:
    """Evaluate a single candidate parameter vector.

    Returns a result dict or None on failure.  Top-level function so
    multiprocessing can pickle it.
    """
    params, cfg_dict = args
    cfg = _rebuild_config(cfg_dict)

    try:
        # Pad with zero cutout params (Phase-1 style: profile only)
        C = cfg.optimization.num_cutout_pairs
        S = _cutout_stride(cfg)
        x_full = np.concatenate([params, np.zeros(C * S)])

        outer, _, cutout_polys = build_weapon_polygon(x_full, cfg)
        outer = validate_geometry(outer)

        if outer.is_empty:
            return None
        if isinstance(outer, MultiPolygon):
            return None

        weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
        weapon = validate_geometry(weapon)
        if weapon.is_empty:
            return None

        metrics = compute_metrics(weapon, cfg)
        score = weighted_score(metrics, cfg)
        pen = constraint_penalty(weapon, cfg)

        return {
            "params": params,
            "score": score,
            "penalty": pen,
            "metrics": metrics,
        }
    except Exception:
        return None


def _rebuild_config(d: dict) -> WeaponConfig:
    """Reconstruct a WeaponConfig from a plain dict (for worker processes)."""
    from weapon_designer.config import (
        Envelope,
        Material,
        Mounting,
        OptimizationParams,
        OptimizationWeights,
        OutputParams,
    )

    mat = Material(**d["material"]) if isinstance(d.get("material"), dict) else d.get("material", Material())
    mount = Mounting(**d["mounting"]) if isinstance(d.get("mounting"), dict) else d.get("mounting", Mounting())
    env = Envelope(**d["envelope"]) if isinstance(d.get("envelope"), dict) else d.get("envelope", Envelope())

    opt_d = d.get("optimization", {})
    if isinstance(opt_d, dict):
        opt_d = dict(opt_d)  # Don't mutate the original
        weights_d = opt_d.pop("weights", {})
        if isinstance(weights_d, dict):
            opt_weights = OptimizationWeights(**weights_d)
        else:
            opt_weights = weights_d
        opt = OptimizationParams(**opt_d, weights=opt_weights)
    else:
        opt = opt_d

    out = OutputParams(**d["output"]) if isinstance(d.get("output"), dict) else d.get("output", OutputParams())

    return WeaponConfig(
        material=mat,
        weapon_style=d.get("weapon_style", "disk"),
        sheet_thickness_mm=d.get("sheet_thickness_mm", 10.0),
        weight_budget_kg=d.get("weight_budget_kg", 5.0),
        rpm=d.get("rpm", 8000),
        mounting=mount,
        envelope=env,
        optimization=opt,
        output=out,
    )


def _cfg_to_dict(cfg: WeaponConfig) -> dict:
    """Serialize a WeaponConfig to a plain dict for passing to workers."""
    return asdict(cfg)


# ---------------------------------------------------------------------------
# Behavior & MAP-Elites helpers
# ---------------------------------------------------------------------------

def behavior_vector(metrics: dict, cfg: WeaponConfig) -> np.ndarray:
    """Extract a 4-D normalized behavior vector from metrics.

    Dimensions:
        0: mass_utilization  — clipped to [0, 2]  then /2
        1: moi_normalized    — moi / max_possible_moi (approx)
        2: structural_integrity  — already [0,1]
        3: bite_normalized   — bite_mm clipped [0,40] then /40
    """
    mass_u = np.clip(metrics.get("mass_utilization", 0.0), 0.0, 2.0) / 2.0

    # Rough upper bound for MOI: solid disk at max radius
    max_r = cfg.envelope.max_radius_mm if cfg.weapon_style != "bar" else cfg.envelope.max_length_mm / 2
    max_moi = 0.5 * cfg.weight_budget_kg * (max_r ** 2)  # kg·mm²
    moi_n = np.clip(metrics.get("moi_kg_mm2", 0.0) / max(max_moi, 1e-6), 0.0, 1.0)

    si = np.clip(metrics.get("structural_integrity", 0.0), 0.0, 1.0)

    bite_n = np.clip(metrics.get("bite_mm", 0.0), 0.0, 40.0) / 40.0

    return np.array([mass_u, moi_n, si, bite_n], dtype=np.float64)


def discretize_cell(bv: np.ndarray, n_bins: int) -> tuple[int, ...]:
    """Map a behavior vector to a MAP-Elites grid cell."""
    indices = np.clip((bv * n_bins).astype(int), 0, n_bins - 1)
    return tuple(indices.tolist())


def combined_distance(
    params_a: np.ndarray, bv_a: np.ndarray,
    params_b: np.ndarray, bv_b: np.ndarray,
    bounds_range: np.ndarray,
) -> float:
    """Weighted distance in combined param + behavior space.

    0.5 * normalized_param_distance + 0.5 * behavior_distance
    """
    p_diff = (params_a - params_b) / np.maximum(bounds_range, 1e-12)
    p_dist = np.linalg.norm(p_diff)
    b_dist = np.linalg.norm(bv_a - bv_b)
    return 0.5 * p_dist + 0.5 * b_dist


# ---------------------------------------------------------------------------
# Candidate generation strategies
# ---------------------------------------------------------------------------

def generate_voronoi_candidates(
    archive: list[ArchiveMember],
    bounds: list[tuple[float, float]],
    n_candidates: int,
    rng: np.random.Generator,
    k_samples: int = 500,
) -> list[np.ndarray]:
    """Farthest-point sampling in normalized parameter space."""
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    span = hi - lo

    # Archive param matrix (normalized)
    if not archive:
        # No archive — pure random
        return [(rng.uniform(0, 1, len(bounds)) * span + lo) for _ in range(n_candidates)]

    arch_norm = np.array([(m.params - lo) / np.maximum(span, 1e-12) for m in archive])

    # Sample K random points in [0,1]^d
    samples = rng.uniform(0, 1, (k_samples, len(bounds)))

    # Min distance from each sample to any archive member
    dists = np.min(
        np.linalg.norm(samples[:, None, :] - arch_norm[None, :, :], axis=2),
        axis=1,
    )

    # Take the top-N farthest
    top_idx = np.argsort(dists)[-n_candidates:]
    return [(samples[i] * span + lo) for i in top_idx]


def generate_rrt_candidates(
    archive: list[ArchiveMember],
    bounds: list[tuple[float, float]],
    n_candidates: int,
    rng: np.random.Generator,
    iteration: int,
    max_iter: int,
) -> list[np.ndarray]:
    """RRT*-style extension from random archive parents."""
    if not archive:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        return [(rng.uniform(0, 1, len(bounds)) * span + lo) for _ in range(n_candidates)]

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    span = hi - lo

    # Step size decays from 15% to 5% over iterations
    progress = min(iteration / max(max_iter, 1), 1.0)
    step_frac = 0.15 - 0.10 * progress

    candidates = []
    for _ in range(n_candidates):
        parent = archive[rng.integers(len(archive))]
        direction = rng.standard_normal(len(bounds))
        direction /= np.linalg.norm(direction) + 1e-12
        step = direction * step_frac * span
        child = np.clip(parent.params + step, lo, hi)
        candidates.append(child)
    return candidates


def generate_mapelites_candidates(
    archive: list[ArchiveMember],
    bounds: list[tuple[float, float]],
    n_candidates: int,
    n_bins: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate candidates targeting empty or low-scoring MAP-Elites cells."""
    if len(archive) < 2:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        return [(rng.uniform(0, 1, len(bounds)) * span + lo) for _ in range(n_candidates)]

    # Build cell → members map
    cell_map: dict[tuple[int, ...], list[ArchiveMember]] = {}
    for m in archive:
        cell_map.setdefault(m.cell, []).append(m)

    # Find empty cells — sample from all possible
    all_cells = set()
    for idx in np.ndindex(*([n_bins] * 4)):
        all_cells.add(idx)
    occupied = set(cell_map.keys())
    empty_cells = list(all_cells - occupied)

    # Also find lowest-score occupied cells
    worst_cells = sorted(cell_map.keys(), key=lambda c: max(m.score for m in cell_map[c]))

    # Build target cell list (prefer empty, then worst)
    targets: list[tuple[int, ...]] = []
    if empty_cells:
        chosen_empty = [empty_cells[i] for i in rng.choice(len(empty_cells), min(n_candidates, len(empty_cells)), replace=False)]
        targets.extend(chosen_empty)
    remaining = n_candidates - len(targets)
    if remaining > 0 and worst_cells:
        targets.extend(worst_cells[:remaining])

    # Ensure we have enough targets
    while len(targets) < n_candidates:
        targets.append(targets[rng.integers(len(targets))])

    candidates = []
    bvs = np.array([m.behavior for m in archive])
    for cell in targets[:n_candidates]:
        # Cell center in behavior space
        center = (np.array(cell, dtype=np.float64) + 0.5) / n_bins

        # Find two nearest archive members by behavior distance
        dists = np.linalg.norm(bvs - center, axis=1)
        nearest_idx = np.argsort(dists)[:2]
        a, b = archive[nearest_idx[0]], archive[nearest_idx[1]]

        # Interpolate/extrapolate
        alpha = rng.uniform(-0.2, 1.2)
        child = a.params + alpha * (b.params - a.params)

        # Add small Gaussian noise
        lo = np.array([bd[0] for bd in bounds])
        hi = np.array([bd[1] for bd in bounds])
        span = hi - lo
        noise = rng.normal(0, 0.02, len(bounds)) * span
        child = np.clip(child + noise, lo, hi)
        candidates.append(child)

    return candidates


# ---------------------------------------------------------------------------
# Archive management
# ---------------------------------------------------------------------------

def compute_novelty(
    member: ArchiveMember,
    archive: list[ArchiveMember],
    bounds_range: np.ndarray,
) -> float:
    """Novelty = 0.5 * min_param_dist + 0.5 * min_behavior_dist (normalized)."""
    if not archive:
        return float("inf")

    min_p = float("inf")
    min_b = float("inf")
    for other in archive:
        p_diff = (member.params - other.params) / np.maximum(bounds_range, 1e-12)
        p_d = np.linalg.norm(p_diff)
        b_d = np.linalg.norm(member.behavior - other.behavior)
        if p_d < min_p:
            min_p = p_d
        if b_d < min_b:
            min_b = b_d
    return 0.5 * min_p + 0.5 * min_b


def try_insert(
    member: ArchiveMember,
    archive: list[ArchiveMember],
    cell_map: dict[tuple[int, ...], list[int]],
    n_target: int,
    bounds_range: np.ndarray,
) -> bool:
    """Try to insert a member into the archive.

    Insert if:
    - Cell is new (always)
    - Better score for existing cell (replace worst)
    - High novelty and archive under 2×target capacity
    """
    cell = member.cell

    if cell not in cell_map or not cell_map[cell]:
        # New cell — always insert
        idx = len(archive)
        archive.append(member)
        cell_map.setdefault(cell, []).append(idx)
        return True

    # Cell exists — check if we beat the worst occupant
    cell_indices = cell_map[cell]
    worst_idx = min(cell_indices, key=lambda i: archive[i].score)
    if member.score > archive[worst_idx].score:
        # Replace worst in this cell
        archive[worst_idx] = member
        return True

    # High novelty + under capacity
    if len(archive) < 2 * n_target:
        novelty = compute_novelty(member, archive, bounds_range)
        if novelty > 0.1:  # reasonable novelty threshold
            idx = len(archive)
            archive.append(member)
            cell_map.setdefault(cell, []).append(idx)
            return True

    return False


def prune_archive(
    archive: list[ArchiveMember],
    n_target: int,
    bounds_range: np.ndarray,
    n_bins: int,
) -> list[ArchiveMember]:
    """Prune archive to ≤ 2×target. Keep best-per-cell, greedily remove least-novel extras."""
    if len(archive) <= 2 * n_target:
        return archive

    # Identify best-per-cell (always keep)
    cell_map: dict[tuple[int, ...], list[int]] = {}
    for i, m in enumerate(archive):
        cell_map.setdefault(m.cell, []).append(i)

    keep_set: set[int] = set()
    for cell, indices in cell_map.items():
        best_idx = max(indices, key=lambda i: archive[i].score)
        keep_set.add(best_idx)

    extras = [i for i in range(len(archive)) if i not in keep_set]

    # Compute novelty for extras
    novelties = []
    for i in extras:
        nov = compute_novelty(archive[i], archive, bounds_range)
        novelties.append((i, nov))

    # Sort by novelty ascending (least novel first) and remove
    novelties.sort(key=lambda x: x[1])
    n_to_remove = len(archive) - 2 * n_target
    remove_set = set(idx for idx, _ in novelties[:n_to_remove])

    return [m for i, m in enumerate(archive) if i not in remove_set]


def rebuild_cell_map(archive: list[ArchiveMember]) -> dict[tuple[int, ...], list[int]]:
    """Rebuild cell → archive index mapping."""
    cell_map: dict[tuple[int, ...], list[int]] = {}
    for i, m in enumerate(archive):
        cell_map.setdefault(m.cell, []).append(i)
    return cell_map


# ---------------------------------------------------------------------------
# Final selection
# ---------------------------------------------------------------------------

def greedy_diverse_subset(
    archive: list[ArchiveMember],
    n_target: int,
    bounds_range: np.ndarray,
) -> list[ArchiveMember]:
    """Greedy farthest-point selection in combined param+behavior space."""
    if len(archive) <= n_target:
        return list(archive)

    # Start with the highest-scoring member
    selected = [max(range(len(archive)), key=lambda i: archive[i].score)]

    while len(selected) < n_target:
        best_idx = -1
        best_min_dist = -1.0

        for i in range(len(archive)):
            if i in selected:
                continue
            min_dist = min(
                combined_distance(
                    archive[i].params, archive[i].behavior,
                    archive[j].params, archive[j].behavior,
                    bounds_range,
                )
                for j in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        if best_idx < 0:
            break
        selected.append(best_idx)

    # Convert to set for O(1) lookup, maintain order
    return [archive[i] for i in selected]


# ---------------------------------------------------------------------------
# Main exploration loop
# ---------------------------------------------------------------------------

def explore_diverse(
    cfg: WeaponConfig,
    n_target: int = 40,
    max_iter: int = 500,
    batch_size: int = 64,
    min_score: float = 0.3,
    n_bins: int = 4,
    seed: int = 42,
    n_workers: int = 0,
    quiet: bool = False,
) -> list[ArchiveMember]:
    """Run the diversity exploration loop."""
    rng = np.random.default_rng(seed)

    bounds = _get_profile_bounds(cfg)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    bounds_range = hi - lo
    ndim = len(bounds)

    cfg_dict = _cfg_to_dict(cfg)

    # --- Seed from archetypes ---
    archetypes = get_archetypes(cfg.weapon_style)
    log.info("Seeding with %d archetypes for style '%s'", len(archetypes), cfg.weapon_style)

    archive: list[ArchiveMember] = []
    cell_map: dict[tuple[int, ...], list[int]] = {}
    design_counter = 0

    seed_params = []
    for arch in archetypes:
        p = archetype_to_params(arch, cfg, profile_only=True)
        # Pad or truncate to match bounds dimension
        if len(p) < ndim:
            p = np.concatenate([p, np.zeros(ndim - len(p))])
        elif len(p) > ndim:
            p = p[:ndim]
        p = np.clip(p, lo, hi)
        seed_params.append(p)

    # Evaluate seeds
    seed_args = [(p, cfg_dict) for p in seed_params]
    for i, (p, result) in enumerate(zip(seed_params, map(evaluate_candidate, seed_args))):
        if result is None:
            continue
        effective_score = result["score"] * result["penalty"]
        if effective_score < min_score:
            continue
        bv = behavior_vector(result["metrics"], cfg)
        cell = discretize_cell(bv, n_bins)
        design_counter += 1
        member = ArchiveMember(
            params=p,
            score=effective_score,
            penalty=result["penalty"],
            metrics=result["metrics"],
            behavior=bv,
            cell=cell,
            generation_method="archetype_seed",
            iteration_found=0,
            design_id=design_counter,
        )
        try_insert(member, archive, cell_map, n_target, bounds_range)

    log.info("Archive after seeding: %d members", len(archive))

    # --- Main loop ---
    if n_workers == 0:
        n_workers = max(1, mp.cpu_count() - 1)

    pool = mp.Pool(processes=n_workers)
    try:
        for iteration in range(1, max_iter + 1):
            # Round-robin: split batch into 3 strategies
            n_per = batch_size // 3
            n_extra = batch_size - 3 * n_per

            voronoi_cands = generate_voronoi_candidates(
                archive, bounds, n_per + n_extra, rng,
            )
            rrt_cands = generate_rrt_candidates(
                archive, bounds, n_per, rng, iteration, max_iter,
            )
            me_cands = generate_mapelites_candidates(
                archive, bounds, n_per, n_bins, rng,
            )

            all_cands = voronoi_cands + rrt_cands + me_cands
            labels = (
                ["voronoi"] * len(voronoi_cands)
                + ["rrt"] * len(rrt_cands)
                + ["mapelites"] * len(me_cands)
            )

            # Evaluate in parallel
            eval_args = [(c, cfg_dict) for c in all_cands]
            results = pool.map(evaluate_candidate, eval_args)

            n_inserted = 0
            for cand, result, label in zip(all_cands, results, labels):
                if result is None:
                    continue

                effective_score = result["score"] * result["penalty"]

                # Phase-1 relaxed feasibility
                mu = result["metrics"].get("mass_utilization", 0.0)
                if not (0.3 < mu < 2.0):
                    continue
                if effective_score < min_score:
                    continue

                bv = behavior_vector(result["metrics"], cfg)
                cell = discretize_cell(bv, n_bins)
                design_counter += 1

                member = ArchiveMember(
                    params=cand,
                    score=effective_score,
                    penalty=result["penalty"],
                    metrics=result["metrics"],
                    behavior=bv,
                    cell=cell,
                    generation_method=label,
                    iteration_found=iteration,
                    design_id=design_counter,
                )

                if try_insert(member, archive, cell_map, n_target, bounds_range):
                    n_inserted += 1

                    # RRT* rewiring: if this candidate beats a nearby member
                    # in a cell with >1 representative, replace it
                    if label == "rrt":
                        for ci, indices in cell_map.items():
                            if len(indices) <= 1:
                                continue
                            for idx in indices:
                                if idx >= len(archive):
                                    continue
                                other = archive[idx]
                                dist = np.linalg.norm(
                                    (cand - other.params) / np.maximum(bounds_range, 1e-12)
                                )
                                if dist < 0.2 and effective_score > other.score:
                                    archive[idx] = member
                                    break

            # Periodic pruning
            if len(archive) > 2 * n_target:
                archive = prune_archive(archive, n_target, bounds_range, n_bins)
                cell_map = rebuild_cell_map(archive)

            if not quiet and iteration % 10 == 0:
                unique_cells = len(set(m.cell for m in archive))
                best_score = max((m.score for m in archive), default=0)
                log.info(
                    "Iter %4d | archive %3d | cells %3d | inserted %2d | best %.3f",
                    iteration, len(archive), unique_cells, n_inserted, best_score,
                )
    finally:
        pool.close()
        pool.join()

    # --- Final selection ---
    log.info("Final selection: picking %d from %d archive members", n_target, len(archive))
    selected = greedy_diverse_subset(archive, n_target, bounds_range)

    # Re-number
    for i, m in enumerate(selected, 1):
        m.design_id = i

    return selected


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_results(
    designs: list[ArchiveMember],
    cfg: WeaponConfig,
    output_dir: Path,
) -> None:
    """Export designs to DXF + JSON stats + summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_entries = []

    for m in designs:
        label = f"design_{m.design_id:03d}"

        # Rebuild polygon for DXF export
        try:
            C = cfg.optimization.num_cutout_pairs
            S = _cutout_stride(cfg)
            x_full = np.concatenate([m.params, np.zeros(C * S)])
            outer, _, cutout_polys = build_weapon_polygon(x_full, cfg)
            outer = validate_geometry(outer)
            weapon = assemble_weapon(outer, cfg.mounting, cutout_polys)
            weapon = validate_geometry(weapon)

            dxf_path = output_dir / f"{label}.dxf"
            export_dxf(weapon, dxf_path, cfg)
        except Exception as exc:
            log.warning("Failed to export %s: %s", label, exc)
            continue

        # Per-design stats JSON
        stats = {
            "design_id": m.design_id,
            "generation_method": m.generation_method,
            "iteration_found": m.iteration_found,
            "score": round(m.score, 6),
            "penalty": round(m.penalty, 6),
            "behavior_vector": [round(float(v), 6) for v in m.behavior],
            "map_elites_cell": list(m.cell),
            "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in m.metrics.items()},
            "params": m.params.tolist(),
        }
        stats_path = output_dir / f"{label}_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        summary_entries.append(stats)

    # Summary JSON
    unique_cells = len(set(tuple(e["map_elites_cell"]) for e in summary_entries))
    scores = [e["score"] for e in summary_entries]
    summary = {
        "n_designs": len(summary_entries),
        "unique_cells": unique_cells,
        "score_min": round(min(scores), 6) if scores else 0,
        "score_max": round(max(scores), 6) if scores else 0,
        "score_mean": round(float(np.mean(scores)), 6) if scores else 0,
        "designs": summary_entries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Summary CSV
    if summary_entries:
        csv_path = output_dir / "summary.csv"
        fieldnames = [
            "design_id", "generation_method", "iteration_found",
            "score", "penalty",
            "bv_mass_util", "bv_moi", "bv_structural", "bv_bite",
            "cell",
        ]
        # Add metric keys from first entry
        metric_keys = list(summary_entries[0]["metrics"].keys())
        fieldnames.extend(metric_keys)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for e in summary_entries:
                row = {
                    "design_id": e["design_id"],
                    "generation_method": e["generation_method"],
                    "iteration_found": e["iteration_found"],
                    "score": e["score"],
                    "penalty": e["penalty"],
                    "bv_mass_util": e["behavior_vector"][0],
                    "bv_moi": e["behavior_vector"][1],
                    "bv_structural": e["behavior_vector"][2],
                    "bv_bite": e["behavior_vector"][3],
                    "cell": str(e["map_elites_cell"]),
                }
                row.update(e["metrics"])
                writer.writerow(row)

    log.info("Exported %d designs to %s", len(summary_entries), output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate maximally diverse weapon designs",
    )
    parser.add_argument("config", help="Path to weapon config JSON")
    parser.add_argument("--n-designs", type=int, default=40, help="Target diverse designs (default: 40)")
    parser.add_argument("--max-iter", type=int, default=500, help="Exploration iterations (default: 500)")
    parser.add_argument("--batch-size", type=int, default=64, help="Candidates per iteration (default: 64)")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum quality threshold (default: 0.3)")
    parser.add_argument("--n-bins", type=int, default=4, help="MAP-Elites bins per dimension (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--workers", type=int, default=0, help="Worker processes (0=auto)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress")
    args = parser.parse_args()

    # Logging
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"explore_{timestamp}")

    log.info("Config: %s  |  Style: %s", args.config, cfg.weapon_style)
    log.info("Target: %d designs  |  Max iter: %d  |  Batch: %d", args.n_designs, args.max_iter, args.batch_size)
    log.info("Output: %s", output_dir)

    t0 = time.monotonic()

    designs = explore_diverse(
        cfg,
        n_target=args.n_designs,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        min_score=args.min_score,
        n_bins=args.n_bins,
        seed=args.seed,
        n_workers=args.workers,
        quiet=args.quiet,
    )

    elapsed = time.monotonic() - t0
    log.info("Exploration complete in %.1fs — %d designs selected", elapsed, len(designs))

    export_results(designs, cfg, output_dir)

    # Final report
    if designs:
        scores = [m.score for m in designs]
        cells = set(m.cell for m in designs)
        methods = {}
        for m in designs:
            methods[m.generation_method] = methods.get(m.generation_method, 0) + 1

        print(f"\n{'=' * 60}")
        print(f"Diversity Explorer — Results")
        print(f"{'=' * 60}")
        print(f"Designs exported:    {len(designs)}")
        print(f"Unique MAP-E cells:  {len(cells)}")
        print(f"Score range:         {min(scores):.3f} – {max(scores):.3f}")
        print(f"Score mean:          {np.mean(scores):.3f}")
        print(f"Generation methods:  {methods}")
        print(f"Output directory:    {output_dir}")
        print(f"Time elapsed:        {elapsed:.1f}s")
        print(f"{'=' * 60}\n")
    else:
        print("No valid designs found. Try lowering --min-score or increasing --max-iter.")


if __name__ == "__main__":
    main()
