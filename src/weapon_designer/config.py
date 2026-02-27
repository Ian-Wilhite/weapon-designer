"""JSON config loading and dataclass definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Material:
    name: str = "AR500"
    density_kg_m3: float = 7850.0
    yield_strength_mpa: float = 1400.0
    hardness_hrc: float = 50.0


@dataclass
class Mounting:
    bore_diameter_mm: float = 25.4
    bolt_circle_diameter_mm: float = 50.0
    num_bolts: int = 4
    bolt_hole_diameter_mm: float = 6.5


@dataclass
class Envelope:
    max_radius_mm: float = 150.0
    max_length_mm: float = 300.0
    max_width_mm: float = 80.0


@dataclass
class OptimizationWeights:
    moment_of_inertia: float = 0.30
    bite: float = 0.15
    structural_integrity: float = 0.20
    mass_utilization: float = 0.10
    balance: float = 0.10
    impact_zone: float = 0.15


@dataclass
class OptimizationParams:
    weights: OptimizationWeights = field(default_factory=OptimizationWeights)
    num_fourier_terms: int = 5
    num_cutout_pairs: int = 3
    num_cutout_fourier_terms: int = 2
    snapshot_interval: int = 20
    min_feature_size_mm: float = 3.0
    min_wall_thickness_mm: float = 5.0
    max_iterations: int = 200
    population_size: int = 60
    # Enhanced evaluation mode (parallel to baseline, for research comparison)
    evaluation_mode: str = "baseline"   # "baseline" | "enhanced"
    cutout_type: str = "fourier"        # "fourier" | "superellipse"
    fea_interval: int = 5               # save FEA frame every N optimizer steps (0=disabled)
    fea_coarse_spacing_mm: float = 10.0 # mesh spacing during optimization (fast)
    fea_fine_spacing_mm: float = 4.0    # mesh spacing for final/frame renders (quality)
    n_bspline_points: int = 12          # number of B-spline radial control points (enhanced mode)
    profile_type: str = "bspline"       # "fourier" | "bspline" | "bezier" | "catmull_rom"
    phase1_iters: int = 0               # explicit Phase-1 iteration cap (0 = 50% of max_iterations)
    phase2_iters: int = 0               # explicit Phase-2 iteration cap (0 = 25% of max_iterations)
    # Topology optimisation Phase-2 parameters (cutout_type = "topology")
    topo_n_iter: int = 60               # SIMP iteration count
    topo_mesh_spacing_mm: float = 6.0   # element edge length for topo mesh (mm)
    topo_p_simp: float = 3.0            # SIMP penalisation exponent
    topo_r_min_factor: float = 2.5      # filter radius = factor × mesh_spacing
    topo_w_compliance: float = 0.5      # weight on compliance (1−this = MOI weight)
    topo_frame_interval: int = 2        # save density frame every N topo iters (0=every iter)
    topo_fix_rim: bool = True           # keep outer rim strip fixed-solid
    # Early-stopping convergence criteria (enhanced optimizer)
    convergence_patience: int = 15      # stop if best score improves < min_delta over this many steps
    convergence_min_delta: float = 0.002  # minimum score improvement to reset patience counter


@dataclass
class OutputParams:
    dxf_path: str = "output.dxf"
    stats_path: str = "output_stats.json"
    preview: bool = False
    base_dir: str = "runs"
    run_subdir: str | None = None
    resume_stats: str | None = None
    run_dir: str | None = None


@dataclass
class WeaponConfig:
    material: Material = field(default_factory=Material)
    weapon_style: Literal["bar", "disk", "eggbeater"] = "disk"
    sheet_thickness_mm: float = 10.0
    weight_budget_kg: float = 5.0
    rpm: int = 8000
    mounting: Mounting = field(default_factory=Mounting)
    envelope: Envelope = field(default_factory=Envelope)
    optimization: OptimizationParams = field(default_factory=OptimizationParams)
    output: OutputParams = field(default_factory=OutputParams)


def _dict_to_dataclass(cls, d: dict):
    """Recursively convert a dict to a dataclass, ignoring unknown keys."""
    if not isinstance(d, dict):
        return d
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {}
    for k, v in d.items():
        if k not in field_names:
            continue
        ft = cls.__dataclass_fields__[k].type
        # Resolve string annotations
        if isinstance(ft, str):
            ft = eval(ft, {**globals(), **{cls.__name__: cls}})
        if hasattr(ft, "__dataclass_fields__") and isinstance(v, dict):
            filtered[k] = _dict_to_dataclass(ft, v)
        else:
            filtered[k] = v
    return cls(**filtered)


def load_config(path: str | Path) -> WeaponConfig:
    """Load a WeaponConfig from a JSON file."""
    with open(path) as f:
        raw = json.load(f)
    return _dict_to_dataclass(WeaponConfig, raw)
