# weapon-designer

Parametric combat robot spinning weapon profile optimizer.

Generates optimized spinning weapon profiles for combat robots using parameterized geometry and multi-objective differential evolution. Outputs CAD-ready DXF files for waterjet or laser cutting.

The project doubles as a research comparison between two optimizer generations (baseline Fourier vs. enhanced B-spline/FEA).

## Requirements

- Python 3.10+

## Installation

```bash
git clone <repo-url>
cd weapon-designer

# Install in editable mode
pip install -e .

# Include visualization/GIF support (matplotlib, Pillow)
pip install -e ".[viz]"

# Include development dependencies (pytest)
pip install -e ".[dev]"
```

## Quick Start

Run the optimizer with one of the included example configs:

```bash
weapon-designer configs/example_disk.json
```

Produces:
- `output_disk.dxf` — weapon profile ready for CAD/cutting
- `output_disk_stats.json` — metrics and optimization results

## CLI Usage

```
weapon-designer <config.json> [--preview] [--dxf PATH] [--stats PATH] [--quiet]
```

| Flag | Description |
|------|-------------|
| `config` | Path to a JSON configuration file (required) |
| `--preview` | Show a matplotlib preview window of the optimized weapon |
| `--dxf PATH` | Override the DXF output file path |
| `--stats PATH` | Override the stats JSON output file path |
| `--quiet` | Suppress progress output |

## Evaluation Harness

`evaluate.py` runs a fixed test-case suite and logs detailed per-objective results:

```bash
python3 evaluate.py                          # baseline mode (all cases)
python3 evaluate.py --mode enhanced          # enhanced optimizer
python3 evaluate.py --mode compare           # side-by-side baseline vs. enhanced
python3 evaluate.py --cases heavyweight_disk,compact_bar
python3 evaluate.py --quick                  # reduced iterations for smoke tests
python3 evaluate.py --output-dir eval_results/my_run
```

Results land in `eval_results/` as timestamped logs and per-case DXF/stats/snapshots.

## Exploration Scripts

**`explore_diverse.py`** — Generate a maximally diverse family of weapons from a single config using MAP-Elites + RRT*-style expansion:

```bash
python3 explore_diverse.py configs/example_disk.json [--archive-size 200] [--iterations 500]
```

**`sweep_explore.py`** — Full parameter sweep across all weapon types, weight classes, and objective emphasis profiles (designed for overnight runs on many-core machines):

```bash
python3 sweep_explore.py [--output-dir sweep_results] [--dry-run] [--resume]
```

Sweep results land in `sweep_results/` with per-job subdirectories and a merged summary JSON.

## Weapon Styles

| Style | Description |
|-------|-------------|
| `disk` | Spinning disk with modulated perimeter |
| `bar` | Rectangular bar with sculpted tips |
| `eggbeater` | Multi-bladed impeller (2–4 blades) with central hub |

## Architecture

```
src/weapon_designer/
├── config.py              # Dataclasses: WeaponConfig, Material, Mounting, Envelope, OptimizationParams
├── geometry.py            # assemble_weapon() — subtracts mounting + cutout holes
├── constraints.py         # validate_geometry(), constraint_penalty(), check_envelope()
├── physics.py             # Mass, MOI, energy, bite (analytic formulas) — BASELINE, frozen
├── structural.py          # Geometric structural proxy (wall thickness, section width)
├── archetypes.py          # Population seeding from known-good archetype shapes
├── exporter.py            # DXF + JSON snapshot export
├── visualization.py       # matplotlib weapon preview
├── fea.py                 # 2D CST plane-stress FEA; centrifugal + contact point loading
│
│── BASELINE optimizer stack (frozen for research comparison)
├── parametric.py          # Fourier outer profiles + Fourier cutouts
├── objectives.py          # Weighted scoring (formula bite, geometric structural proxy)
├── optimizer.py           # Two-phase differential evolution
│
│── ENHANCED optimizer stack (active development)
├── profile_builder.py     # Dispatcher: build_profile(type, radii, cfg) → Polygon
├── bspline_profile.py     # Periodic cubic B-spline outer profile (C², local support)
├── profile_splines.py     # Composite cubic Bézier (C¹) + centripetal Catmull-Rom (C¹)
├── parametric_cad.py      # Superellipse/polar cutouts (r,φ,a,b,n) — CAD-interpretable
├── spiral_contact.py      # Archimedean spiral contact analysis + FEA point forces
├── objectives_enhanced.py # Kinematic spiral bite + FEA-in-loop metrics
├── optimizer_enhanced.py  # Enhanced two-phase DE: any profile type + sidecars + GIF
├── topo_optimizer.py      # SIMP topology optimizer (Phase 2 alternative, cutout_type="topology")
└── fea_viz.py             # FEA stress-map frame renderer + GIF assembler
```

Top-level tools:

```
evaluate.py           # Run baseline / enhanced / compare mode on fixed test cases
sweep_profiles.py     # Full comparison sweep: all 4 profile families × all cases → charts
fea_replay.py         # Reassemble convergence GIFs from saved .npz sidecar files
fea_speed_sweep.py    # 3-D stacked FEA heatmap: centrifugal + spiral-impact vs. RPM sweep
```

### Baseline vs. Enhanced

The **baseline** stack (`parametric.py`, `objectives.py`, `optimizer.py`, `physics.py`) is frozen — it defines the research comparison baseline and must not be modified.

The **enhanced** stack adds:

1. **Profile dispatcher** (`profile_builder.py`): `build_profile(profile_type, radii, cfg)` routes to any of four profile families. Swap `profile_type` in config with no other code changes.

2. **Four profile families** — all share the same DE bounds (`n_bspline_points × (r_min, r_max)`):
   | Key | Module | Continuity | Notes |
   |-----|--------|-----------|-------|
   | `fourier` | `parametric.py` | C∞ | Baseline only; global support (Gibbs-prone) |
   | `bspline` | `bspline_profile.py` | C² | Default enhanced; scipy splprep periodic |
   | `bezier` | `profile_splines.py` | C¹ | Composite cubic; CR-style tangents |
   | `catmull_rom` | `profile_splines.py` | C¹ | Interpolates control points; centripetal α=0.5 |

3. **Polar cutouts** (`parametric_cad.py`): `(r, φ, a, b, n)` superellipse pockets positioned in polar coordinates. Asymmetric layouts supported. Analytic mass normalization scales `a,b` to hit the weight target exactly.

4. **Topology optimisation** (`topo_optimizer.py`): SIMP (Solid Isotropic Material with Penalization) continuum topology optimiser as a Phase-2 alternative to parametric cutouts. Discovers the optimal material layout within the Phase-1 outer profile under centrifugal loading. Set `cutout_type = "topology"` to activate. Produces three animated GIFs (density evolution, binary threshold, FEA stress) plus a convergence plot.

5. **Kinematic spiral bite** (`objectives_enhanced.py`): Archimedean spiral simulation counts tooth contacts per revolution for geometry-derived bite depth. One large protrusion → maximum bite; many small teeth → correctly penalized.

6. **Spiral contact analysis** (`spiral_contact.py`): Analyses a full family of N spirals at uniform approach angles. Returns first-contact location, bite depth, and tangential force direction for each spiral. Output can be passed directly to the FEA as point loads superimposed on centrifugal loading.

7. **FEA in optimization loop** (`fea.py`): Coarse-mesh 2D plane-stress FEA at every objective evaluation. Optionally augmented with contact point forces from `spiral_contact.contact_forces()`. Safety factor directly enters the weighted score.

8. **GIF export with sidecars** (`fea_viz.py`, `optimizer_enhanced.py`): FEA stress-map frames saved every `fea_interval` steps. Each frame `frame_NNNN.png` has matching sidecar files `frame_NNNN.npz` (mesh arrays) and `frame_NNNN_meta.json` (step metadata). `fea_replay.py` can reassemble GIFs from sidecars post-hoc.

9. **Topology optimisation** (`topo_optimizer.py`): SIMP-based Phase 2 alternative. Replaces discrete hole placement with continuous density field optimisation (minimize compliance, constrain volume fraction). Activate with `cutout_type = "topology"`. Fixes the hub annulus and optional outer rim as always-solid; post-processes to a clean Shapely polygon via adaptive threshold + buffer smoothing.

## Configuration

Configs are JSON files. See `configs/` for examples. Key fields:

```json
{
  "material": {
    "name": "AR500",
    "density_kg_m3": 7850,
    "yield_strength_mpa": 1400,
    "hardness_hrc": 50
  },
  "weapon_style": "disk",
  "sheet_thickness_mm": 10,
  "weight_budget_kg": 5.0,
  "rpm": 8000,
  "mounting": {
    "bore_diameter_mm": 25.4,
    "bolt_circle_diameter_mm": 50,
    "num_bolts": 4,
    "bolt_hole_diameter_mm": 6.5
  },
  "envelope": {
    "max_radius_mm": 150
  },
  "optimization": {
    "weights": {
      "moment_of_inertia": 0.35,
      "bite": 0.20,
      "structural_integrity": 0.25,
      "mass_utilization": 0.10,
      "balance": 0.10
    },
    "num_fourier_terms": 4,
    "num_cutout_pairs": 2,
    "min_feature_size_mm": 3.0,
    "min_wall_thickness_mm": 5.0,
    "max_iterations": 100,
    "population_size": 40,
    "evaluation_mode": "enhanced",
    "cutout_type": "superellipse",
    "profile_type": "bspline",
    "fea_interval": 5,
    "fea_coarse_spacing_mm": 10.0,
    "fea_fine_spacing_mm": 4.0,
    "n_bspline_points": 12,
    "phase1_iters": 0,
    "phase2_iters": 0,
    "topo_n_iter": 60,
    "topo_mesh_spacing_mm": 6.0,
    "topo_p_simp": 3.0,
    "topo_r_min_factor": 2.5,
    "topo_w_compliance": 0.5,
    "topo_frame_interval": 2,
    "topo_fix_rim": true
  },
  "output": {
    "dxf_path": "output_disk.dxf",
    "stats_path": "output_disk_stats.json",
    "preview": false
  }
}
```

### Configuration Reference

**material** — Stock material properties.

| Field | Description |
|-------|-------------|
| `density_kg_m3` | Material density |
| `yield_strength_mpa` | Yield strength |
| `hardness_hrc` | Rockwell C hardness |

**weapon_style** — One of `"disk"`, `"bar"`, or `"eggbeater"`.

**mounting** — Hub features subtracted from the profile.

| Field | Description |
|-------|-------------|
| `bore_diameter_mm` | Center spindle bore diameter |
| `bolt_circle_diameter_mm` | Bolt pattern circle diameter |
| `num_bolts` | Number of mounting bolts |
| `bolt_hole_diameter_mm` | Bolt hole diameter |

**envelope** — Maximum weapon dimensions.

| Field | Description |
|-------|-------------|
| `max_radius_mm` | Max radius (disk/eggbeater) |
| `max_length_mm` | Max length (bar only) |
| `max_width_mm` | Max width (bar only) |

**optimization.weights** — Relative importance of each objective (should sum to 1.0).

| Weight | Description |
|--------|-------------|
| `moment_of_inertia` | Energy storage capacity |
| `bite` | Impact spacing (targets ~20mm ideal) |
| `structural_integrity` | Geometric + FEA structural score |
| `mass_utilization` | How fully the weight budget is used |
| `balance` | Penalizes center-of-mass offset |
| `impact_zone` | Concentration of mass at the impact perimeter |

**optimization** — Optimizer tuning.

| Field | Default | Description |
|-------|---------|-------------|
| `num_fourier_terms` | 4 | Fourier series complexity (baseline only) |
| `num_cutout_pairs` | 2 | Number of lightening pocket pairs |
| `min_feature_size_mm` | 3.0 | Smallest cuttable feature |
| `min_wall_thickness_mm` | 5.0 | Minimum wall between features |
| `max_iterations` | 100 | DE iteration limit |
| `population_size` | 40 | Population per generation |
| `evaluation_mode` | `"baseline"` | `"baseline"` or `"enhanced"` |
| `cutout_type` | `"fourier"` | `"fourier"`, `"superellipse"`, or `"topology"` |
| `profile_type` | `"bspline"` | `"fourier"`, `"bspline"`, `"bezier"`, or `"catmull_rom"` |
| `fea_interval` | `0` | Save FEA GIF frame every N steps (0 = disabled) |
| `fea_coarse_spacing_mm` | `10.0` | Mesh spacing during optimization |
| `fea_fine_spacing_mm` | `4.0` | Mesh spacing for final/frame renders |
| `n_bspline_points` | `12` | Control points shared by all spline-family profiles |
| `phase1_iters` | `0` | Phase-1 iteration cap (0 = 50% of max_iterations) |
| `phase2_iters` | `0` | Phase-2 iteration cap (0 = 25% of max_iterations) |

**optimization (topology-only keys)** — Only active when `cutout_type = "topology"`.

| Field | Default | Description |
|-------|---------|-------------|
| `topo_n_iter` | `60` | SIMP iteration count |
| `topo_mesh_spacing_mm` | `6.0` | FEA mesh element edge length (mm) |
| `topo_p_simp` | `3.0` | SIMP penalisation exponent |
| `topo_r_min_factor` | `2.5` | Density filter radius = factor × mesh_spacing |
| `topo_w_compliance` | `0.5` | Compliance weight (vs. MOI weight = 1 − topo_w_compliance) |
| `topo_frame_interval` | `2` | Save GIF frame every N SIMP iterations (0 = disabled) |
| `topo_fix_rim` | `true` | Lock outer rim strip (r ≥ max_r × 0.88) as always-solid |

## Output

### DXF File

- `WEAPON` layer (white) — exterior profile as a closed polyline
- `WEAPON_HOLES` layer (red) — interior holes (bore, bolts, cutouts)

Units are millimeters. Import directly into CAD software or send to a waterjet/laser cutter.

### Stats JSON

Contains optimization results and computed metrics:

- Mass, moment of inertia, stored energy
- Bite distance, center-of-mass offset
- Structural integrity score, FEA safety factor
- Overall optimization score and constraint penalty
- Configuration summary

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Docs

- [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) — Two-phase DE strategy, profile families, spiral contact analysis, FEA, topology optimisation, archetype seeding
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Swap-point table, permutation matrix, data-flow diagram, extension guides
