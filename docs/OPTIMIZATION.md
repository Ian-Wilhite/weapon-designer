# Optimization Strategy

This document describes the optimization approach used by weapon-designer, the
reasoning behind key design decisions, and the differences between the baseline
and enhanced evaluation stacks.

---

## Problem Characteristics

| Property | Baseline | Enhanced (polar) | Enhanced (topology) |
|----------|----------|---------|---------------------|
| Profile params | 2k+1 Fourier coefficients (k=4 → 9 params) | N radii, N=12 default | N radii, N=12 default |
| Phase-2 params | 6 per pair × num_pairs (ellipse position + size) | 5 per cutout × num_cutouts (`r, φ, a, b, n`) | n_elem density values (continuum) |
| Total dims | ~19–21 | ~22 (12 profile + 10 cutouts) | 12 (Phase 1) + ~500–2000 (Phase 2 mesh) |
| Landscape | Non-convex, multimodal | Same, smoother (local support) | Convex (OC guaranteed descent) |
| Constraints | Connectivity, mass, envelope, min feature | Same + analytic mass normalization | Volume fraction via OC bisection |
| Evaluation cost | ~2–5 ms (geometry + proxy) | ~50–200 ms (geometry + coarse FEA) | ~1–5 s/iter × n_iter (sparse FEA solve) |
| Phase-2 method | Differential Evolution | Differential Evolution | Optimality Criteria (OC) |

---

## Two-Phase Optimization

Both stacks use the same decomposition: Phase 1 optimizes the outer profile,
Phase 2 fixes it and optimizes cutouts.

### Phase 1 — Outer Profile

Optimizes the profile parameters that define the weapon's outer boundary shape.
The objective focuses on:

- Moment of inertia (energy storage)
- Bite depth (geometry-derived in enhanced mode)
- Balance (CoM offset from spin axis)
- Envelope compliance

Default iteration split: 50 % of `max_iterations`.  Override with
`phase1_iters` and `phase2_iters` in the config.

**Seeding (baseline only):** the initial population is seeded from the
**archetype library** — normalized Fourier coefficient vectors for known-good
shapes (solid disk, 2-tooth, 3-tooth, etc.) scaled to the target envelope.
Enhanced mode uses random initialization because spline control points do not
map cleanly onto Fourier archetypes.

### Phase 2 — Weight Reduction (two implementations)

Fixes the outer profile from Phase 1 and removes material to hit the mass
budget.  Two Phase-2 strategies are available, selected by `cutout_type`.

#### `cutout_type = "superellipse"` — Polar Parametric Cutouts (default)

Optimizes `(r, φ, a, b, n)` polar superellipse pocket positions and sizes via
Differential Evolution.  Default iteration split: 25 % of `max_iterations`.

**Analytic mass normalization:** instead of letting the DE explore cutout sizes
freely, Phase 2 scales all `a` and `b` semi-axes uniformly by
`s = sqrt(A_target / A_effective)` after convergence.  This removes the mass
constraint from the search space and lets the DE focus entirely on hole shape,
position, and orientation.

#### `cutout_type = "topology"` — SIMP Topology Optimisation

Replaces the parametric cutout search with a continuum topology optimiser
(`topo_optimizer.py`) based on the **SIMP** (Solid Isotropic Material with
Penalization) method.

**How it works:**

1. The solid Phase-1 polygon (with bore/bolt holes, no weight-reduction pockets)
   is meshed into CST triangular elements at `topo_mesh_spacing_mm` spacing.
2. Each element is assigned a density variable `ρ_e ∈ [ρ_min, 1]`.
3. Each SIMP iteration:
   - Assemble stiffness `K = Σ [E_min + ρ_e^p (E₀ − E_min)] K_e^unit`
   - Apply centrifugal body load scaled by `ρ_e` (design-dependent)
   - Solve `K u = F` (sparse direct solver)
   - Compute combined sensitivity:
     ```
     ∂J/∂ρ_e = w_C · ∂C/∂ρ_e  −  w_MOI · ∂MOI/∂ρ_e
     ```
     where `∂C/∂ρ_e = −p ρ^(p−1) u_e^T K_e^unit u_e` (≤ 0) and
     `∂MOI/∂ρ_e = ρ_material · t · A_e · r_e²` (≥ 0)
   - Apply heuristic sensitivity filter (radius `r_min = topo_r_min_factor × mesh_spacing`)
   - Optimality Criteria update with bisection on the Lagrange multiplier
     to enforce the volume-fraction constraint `Σ ρ_e A_e / A_total = V_f`
4. After convergence, find the adaptive threshold `τ` (via binary search) such
   that the void elements have total area exactly `(1 − V_f) × A_total`.  This
   ensures the extracted binary design has the correct mass.
5. Void element triangles are union'd, smoothed, and subtracted from the outer
   profile to produce the final weapon polygon.

**Fixed regions:** Elements within `bore_radius × 2.5` are pinned solid (hub
attachment zone).  If `topo_fix_rim = true`, elements with `r ≥ max_radius × 0.88`
are also pinned solid (continuous outer rim for structural integrity).

**Objective balance:** `topo_w_compliance` (default 0.5) weights structural
stiffness; the remainder weights MOI maximisation.  Higher compliance weight
→ more load-path-following topology (spokes toward rim).  Higher MOI weight
→ more rim-concentrated material (wider ring with less internal structure).

**Visualisation output** per run:
- `frames_topo/` → density field frames (grayscale heatmap + strain-energy)
- `frames_topo_binary/` → binary threshold + extracted polygon frames
- `frames_topo_fea/` → FEA stress-map frames every `fea_interval` iterations
- `convergence_topo_density.gif`, `convergence_topo_binary.gif`, `convergence_topo_fea.gif`
- `topo_convergence.png` — 4-panel static plot (compliance, Vf, MOI, Δρ)
- `topo_final_stress.png` — high-quality FEA stress map of the final design

---

## Profile Families

The enhanced optimizer supports four outer-profile parametrizations, all sharing
the same DE bounds (`N × (r_min, r_max)`) and controlled by `profile_type` in
the config.

| Key | Continuity | Support | Notes |
|-----|-----------|---------|-------|
| `fourier` | C∞ | Global | Baseline only — frozen. Every coefficient affects the entire profile; susceptible to Gibbs oscillation near sharp features. |
| `bspline` | C² | Local (~±2/N) | Default enhanced. scipy `splprep(per=True, k=3)`. Smooth convergence; control point i shifts profile only near angle θ_i. |
| `bezier` | C¹ | Local (~±2/N) | Composite cubic Bézier. Tangents from Catmull-Rom finite differences (T_i = 0.5·(P_{i+1} − P_{i-1})). Approximates control points. |
| `catmull_rom` | C¹ | Local (~±2/N) | Centripetal Catmull-Rom (α=0.5, Barry-Goldman algorithm). Interpolates all control points exactly. |

All three spline families produce visually similar profiles; differences appear
mainly in how the optimizer converges.  B-spline tends to give the smoothest
shapes; Catmull-Rom passes through control points so teeth are positioned more
predictably.

---

## Bite Evaluation

### Baseline — Analytic Formula

`physics.bite_mm(n_teeth, rpm)` returns a constant value determined purely by
weapon style and RPM.  Geometry cannot influence it.  This is a known
limitation documented intentionally for the research comparison.

### Enhanced — Kinematic Spiral Simulation (`objectives_enhanced.kinematic_spiral_bite`)

In the weapon's rotating reference frame the approaching opponent's face traces
an Archimedean spiral:

    r_enemy(θ) = r_start − v_per_rad · θ

where `v_per_rad = v_approach / ω` (mm per radian of weapon rotation).

Each zero-crossing from outside to inside the radial profile is one tooth
contact.  Bite depth per contact:

    bite_mm = v_per_rad · 2π / n_contacts

| Profile shape | n_contacts | bite vs. max |
|---------------|-----------|-------------|
| Smooth disk | 1 | 100 % |
| 3 large teeth | 3 | 33 % |
| 12 small teeth | 12 | 8 % |

This correctly rewards fewer, larger protrusions and penalizes saw-blade
profiles.  Default approach speed: 3 m/s.

### Spiral Contact Analysis (`spiral_contact.py`)

A complementary analysis tool (not used inside the optimization loop by
default).  Runs a *family* of N spirals at uniformly-spaced initial angles
θ₀ ∈ [0, 2π) and returns a `ContactResult` for each:

- `r_contact` — radius of first contact
- `bite_depth = r_start − r_contact` — how far inward the opponent reached
- `force_direction` — unit tangential force vector opposing weapon rotation

The bite-depth distribution distinguishes designs:

| Design | Bite-depth range (v=10 m/s, 1000 rpm) |
|--------|--------------------------------------|
| Smooth disk (R=80 mm) | 1.7 mm (all spirals identical) |
| 3-tooth weapon (R_mean=70, A=22 mm) | 2–46 mm (varies strongly with θ₀) |

`contact_forces()` converts results to `{x, y, fx, fy}` dicts that can be
passed to `fea_stress_analysis_with_mesh(contact_forces=...)` to superimpose
tangential impact loads on the centrifugal body load.

---

## Structural Analysis

### Geometric Proxy — Baseline (`structural.py`)

Used inside the baseline optimization loop (fast, ~0 ms extra cost):

- Minimum cross-section width via radial ray casting
- Wall-to-hole and hole-to-hole distances
- Penalises features thinner than `min_wall_thickness_mm`

### 2D FEA — Enhanced (`fea.py`)

Lightweight constant-strain triangular (CST) finite element analysis run at
every objective call in enhanced mode:

- Delaunay triangulation of the weapon polygon; grid spacing controlled by
  `fea_coarse_spacing_mm` (default 10 mm during optimization) or
  `fea_fine_spacing_mm` (default 4 mm for final renders)
- Linear elastic plane-stress; mm–N–MPa–tonne unit system
- **Centrifugal body load**: `f = ρ ω² r · A · t`, distributed to element nodes
- **Contact point loads** (optional): tangential impact forces from
  `spiral_contact.contact_forces()` added to nearest mesh nodes via KDTree
- Peak von Mises stress → safety factor → `fea_score` in [0, 1]
- FEA-based `structural_integrity` replaces the geometric proxy in the
  enhanced weighted score

Safety factor → score mapping:

    sf ≥ 2.0  → fea_score = 1.0
    1 ≤ sf < 2 → fea_score = 0.5 + 0.5·(sf − 1)
    sf < 1    → fea_score = 0.5·sf

FEA is approximate (2D plane stress, linear elastic, no dynamic impact) but
reliably identifies stress concentrations around holes and thin sections that
the geometric proxy misses.

---

## Topology Optimisation

An alternative to discrete cutout placement (Phase 2).  Activated by setting
`cutout_type = "topology"` in the config.  Implemented in `topo_optimizer.py`.

### Method: SIMP

Solid Isotropic Material with Penalisation (SIMP):

    E_eff(ρ) = E_min + ρᵖ · (E₀ − E_min)    p = 3 (default)

where `ρ ∈ [0, 1]` is the element density variable and `E_min = 1e-9 · E₀` is
a small lower bound to prevent singularity.

### Objective

    min  J = w_C · Ĉ − w_MOI · IMOI_hat

where `Ĉ` is normalised compliance (structural flexibility) and `IMOI_hat` is
normalised second moment of area.  Default `w_C = w_MOI = 0.5` balances
stiffness against energy storage.

### Constraint

    Σ ρ_e · A_e / A_total = V_f

where `V_f = mass_budget / mass_solid` (volume fraction from the weight limit).

### Update rule

Optimality Criteria (OC) with bisection on the Lagrange multiplier for the
volume constraint; density filter radius `r_min = topo_r_min_factor × mesh_spacing`
enforces a minimum length scale and prevents checkerboard patterns.

### Fixed-solid regions

- **Hub annulus**: nodes with `r ≤ bore_r × 2.5` are always fully dense (`ρ = 1`).
- **Outer rim** (optional, `topo_fix_rim = true`): strip `r ≥ max_r × 0.88` kept
  solid to maintain a continuous striking edge.

### Post-processing

After convergence an adaptive threshold `τ` is found by binary search such that
the resulting binary mask meets the volume fraction exactly.  Void triangles are
unioned into a Shapely MultiPolygon, buffered for smoothness, then subtracted
from the outer profile — producing a clean weapon polygon compatible with
`geometry.assemble_weapon()` and the DXF exporter.

### Outputs

| Directory / file | Contents |
|-----------------|---------|
| `frames_topo/` | Density field animation frames |
| `frames_topo_binary/` | Binary (thresholded) density frames |
| `frames_topo_fea/` | FEA stress map on the binary geometry, per SIMP iteration |
| `topo_convergence.png` | Compliance + volume fraction convergence curves |
| `topo_final_stress.png` | Fine-mesh FEA stress map of the final topology |

---

## Objective Weights and Scoring

### Baseline (`objectives.weighted_score`)

| Objective | Default weight | Sub-score definition |
|-----------|---------------|---------------------|
| `moment_of_inertia` | 0.30 | I / I_max_solid (clamped to 1) |
| `bite` | 0.15 | 1 − |bite − 20 mm| / 20 mm  (targets 20 mm ideal) |
| `structural_integrity` | 0.20 | Geometric proxy score [0, 1] |
| `mass_utilization` | 0.10 | mass / budget (penalised heavily if > 1) |
| `balance` | 0.10 | 1 − CoM offset / (0.1 · R_max) |
| `impact_zone` | 0.15 | Fraction of perimeter with solid striking zone |

### Enhanced (`objectives_enhanced.weighted_score_enhanced`)

Same structure; two components change:

| Objective | Change |
|-----------|--------|
| `bite` | Monotone: `bite_mm / max_bite_mm` where `max_bite_mm = v/f` (no arbitrary 20 mm target) |
| `structural_integrity` | FEA-based `fea_score` instead of geometric proxy |

All weights are configurable per design in `optimization.weights`.

---

## Algorithm: Differential Evolution

SciPy `differential_evolution` with dithered mutation `(0.5, 1.5)` and
`recombination=0.8`.  Population-based, naturally parallel.

Default population size: 60 (enhanced), 40 (baseline).  Workers: all
available cores (process-based to avoid Shapely GIL issues).

Phase iteration caps default to 50 % / 25 % of `max_iterations`.  Override
explicitly with `phase1_iters` and `phase2_iters` when you need asymmetric
splits (e.g., more Phase 1 budget to converge the outer shape first).

---

## Convergence GIF and Sidecar Files

When `fea_interval > 0`, the optimizer saves a two-panel FEA stress-map frame
every N steps:

```
runs/<run_id>/frames_p1/
    frame_0000.png          rendered stress map
    frame_0000.npz          nodes (N×2), elements (M×3), vm_stresses (M,), polygon_xy, holes_xy
    frame_0000_meta.json    step, phase, score, metrics, cfg_snapshot
    ...
convergence_phase1.gif      assembled animation
```

`fea_replay.py` can re-render and re-assemble GIFs from the `.npz` sidecars
post-hoc with different colourscales, DPI, or frame ranges — without
re-running the optimizer.

---

## sweep_profiles.py

Runs all four profile families across a fixed set of test cases and generates
comparison charts:

```bash
python3 sweep_profiles.py \
    --iterations 100 --popsize 30 \
    --phase1-iters 60 --phase2-iters 20 \
    --fea-interval 5 --output-dir profile_sweep --resume
```

Outputs per-case convergence plots, final-score bar charts, and a heatmap of
score × method × case.
---

## fea_speed_sweep.py

Standalone post-hoc analysis tool.  Loads a completed weapon DXF, sweeps
through a logarithmic range of RPMs, and produces a 3-D figure with the
weapon's FEA stress field stacked along the speed axis.

### Two load cases, combined per RPM slice

| Load | Expression | Scaling |
|------|-----------|---------|
| Centrifugal body force | `f = ρ ω² r · A · t` (distributed to all elements) | ω² |
| Spiral-impact force | `F = ½ · m · ω² · r̄_contact` (centripetal reaction at mean contact radius) | ω² |

The impact force is **distributed equally across all contact points** detected
in one revolution.  A weapon with N contacts receives `F/N` at each site —
representing the physical reality that symmetric teeth share the load.  A single
large tooth receives the full `F`.

### Usage

```bash
python3 fea_speed_sweep.py output_disk.dxf output_disk_stats.json \
    --rpm-min 500 --rpm-max 12000 --rpm-steps 10 \
    --mesh-spacing 5.0 --out speed_sweep.png
```

### Figure layout

| Panel | Contents |
|-------|---------|
| 3-D stack (left) | Weapon cross-section at each RPM level.  Colour = σ_VM / σ_yield.  Yellow curve = Archimedean spiral.  Orange dots = contact points. |
| Safety factor (top-right) | SF centrifugal-only (blue) and combined (red) vs. RPM.  Green / orange / red bands mark safe, marginal, and failed zones. |
| Peak stress (top-right) | σ_peak centrifugal and combined vs. RPM; yield line shown. |
| Worst-case heatmap (bottom-centre) | 2-D stress field at maximum RPM with all contact points marked. |
| Summary table (bottom-right) | Safe RPM limit, failure RPM, bite depth at key speeds, impact force. |

### Stiffness matrix caching

The CST stiffness matrix **K** is assembled once from the mesh geometry and
reused for all RPM levels.  Only the load vectors `F_centrifugal + F_impact`
are recomputed per slice, making the sweep O(n_RPM) in solve cost rather than
O(n_RPM × n_elements²).
