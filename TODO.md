# TODO — Weapon Designer Research Project

> **Parallel-module philosophy:** every category should have multiple swappable
> implementations so results can be compared directly. Never modify baseline files
> (`parametric.py`, `objectives.py`, `optimizer.py`, `physics.py`, `structural.py`).
> Add new sibling modules and register them in the relevant dispatcher or config flag.

---

## Priority legend
- `[P0]` blocks meaningful experiments — do first
- `[P1]` high research value, needed for paper
- `[P2]` useful improvements, do incrementally
- `[P3]` nice-to-have / future work

---

## 0. Run infrastructure & stability  `[P0]`

### 0.1 Staged evaluation gate  ✓
- [x] Create `staged_eval.py` with three gate stages:
  - **Stage 0** (µs): polygon validity, self-intersection check, min-web
    thickness, manufacturability pass/fail
  - **Stage 1** (ms): mass, MOI, spiral engagement (no FEA), structural proxy
  - **Stage 2** (expensive): FEA — only if Stage 1 score in top X% of population
- [x] Wire gate into `optimizer_enhanced.py` objective functions
- [x] Config key: `staged_eval_gate: float = 1.0` (1.0 = all reach FEA, current behaviour)
- [x] Expected speedup: 5–20× without sacrificing solution quality

### 0.2 Memory / per-step callback  ✓
- [x] Default `fea_interval = 0` in all sweep scripts (eliminates per-step double
  FEA + matplotlib heap growth that caused the 12-hr run slowdown)
- [x] Add `del fig` after every `plt.close(fig)` in `fea_viz.py`
- [x] Log wall-clock time spent inside `_FEACallback.__call__` so overhead is visible

### 0.3 Config additions  `config.py`  ✓
Added to `OptimizationParams` (all backward-compatible defaults):
```python
staged_eval_gate: float = 1.0
mfg_concave_radius_mm: float = 2.0
mfg_method: str = "minkowski"        # "vertex" | "minkowski"
use_ks_stress: bool = False          # False = current max-stress
ks_rho: float = 20.0
use_continuous_bite: bool = False    # False = current discrete contact count
sigmoid_beta: float = 10.0
use_robust_normalization: bool = False
trust_region_enabled: bool = False
trust_region_d_max_mm: float = 5.0
structured_seed_frac: float = 0.0   # 0 = pure LHS (current)
structured_seed_n_range: tuple = (1, 5)
interior_type: str = "superellipse" # "superellipse"|"spoke"|"slot"|"web"|"topology"
```

---

## 1. Parameterization — new parallel modules  `[P1]`

> Existing: `parametric.py` (Fourier baseline), `bspline_profile.py` (B-spline),
> `profile_splines.py` (Bézier, Catmull-Rom). Dispatcher: `profile_builder.py`.

### 1.1 Functional / lobed profiles  `functional_profiles.py`  ✓
- [x] `lobed_profile(n_teeth, R0, A1, A2, A3, phi, N_ctrl)` → radii vector
  using `r(θ) = R0 + A1·cos(n·(θ−φ)) + A2·cos(2n·(θ−φ)) + A3·cos(3n·(θ−φ))`
- [x] `functional_seed_bank(cfg, n_range=(1,5), k_per_n=10)` → list of radii vectors
- [x] Choose `n` from spiral parameters to hit a target bite:
  `n_contacts ≈ v_rad·2π / b_target`
- [x] Register as `profile_type = "functional"` in `profile_builder.py`

### 1.2 Low-dimensional functional optimizer  `optimizer_functional.py`  ✓
- [x] Stage A: 6-dim DE on `[n_teeth, R0, A1, A2, A3, phi]` (popsize=20, fast landscape)
- [x] Stage B: lift to N-dim B-spline, warm-start from `lobed_radii(stage_a_params)`
- [x] Score-plateau proximity annealing: λ halved each time score stagnates for `patience` steps
- [x] Entry point `optimize_functional(cfg, case_dir)`, wired into `evaluate.py --mode functional`
- [ ] Compare convergence against direct N-dim radii search (paper result)  ← needs run data

### 1.3 Adaptive knot-spacing B-spline  `[P2]`
- [ ] `adaptive_bspline_profile.py`: place more knots where profile curvature is
  highest; compare vs. uniform-spacing B-spline at same N

### 1.4 Adaptive Fourier refinement  `[P2]`
- [ ] Start with 2 Fourier terms, optimize, add terms, re-optimize from solution
  (coarse-to-fine schedule)

---

## 2. Population seeding  `seeding.py`  `[P1]`  ✓

- [x] `functional_seeds(cfg, n_range, k_per_n)` — lobed radii vectors
- [x] `archetype_seeds(cfg)` — existing archetypes resampled to radii vector
- [x] `phase_sweep_seeds(cfg, n, n_phases=24)` — sweep φ, keep top-k by cheap metric
- [x] `perturb_seed(r, sigma, k)` — Gaussian cloud of k variants around a seed
- [x] `mixed_init_population(cfg, pop_size, structured_frac=0.25)` → array for
  `init=` in `differential_evolution`
- [x] Wire `init=mixed_init_population(...)` into `optimizer_enhanced.py`
- [ ] Apply manufacturability conditioning to seeds before returning (phenotype =
  genotype)  ← optional P2 improvement

---

## 3. Manufacturability  `manufacturability.py`  `[P1]`  ✓

- [x] `GeometryConditioner` class:
  - **Method A** (vertex): classify convex/concave by turning-angle sign;
    replace concave corners with arcs of radius `R_min`
  - **Method B** (Minkowski): `P' = buffer(buffer(P, −R), +R)` on interior
    cutout boundaries only (not outer profile)
  - Returns conditioned polygon + `conditioning_delta` dict (area change,
    n_corners_modified, max curvature before/after)
- [x] Config keys: `mfg_concave_radius_mm`, `mfg_method`
- [x] Standalone script: `scripts/apply_mfg.py <dxf>`
- [x] Tests: `tests/test_manufacturability.py` (22 tests, all pass)
- [ ] Apply in Stage 0 of staged evaluator  ← P2 improvement
- [ ] Apply to seeds in `seeding.py`  ← P2 improvement

---

## 4. Objectives — smooth/stable variants  `[P1]`

> Existing: `objectives.py` (baseline, frozen), `objectives_enhanced.py`.
> Add `objectives_smooth.py` as a third parallel implementation.

### 4.1 `objectives_smooth.py`  ✓
- [x] **KS stress aggregator** replaces σ_VM_max:
  `σ_KS = (1/ρ)·log(Σ_e exp(ρ·σ_e/σ_ref))` with config param `ks_rho`
- [x] **Continuous bite metric** replaces discrete contact count:
  - `C(θ) = sigmoid(β·(r_profile(θ) − r_enemy(θ)))` along spiral
  - `engagement = ∫C(θ)dθ`, peaks via `scipy.signal.find_peaks`
  - Expose `sigmoid_beta` config param
- [x] **Robust online normalization**:
  - Rolling `(median, IQR)` over last 200 evaluations per term
  - `m̃_j = (m_j − median_j) / (IQR_j + ε)` before weighting
  - Eliminates the "one term dominates because its scale changed" problem

### 4.2 Contact-as-traction FEA loading  `[P1]`  ✓
- [x] `_find_boundary_edges()` — O(M) edge-occurrence counter; identifies boundary edges
- [x] `_apply_contact_forces(mode="neumann_edge")` — distributes force between the two
  nearest boundary nodes by projecting contact point onto the edge (linear interpolation)
- [x] `contact_load_mode: str = "neumann_edge"` config key (backward-compat default)
- [x] `fea_stress_envelope(poly, cfg, n_patches=12)` — superimposed envelope load, KS aggregation
- [x] Contact force scale corrected: `contact_force_scale=0.02` reduces centripetal force
  from ~85 kN to ~1.7 kN per contact (physically appropriate for linear FEA at 10 mm mesh)

---

## 5. Optimizer — new parallel strategies  `[P1]`

> Existing: `optimizer.py` (baseline DE), `optimizer_enhanced.py` (enhanced DE).

### 5.1 Shape-distance trust region  ✓  (in `optimizer_enhanced.py`)
- [x] Module-level `_TR_BEST_X` updated by callback; objective early-rejects candidates
  with `_shape_distance(x, _TR_BEST_X) > trust_region_d_max_mm` (saves FEA cost entirely)
- [x] Forces `workers=1` when TR enabled (subprocess workers can't see main-process updates)
- [x] Config keys: `trust_region_enabled`, `trust_region_d_max_mm`

### 5.2 Shape-distance logging in existing optimizer  ✓  `[P0]`
- [x] In `_FEACallback.__call__`: compute and log `shape_distance(xk, x_best)`,
  store in sidecar JSON meta as `"shape_distance"` key
- [x] Reveals whether DE is converging or thrashing (key diagnostic)

### 5.3 CMA-ES Phase 1  `[P2]`
- [ ] Replace DE with CMA-ES for outer profile optimization (better in smooth 10–50D)
- [ ] Use `cma` package or `scipy`'s experimental CMA interface
- [ ] Compare convergence curve vs. DE (paper result)

### 5.4 Surrogate-accelerated optimizer  `optimizer_surrogate.py`  `[P2]`
- [ ] GP surrogate on Stage 1 scores to pre-screen DE candidates
- [ ] Run FEA only where GP uncertainty is high OR predicted score beats threshold
- [ ] Compare convergence and wall-clock cost vs. plain DE

### 5.5 Bayesian optimization option  `[P3]`
- [ ] Worth exploring if fine-mesh FEA cost increases further

---

## 6. Cutout / interior representation — rethink  `[P1]`

> Current Phase 2: `parametric_cad.py` (polar superellipses) — not working well.
> `topo_optimizer.py` (SIMP) is the alternative.
> Add structured-template approach as a third option.

### 6.1 Web/spoke/slot templates  `interior_templates.py`  ✓
- [x] Parameterizations:
  - **Spoke**: `n_spokes, spoke_width_mm, hub_r, rim_t` → void polygons
  - **Slot**: `n_slots, slot_width_mm, slot_r_inner, slot_r_outer, phi_offset`
  - **Web offset**: single `web_thickness_mm` → inward-offset ring
- [x] Dispatcher: `build_interior(interior_type, params, cfg)` in `profile_builder.py`
- [ ] All outputs pre-conditioned to `mfg_concave_radius_mm`  ← P2 improvement
- [ ] Compare vs. SIMP and vs. superellipses (paper table)  ← requires run data

### 6.2 SIMP with manufacturability guardrails  (improve `topo_optimizer.py`)
- [ ] Add n-fold rotational symmetry constraint option
- [ ] Add projection continuation for sharper 0/1 convergence
- [ ] Post-process with `GeometryConditioner` before FEA scoring

---

## 7. Diagnostics and analysis scripts  `[P1]`

### 7.1 Sensitivity probe  `scripts/sensitivity_probe.py`  ✓
- [x] Given a saved best-solution JSON, perturb each control index ±δ
- [x] Log: `Δscore, ΔMOI, Δbite, ΔσFEA, Δstructural, Δcom_offset` per index
- [x] Output: sensitivity summary JSON + optional bar-chart PNG (`--plot`)
- [x] No FEA by default (Stage-1); `--fea` flag for full scoring

### 7.2 Objective landscape slices  `scripts/landscape_probe.py`  ✓
- [x] 2D grid sweep through two control-point axes (configurable via `--dim1 --dim2`)
- [x] Stage 1 only by default (cheap); `--fea` flag for full scoring
- [x] Output heatmap PNG + raw .npy grid for further analysis

### 7.3 Convergence comparison  `scripts/plot_convergence.py`  ✓
- [x] Load all `output_stats.json` from run directories / recursive glob
- [x] Plot best-so-far score vs. wall-clock time AND vs. DE step
- [x] Outputs `score_vs_time.png` + `score_vs_step.png`
- [ ] Overlay mean ± std across replicates  ← needs compare_baselines data
- [ ] Normalize x-axis by n_evaluations  ← P2 improvement

### 7.4 Shape distance over optimization  `scripts/plot_shape_distance.py`  ✓
- [x] Load sidecar meta JSONs from `frames_p1/` or `frames_p2/` directories
- [x] Also supports `--from-stats` to read inline convergence history
- [x] Plot `score` + running best + `shape_distance` vs. step (dual panel)
- [x] Marks first step where shape distance drops below 0.5mm (convergence indicator)

---

## 8. Research comparison  `[P1]`

- [ ] Run full `--mode compare` evaluation; record baseline vs. enhanced delta table
- [ ] Statistical significance: 3–5 replicates per case, report mean ± std
- [ ] Verify enhanced spiral-bite values against physical intuition per archetype
- [ ] Results table comparing profile families (score, convergence, shape distance)
- [ ] Results table comparing objective variants (discrete vs. continuous bite;
  max-stress vs. KS-stress)
- [ ] Results table comparing interior representations (superellipse / spoke / SIMP)
- [ ] Document intentional baseline weaknesses in paper (see MEMORY.md §Known design weaknesses)

---

## 9. Documentation  `[P2]`

- [ ] `docs/ENHANCED.md` — enhanced stack, B-spline profile, polar cutouts, spiral
  bite physics, FEA-in-loop tradeoffs
- [ ] `docs/TOPOLOGY.md` — SIMP theory, sensitivity derivation, OC convergence, trade-off charts
- [ ] `docs/MANUFACTURABILITY.md` — GeometryConditioner methods, conditioning delta metrics
- [ ] Add docstrings to `bspline_profile.py`, `parametric_cad.py`,
  `objectives_enhanced.py`, `optimizer_enhanced.py`, `topo_optimizer.py`
- [ ] Document MAP-Elites + RRT* diversity strategy in `docs/`

---

## 10. Code quality  `[P2]`

- [ ] `optimizer_enhanced.py`: `_constraint_penalty_no_mass()` is a local copy —
  refactor `constraints.py` to accept a mask of active constraints
- [ ] Phase 2 mass normalization: add convergence warning if 3-iteration tolerance
  not met
- [ ] `fea_viz.py`: document why `optimize=False` is required for Pillow GIF assembly
- [ ] Pin dependency versions in `pyproject.toml` for reproducible research builds
- [ ] `fea_viz.py`: add `del fig` after every `plt.close(fig)`

---

## 11. Future work  `[P3]`

- [ ] Contact/rake angle objective: concentrate tip mass and optimize rake angle
- [ ] 3D or impact FEA: current 2D misses through-thickness bending under impact
- [ ] Export to STEP/IGES via CadQuery or FreeCAD Python API
- [ ] Tooth geometry primitives: explicit rake/relief angle params
- [ ] GUI / web frontend (Gradio or Streamlit)

---

## Completed  ✓

- [x] B-spline outer profile (`bspline_profile.py`)
- [x] Bézier + Catmull-Rom profiles (`profile_splines.py`)
- [x] Profile dispatcher (`profile_builder.py`)
- [x] Polar superellipse cutouts (`parametric_cad.py`)
- [x] Spiral contact analysis (`spiral_contact.py`)
- [x] FEA contact-force loading (`fea.py`)
- [x] Enhanced objectives with spiral bite + FEA (`objectives_enhanced.py`)
- [x] Enhanced optimizer with two-phase DE (`optimizer_enhanced.py`)
- [x] SIMP topology optimizer (`topo_optimizer.py`)
- [x] FEA frame renderer + GIF assembler (`fea_viz.py`)
- [x] Spiral contact frame renderer (`fea_viz.py`)
- [x] FEA sidecar files (.npz + _meta.json)
- [x] FEA replay tool (`fea_replay.py`)
- [x] FEA speed sweep (`fea_speed_sweep.py`)
- [x] Sweep script with disk-only cases (`sweep_profiles.py`)
- [x] Comprehensive baseline comparison (`compare_baselines.py`)
- [x] 13 test files, 169+ tests passing
- [x] `.gitignore`, `docs/ARCHITECTURE.md`, `docs/OPTIMIZATION.md`, `README.md`
- [x] Staged evaluation gate (`staged_eval.py`) + wired into optimizer_enhanced
- [x] Shape-distance logging in _FEACallback (sidecar meta + convergence history)
- [x] Config additions (15 new P0/P1 fields to OptimizationParams)
- [x] Memory fix: `del fig` after `plt.close(fig)` in `fea_viz.py`
- [x] Functional/lobed profile type (`functional_profiles.py`) + dispatcher wiring
- [x] Population seeding (`seeding.py`) + wired into optimizer_enhanced (both phases)
- [x] Manufacturability conditioning (`manufacturability.py`, 22 tests, `apply_mfg.py`)
- [x] Smooth objectives (`objectives_smooth.py`): KS-stress, continuous bite, RobustNormalizer
- [x] Interior templates (`interior_templates.py`): spoke, slot, web-offset
- [x] Interior dispatcher in `profile_builder.py` (`build_interior`, `get_interior_bounds`)
- [x] Diagnostic scripts: `sensitivity_probe.py`, `landscape_probe.py`,
      `plot_convergence.py`, `plot_shape_distance.py`
- [x] Test suite: 201 tests pass
- [x] Contact-as-traction FEA loading (Neumann edge BCs, `fea_stress_envelope`)
- [x] Shape-distance trust region in `optimizer_enhanced.py`
- [x] Low-dimensional functional optimizer (`optimizer_functional.py`, Stage A+B)
- [x] Pareto runtime-vs-optimality metrics in `compare_baselines.py` + `plot_pareto.py`
- [x] Optimizer hyperparameter sweep (`sweep_optimizer_params.py`)
- [x] DE mutation/recombination exposed as config fields (`de_mutation_lo/hi`, `de_recombination`)
- [x] Contact quality scoring: quadratic penalty `bite_score = depth × quality²`
- [x] Impact velocity doubled to 6 m/s (`drive_speed_mps` config field)
- [x] Contact force corrected to ~1–2 kN (`contact_force_scale=0.02`)
- [x] Spiral contact viz: bite bars now scale from weapon boundary, always visible
