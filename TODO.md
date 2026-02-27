# TODO

## Research comparison (high priority)

- [ ] Run full `--mode compare` evaluation across all cases and record baseline vs. enhanced delta table
- [ ] Establish statistical significance: run each case 3–5 times and report mean ± std for each objective
- [ ] Document intentional design weaknesses of baseline in paper/writeup (see MEMORY.md §Known design weaknesses)
- [ ] Verify enhanced spiral-bite values against physical intuition for each archetype (smooth disk → max bite, multi-tooth → penalized)
- [ ] Add a `--mode enhanced-only` fast path that skips baseline for iteration speed

## Test coverage (medium priority)

- [x] `tests/test_geometry.py` — assemble_weapon(), hole subtraction, connectivity checks (done)
- [x] `tests/test_constraints.py` — constraint_penalty(), check_envelope(), validate_geometry() (done)
- [x] `tests/test_objectives.py` — weighted_score(), impact_zone_score() round-trip (done)
- [x] `tests/test_parametric.py` — Fourier profile round-trip, cutout placement, no self-intersection (done)
- [x] `tests/test_bspline_profile.py` — B-spline eval, periodicity, C2 continuity at join (done)
- [x] `tests/test_profile_splines.py` — bezier/catmull_rom: valid Polygon area > 0, closure, no self-intersection for all archetypes (done)
- [x] `tests/test_profile_builder.py` — dispatcher: all four profile_type values return valid Polygon at bounds midpoint (done)
- [x] `tests/test_fea_replay.py` — sidecar roundtrip: save .npz + meta JSON, reload, reconstruct polygon (done)
- [x] `tests/test_parametric_cad.py` — polar cutout superellipse, analytic mass normalisation convergence (done)
- [x] `tests/test_fea.py` — FEA on known-geometry annular ring, stress < yield for safe design (done)
- [x] `tests/test_objectives_enhanced.py` — spiral bite: smooth disk → max bite, N-tooth → 1/N bite (done)
- [x] `tests/test_topo_optimizer.py` — volume fraction constraint satisfaction, mass error < 10%, GIF output exists (done; heavy tests marked @pytest.mark.slow)

## Documentation (medium priority)

- [x] `docs/ARCHITECTURE.md` — swap-point table, permutation matrix, data flow diagram (done)
- [x] `docs/OPTIMIZATION.md` — topology optimisation Phase-2 section, updated problem characteristics table (done)
- [x] `README.md` — topology module entry, config reference table updated (done)
- [ ] `docs/ENHANCED.md` — describe enhanced stack architecture, B-spline profile, polar cutouts, spiral bite physics, FEA-in-loop tradeoffs
- [ ] `docs/TOPOLOGY.md` — extended write-up of SIMP theory, sensitivity analysis derivation, OC convergence proof, design trade-off charts
- [ ] Add docstrings to `bspline_profile.py`, `parametric_cad.py`, `objectives_enhanced.py`, `optimizer_enhanced.py`, `topo_optimizer.py`
- [ ] Document the MAP-Elites + RRT* diversity strategy in `docs/` (currently only in `explore_diverse.py` docstring)

## Code quality (low priority)

- [x] Add `.gitignore` (done)
- [x] `scripts/compare_baselines.py` — comprehensive baseline vs. enhanced comparison script with N replicates, mean±std, Welch t-tests, all 10 cases (done)
- [ ] `optimizer_enhanced.py`: `_constraint_penalty_no_mass()` is a local copy — refactor `constraints.py` to accept a mask of active constraints instead
- [ ] Phase 2 mass normalization iterates up to 3× analytically — add a convergence warning if tolerance not met after 3 iterations
- [ ] `fea_viz.py` frame resize: document why `optimize=False` is required for Pillow GIF assembly
- [ ] Pin dependency versions in `pyproject.toml` for reproducible research builds

## Features / future work

- [x] **Bézier outer profile** (`profile_splines.py`): composite cubic, C¹ joins, local support (done)
- [x] **Catmull-Rom outer profile** (`profile_splines.py`): centripetal, interpolates control points (done)
- [x] **Profile dispatcher** (`profile_builder.py`): `profile_type` config key routes to any family (done)
- [x] **FEA sidecar files** (`optimizer_enhanced.py`): `.npz` + `_meta.json` saved alongside PNGs (done)
- [x] **FEA replay tool** (`fea_replay.py`): assemble GIFs from sidecars without re-running optimizer (done)
- [x] **Topology optimisation** (`topo_optimizer.py`): SIMP Phase-2 alternative with OC update, density filter, adaptive threshold, 3× GIF output (done)
- [x] **FEA speed sweep** (`fea_speed_sweep.py`): 3-D stacked heatmap, centrifugal + balanced multi-contact spiral-impact loading across RPM range, operating range identification (done)
- [ ] **Adaptive Fourier refinement** (from OPTIMIZATION.md): start coarse (2 terms), optimize, add terms, re-optimize from solution
- [ ] **CMA-ES Phase 1**: replace DE with CMA-ES for outer profile — better convergence in smooth 10-50D landscape (scipy-cma or `cma` package)
- [ ] **Contact/rake angle objective**: concentrate tip mass and optimize rake angle for improved energy transfer at impact
- [ ] **3D FEA or impact FEA**: current 2D plane-stress FEA misses through-thickness bending under impact; consider a fast beam model
- [ ] **Bayesian optimization option**: worth exploring if FEA call cost increases with finer mesh
- [ ] **Export to STEP/IGES**: ezdxf can't do solids; explore CadQuery or FreeCAD Python API for 3D export
- [ ] **Tooth geometry primitives**: explicit tooth rake/relief angle parameters rather than relying on Fourier/B-spline to discover tooth shapes
- [ ] **GUI / web frontend**: simple parameter editor with live preview (e.g., Gradio or Streamlit) for non-CLI users
