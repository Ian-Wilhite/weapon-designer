"""Staged evaluation gate for the enhanced optimizer.

Avoids running expensive FEA on every candidate by applying a three-stage
funnel:

  Stage 0 (µs)  — polygon validity, self-intersection, min area
  Stage 1 (ms)  — mass, MOI, spiral contact proxy, geometric structural score
  Stage 2 (FEA) — full compute_metrics_enhanced; only run when Stage 1 score
                  is in the top `gate_frac` of the rolling population

Setting gate_frac=1.0 (the default) disables gating and every candidate
proceeds to FEA — identical to the current behaviour.

Thread safety
-------------
`EvalGate` is shared across the parallel DE worker processes via a
`multiprocessing.Manager` list.  To avoid deadlock-prone Manager proxies,
we instead use a module-level `threading.Lock` + a `collections.deque`
capped at `_HISTORY_MAX` entries.  Scipy's `workers=-1` mode uses
`multiprocessing` (not threading), so we can't share state between workers
at the *process* level cheaply.  Instead each *worker process* gets its own
gate instance (via pickling); the gate starts empty in each worker and builds
up its own Stage-1 history.  This is conservative (each worker warms up
independently) but correct and race-free.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from .config import WeaponConfig

_HISTORY_MAX = 500   # rolling window for Stage-1 score percentile


class EvalGate:
    """Three-stage evaluation gate.

    Parameters
    ----------
    cfg       : weapon configuration (reads structural thresholds)
    gate_frac : fraction of candidates that proceed from Stage 1 to FEA.
                1.0 = all proceed (current behaviour, gate disabled).
    warmup    : number of Stage-1 evaluations before gating kicks in.
                During warm-up all candidates proceed to FEA.
    """

    def __init__(
        self,
        cfg: WeaponConfig,
        gate_frac: float = 1.0,
        warmup: int = 20,
    ) -> None:
        self.cfg       = cfg
        self.gate_frac = float(gate_frac)
        self.warmup    = int(warmup)
        self._lock     = threading.Lock()
        self._history: deque[float] = deque(maxlen=_HISTORY_MAX)
        self._n_total  = 0    # total candidates seen
        self._n_skipped = 0   # candidates that were gate-rejected at Stage 1

    # ------------------------------------------------------------------
    # Stage 0 — geometry validity (µs)
    # ------------------------------------------------------------------

    def stage0_check(self, poly: Polygon | MultiPolygon) -> bool:
        """Return True if the polygon is valid and worth evaluating further.

        Checks:
        - Polygon is not empty
        - Shapely is_valid (no self-intersections)
        - Area above a minimum threshold (avoids degenerate shapes)
        - (Optional) Applies GeometryConditioner if mfg_concave_radius_mm > 0
          to enforce minimum feature size before Stage-1 evaluation.
          If conditioning collapses the polygon it is rejected here.
        """
        if poly is None or poly.is_empty:
            return False
        if not poly.is_valid:
            return False
        min_area = np.pi * (self.cfg.mounting.bore_diameter_mm / 2.0 + 5.0) ** 2
        if poly.area < min_area:
            return False

        # GeometryConditioner gate: reject if conditioning collapses the shape.
        # Only applies when gate_frac < 1 (gating is active) to avoid overhead
        # on every candidate when running in full-pass mode.
        if self.gate_frac < 1.0:
            r_min = float(getattr(self.cfg.optimization, "mfg_concave_radius_mm", 0.0))
            if r_min > 0.0:
                try:
                    from .manufacturability import GeometryConditioner
                    method = str(getattr(self.cfg.optimization, "mfg_method", "minkowski"))
                    gc = GeometryConditioner(R_min_mm=r_min, method=method)
                    outer = Polygon(poly.exterior) if isinstance(poly, Polygon) else poly
                    conditioned, _ = gc.condition_weapon(poly, outer)
                    if conditioned is None or conditioned.is_empty:
                        return False
                except Exception:
                    pass  # Conditioner unavailable — do not reject on its behalf

        return True

    # ------------------------------------------------------------------
    # Stage 1 — cheap metrics (ms)
    # ------------------------------------------------------------------

    def stage1_score(self, poly: Polygon | MultiPolygon) -> float:
        """Compute a fast scalar score using only geometric and kinematic metrics.

        No FEA.  Uses:
        - mass vs budget (0–1)
        - MOI (normalised to envelope area estimate)
        - spiral bite proxy (n_teeth from outer profile, no full spiral simulation)
        - geometric structural score (wall thickness / min section width)

        Updates the internal rolling history for percentile gating.
        Returns the score (higher = better, consistent with full score).
        """
        from .physics import polygon_mass_kg, mass_moi_kg_mm2
        from .structural import structural_score
        from .objectives_enhanced import detect_teeth

        cfg = self.cfg
        try:
            mass   = polygon_mass_kg(poly, cfg.material.density_kg_m3, cfg.sheet_thickness_mm)
            moi    = mass_moi_kg_mm2(poly, cfg.material.density_kg_m3, cfg.sheet_thickness_mm)
            teeth  = detect_teeth(poly)
            struct = structural_score(
                poly,
                cfg.optimization.min_feature_size_mm,
                cfg.optimization.min_wall_thickness_mm,
            )

            # Normalise each term to [0, 1]
            mass_budget = cfg.weight_budget_kg
            mass_util   = min(1.0, mass / max(mass_budget, 1e-6))
            mass_score  = 1.0 - abs(mass_util - 1.0)   # peak at mass == budget

            max_r  = cfg.envelope.max_radius_mm
            max_moi = mass_budget * max_r ** 2          # rough upper bound
            moi_score = float(np.clip(moi / max(max_moi, 1.0), 0.0, 1.0))

            n_t    = max(teeth["n_teeth"], 1)
            bite_score = 1.0 / n_t                      # fewer teeth = more bite

            s1 = 0.3 * moi_score + 0.3 * struct + 0.2 * mass_score + 0.2 * bite_score

        except Exception:
            s1 = 0.0

        with self._lock:
            self._history.append(s1)
        return s1

    # ------------------------------------------------------------------
    # Stage 2 gate decision
    # ------------------------------------------------------------------

    def should_run_fea(self, stage1_score: float) -> bool:
        """Return True if this candidate should proceed to full FEA evaluation.

        During warm-up (first `warmup` evaluations) always returns True.
        After warm-up, returns True only if stage1_score is in the top
        `gate_frac` fraction of observed Stage-1 scores.
        """
        with self._lock:
            self._n_total += 1
            n_seen = len(self._history)

        if self.gate_frac >= 1.0:
            return True

        if n_seen < self.warmup:
            return True

        with self._lock:
            history_arr = np.array(self._history)

        threshold = float(np.percentile(history_arr, (1.0 - self.gate_frac) * 100.0))
        passes = stage1_score >= threshold

        if not passes:
            with self._lock:
                self._n_skipped += 1

        return passes

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def skip_rate(self) -> float:
        """Fraction of candidates rejected by the Stage-1 gate."""
        with self._lock:
            total = self._n_total
            skipped = self._n_skipped
        if total == 0:
            return 0.0
        return skipped / total

    def summary(self) -> str:
        """Human-readable gate statistics."""
        with self._lock:
            n = self._n_total
            s = self._n_skipped
            hist_len = len(self._history)
        return (
            f"EvalGate(gate_frac={self.gate_frac:.2f}): "
            f"{n} candidates seen, {s} skipped ({100*s/max(n,1):.1f}%), "
            f"history_size={hist_len}"
        )
