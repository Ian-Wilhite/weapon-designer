"""Smooth objective function variants for stable optimization.

Parallel to objectives_enhanced.py — never modifies that file.
Activated via config flags: use_ks_stress, use_continuous_bite,
use_robust_normalization.
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# KS stress aggregator
# ---------------------------------------------------------------------------

def ks_stress_aggregator(
    vm_stresses: np.ndarray,
    yield_mpa: float,
    rho: float = 20.0,
) -> float:
    """Kreisselmeier-Steinhauser smooth maximum of element stresses.

    Provides a differentiable approximation to max(vm_stresses) that is
    suitable for gradient-based and population-based optimization.

    Formula
    -------
        sigma_KS = (sigma_ref / rho) * log( sum_e exp(rho * sigma_e / sigma_ref) )

    where sigma_ref = yield_mpa.

    The log-sum-exp trick is used for numerical stability:
        m = max(a_i)
        log( sum exp(a_i) ) = m + log( sum exp(a_i - m) )

    Parameters
    ----------
    vm_stresses : array of per-element Von Mises stress values in MPa.
                  May be empty.
    yield_mpa   : yield strength in MPa used as the reference stress.
    rho         : aggregation sharpness parameter. Higher values make
                  sigma_KS approach max(sigma) more closely; lower values
                  give a smoother average. Typical range: 5–100.

    Returns
    -------
    float : KS-aggregated stress in MPa. Returns 0.0 for empty input.
    """
    if vm_stresses is None or len(vm_stresses) == 0:
        return 0.0

    sigma_ref = max(float(yield_mpa), 1e-6)
    arr = np.asarray(vm_stresses, dtype=float)

    # Numerically stable log-sum-exp
    a = rho * arr / sigma_ref
    m = float(a.max())
    lse = m + np.log(np.sum(np.exp(a - m)))

    return float(sigma_ref / rho * lse)


# ---------------------------------------------------------------------------
# KS-based safety factor
# ---------------------------------------------------------------------------

def ks_safety_factor(
    vm_stresses: np.ndarray,
    yield_mpa: float,
    rho: float = 20.0,
) -> float:
    """Safety factor in [0, 1] computed from KS-aggregated stress.

    Uses the same piecewise mapping as the baseline fea_stress_analysis():
        SF >= 2  ->  score = 1.0
        SF =  1  ->  score = 0.5
        SF <  1  ->  score = 0.5 * SF   (proportionally penalise overstress)

    The score is clamped to [0, 1].

    Parameters
    ----------
    vm_stresses : per-element Von Mises stresses (MPa).
    yield_mpa   : yield strength (MPa).
    rho         : KS aggregation sharpness (forwarded to ks_stress_aggregator).

    Returns
    -------
    float in [0, 1].
    """
    ks_stress = ks_stress_aggregator(vm_stresses, yield_mpa, rho)
    ks_stress = max(ks_stress, 1e-6)
    sf = float(yield_mpa) / ks_stress

    if sf >= 2.0:
        score = 1.0
    elif sf >= 1.0:
        # Linear interpolation: SF=1 -> 0.5, SF=2 -> 1.0
        score = 0.5 + 0.5 * (sf - 1.0)
    else:
        score = 0.5 * sf

    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# Continuous bite metric
# ---------------------------------------------------------------------------

def continuous_bite_metric(
    poly,
    rpm: float,
    drive_speed_mps: float = 3.0,
    n_samples: int = 1440,
    sigmoid_beta: float = 10.0,
) -> dict:
    """Smooth bite metric replacing discrete contact-count detection.

    Rather than counting hard zero-crossings between the Archimedean spiral
    and the weapon profile (which is discontinuous w.r.t. small geometry
    changes), this function computes a sigmoid-based "engagement" integral
    over one revolution.

    Algorithm
    ---------
    1. Sample the outer profile r(theta) at ``n_samples`` uniform angles.
    2. Construct the Archimedean spiral r_enemy(theta) = r_start - v_per_rad * theta
       over one full revolution (theta in [0, 2*pi]).
    3. Compute the sigmoid engagement function:
           C(theta_j) = 1 / (1 + exp(-sigmoid_beta * (r_profile - r_enemy)))
       C is near 1 when the weapon profile is outside the enemy (contact)
       and near 0 when the weapon is inside the enemy (no contact).
    4. engagement = mean(C)   (normalized integral approximation in [0, 1])
    5. Peaks in C (height > 0.5, prominence > 0.1) estimate n_contacts.
    6. bite_mm = v_per_rad * 2*pi / n_contacts.

    Parameters
    ----------
    poly            : weapon polygon (Shapely Polygon or MultiPolygon)
    rpm             : weapon rotational speed (RPM)
    drive_speed_mps : enemy approach speed in m/s (default 3 m/s)
    n_samples       : number of angular samples for profile and spiral
    sigmoid_beta    : steepness of sigmoid transition; higher = more like
                      a hard step (discrete bite), lower = smoother gradient

    Returns
    -------
    dict with keys:
        bite_mm        : effective bite depth (mm)
        n_contacts     : estimated number of contacts per revolution
        engagement     : mean sigmoid engagement in [0, 1]
        v_per_rad_mm   : mm of advance per radian of weapon rotation
        max_bite_mm    : theoretical max bite (single contact)
    """
    from scipy.interpolate import interp1d
    from scipy.signal import find_peaks
    from shapely.geometry import MultiPolygon

    # Handle MultiPolygon: use largest component
    if hasattr(poly, "geoms"):
        poly = max(poly.geoms, key=lambda p: p.area)

    omega = 2.0 * np.pi * max(float(rpm), 1.0) / 60.0       # rad/s
    v_mm_s = drive_speed_mps * 1000.0                         # mm/s
    v_per_rad = v_mm_s / omega                                 # mm per radian
    max_bite = v_per_rad * 2.0 * np.pi                        # theoretical max

    fallback = {
        "bite_mm":       max_bite,
        "n_contacts":    1,
        "engagement":    0.5,
        "v_per_rad_mm":  v_per_rad,
        "max_bite_mm":   max_bite,
    }

    try:
        cx, cy = poly.centroid.x, poly.centroid.y
        ext = np.array(poly.exterior.coords[:-1])

        if len(ext) < 4:
            return fallback

        angles = np.arctan2(ext[:, 1] - cy, ext[:, 0] - cx)
        radii = np.hypot(ext[:, 0] - cx, ext[:, 1] - cy)

        sort_idx = np.argsort(angles)
        angles_s = angles[sort_idx]
        radii_s = radii[sort_idx]

        # Wrap ±2pi copies for periodic interpolation at boundaries
        angles_w = np.concatenate([
            angles_s - 2.0 * np.pi,
            angles_s,
            angles_s + 2.0 * np.pi,
        ])
        radii_w = np.tile(radii_s, 3)
        angles_u, uid = np.unique(angles_w, return_index=True)
        radii_u = radii_w[uid]

        if len(angles_u) < 4:
            return fallback

        f_r = interp1d(angles_u, radii_u, kind="linear", fill_value="extrapolate")
        # Sample over [0, 2*pi] — same range as the spiral theta
        theta_grid = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
        # Map to [-pi, pi] for interpolation (profile sampled at those angles)
        theta_interp = theta_grid - np.pi   # shift: [0, 2pi) -> [-pi, pi)
        r_profile = f_r(theta_interp)

        # Archimedean spiral
        r_start = float(r_profile.max()) * 1.05
        r_enemy = r_start - v_per_rad * theta_grid

        # Sigmoid engagement function
        delta = r_profile - r_enemy          # positive where weapon protrudes past enemy
        C = 1.0 / (1.0 + np.exp(-sigmoid_beta * delta))
        engagement = float(np.mean(C))

        # Count peaks in C to estimate discrete contact count
        peaks, _ = find_peaks(C, height=0.5, prominence=0.1)
        n_contacts = max(1, int(len(peaks)))

        bite_mm = v_per_rad * 2.0 * np.pi / n_contacts

        return {
            "bite_mm":      float(bite_mm),
            "n_contacts":   n_contacts,
            "engagement":   engagement,
            "v_per_rad_mm": float(v_per_rad),
            "max_bite_mm":  float(max_bite),
        }

    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Robust online normalizer
# ---------------------------------------------------------------------------

class RobustNormalizer:
    """Online rolling normalizer using median and IQR.

    Maintains a rolling window of observations for each metric key and
    normalizes incoming values via (value - median) / (IQR + eps).

    This is robust to outliers (unlike mean/std normalization) and adapts
    as the objective landscape changes during optimization.

    Parameters
    ----------
    window : int
        Maximum number of observations to retain per key (ring buffer).
    eps : float
        Small constant added to IQR to prevent division by zero.

    Usage
    -----
    normalizer = RobustNormalizer(window=200)
    normalizer.update(metrics_dict)          # add new observation
    normed = normalizer.normalize(metrics_dict)  # returns normalized copy
    """

    def __init__(self, window: int = 200, eps: float = 1e-6) -> None:
        self._window = window
        self._eps = eps
        # Per-key deque of float observations
        self._history: dict[str, collections.deque] = {}

    def update(self, metrics: dict) -> None:
        """Add numeric values from ``metrics`` to the rolling history.

        Non-numeric values and keys starting with '_' are silently skipped.
        """
        for key, value in metrics.items():
            if key.startswith("_"):
                continue
            if not isinstance(value, (int, float)):
                continue
            if not np.isfinite(float(value)):
                continue
            if key not in self._history:
                self._history[key] = collections.deque(maxlen=self._window)
            self._history[key].append(float(value))

    def normalize(self, metrics: dict) -> dict:
        """Return a new dict with numeric values normalized via median/IQR.

        Keys with fewer than 5 observations are returned unchanged.
        Non-numeric and internal (underscore-prefixed) keys are passed
        through unchanged.
        """
        result = {}
        for key, value in metrics.items():
            if (
                key.startswith("_")
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or key not in self._history
                or len(self._history[key]) < 5
            ):
                result[key] = value
                continue

            hist = np.array(self._history[key])
            median = float(np.median(hist))
            q75 = float(np.percentile(hist, 75))
            q25 = float(np.percentile(hist, 25))
            iqr = q75 - q25
            result[key] = (float(value) - median) / (iqr + self._eps)

        return result
