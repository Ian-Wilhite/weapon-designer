"""GP surrogate for structural safety factor prediction.

Replaces the POD+GP stress-field surrogate with a single scalar GP
that learns to predict the FEA safety factor directly.

This fixes the saturation problem: the old GP predicted a composite
E_score that was ~1.0 for all valid shapes, giving near-zero variance
and 100% FEA call rate. The safety factor ranges from ~0.8 (failure)
to ~4.0+ (overbuilt), giving genuine learning signal.

Usage
-----
    from weapon_designer.surrogate_fea import SFSurrogate

    surr = SFSurrogate()
    surr.fit(params_matrix, sf_values)   # (N×d), (N,)
    mu, sigma = surr.predict_sf(x)       # predicted SF mean + std
    p = surr.p_violated(x, sf_min=1.5)  # P(SF < sf_min)

    # Persistence
    surr.save("sf_surrogate.pkl")
    surr2 = SFSurrogate.load("sf_surrogate.pkl")
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.special import ndtr   # standard normal CDF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_params(params: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Normalise params to [0, 1] range using per-dimension min/max."""
    rng = hi - lo
    rng[rng < 1e-12] = 1.0
    return (params - lo) / rng


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SFSurrogate:
    """Single GP surrogate predicting FEA safety factor.

    Replaces FEASurrogate (POD + k GPs on stress modes) with a single GP
    on the SF scalar, enabling:

    - P(SF violated) = Φ((sf_min − μ_SF) / σ_SF) — Bayesian feasibility
    - UCB acquisition: E_transfer_analytical × (1 − P(SF_violated))
    - FEA trigger: run when σ_SF > threshold OR p_violated < 0.05

    Parameters
    ----------
    kernel_params : dict of sklearn GP kernel keyword args.
                    Recognised keys: 'length_scale', 'noise_level'.
    """

    def __init__(self, kernel_params: dict | None = None):
        self.kernel_params = kernel_params or {}
        self.gp_ = None
        self.param_lo_: np.ndarray | None = None
        self.param_hi_: np.ndarray | None = None
        self.n_train_: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, params: np.ndarray, sf_values: np.ndarray) -> "SFSurrogate":
        """Fit GP on safety factor scalar values.

        Parameters
        ----------
        params    : (N, d) design parameter vectors
        sf_values : (N,)  FEA safety factor values
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        params = np.atleast_2d(params)
        sf_values = np.asarray(sf_values, dtype=float).ravel()
        assert params.shape[0] == len(sf_values), (
            f"params rows ({params.shape[0]}) must match sf_values length ({len(sf_values)})"
        )

        self.param_lo_ = params.min(axis=0)
        self.param_hi_ = params.max(axis=0)
        X_norm = _normalise_params(params, self.param_lo_.copy(), self.param_hi_.copy())

        length_scale = self.kernel_params.get("length_scale", 1.0)
        noise_level  = self.kernel_params.get("noise_level", 1e-3)
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)

        self.gp_ = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
        )
        self.gp_.fit(X_norm, sf_values)
        self.n_train_ = len(sf_values)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_sf(self, x: np.ndarray) -> tuple[float, float]:
        """Predict safety factor mean and uncertainty.

        Parameters
        ----------
        x : (d,) or (1, d) design parameter vector

        Returns
        -------
        (mu_SF, sigma_SF) — predicted mean and standard deviation
        """
        self._check_fitted()
        x = np.atleast_2d(x)
        X_norm = _normalise_params(x, self.param_lo_.copy(), self.param_hi_.copy())
        mu, std = self.gp_.predict(X_norm, return_std=True)
        return float(mu[0]), float(std[0])

    def p_violated(self, x: np.ndarray, sf_min: float = 1.5) -> float:
        """Probability that safety factor is below sf_min.

        P(SF < sf_min) = Φ((sf_min − μ_SF) / σ_SF)

        Uses Gaussian CDF approximation (ndtr = scipy normal CDF).

        Returns
        -------
        float in [0, 1]. High value = likely to fail structurally.
        """
        mu, sigma = self.predict_sf(x)
        if sigma < 1e-10:
            return 0.0 if mu >= sf_min else 1.0
        z = (sf_min - mu) / sigma
        return float(ndtr(z))

    def acquisition_score(
        self,
        x: np.ndarray,
        analytical_score: float,
        sf_min: float = 1.5,
    ) -> float:
        """UCB-style acquisition: E_transfer_analytical × (1 − P(SF_violated)).

        High acquisition = high predicted energy transfer AND low probability
        of structural failure. Used by optimizer_surrogate to decide whether
        to run FEA or trust the surrogate.

        Parameters
        ----------
        x                : design parameter vector
        analytical_score : E_transfer score from score_analytical() (Joules)
        sf_min           : minimum acceptable safety factor
        """
        p_fail = self.p_violated(x, sf_min=sf_min)
        return float(analytical_score * (1.0 - p_fail))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Serialize surrogate to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SFSurrogate] saved to {path}")

    @classmethod
    def load(cls, path) -> "SFSurrogate":
        """Load a serialized surrogate from a pickle file."""
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        return obj

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary string."""
        if self.gp_ is None:
            return "SFSurrogate (unfitted)"
        return f"SFSurrogate(n_train={self.n_train_})"

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _is_fitted(self) -> bool:
        return self.gp_ is not None

    def _check_fitted(self):
        if not self._is_fitted():
            raise RuntimeError("SFSurrogate must be fitted before predicting.")


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: Alias so existing code that does ``from .surrogate_fea import FEASurrogate``
#: continues to work unchanged.
FEASurrogate = SFSurrogate
