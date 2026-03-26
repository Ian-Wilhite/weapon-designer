"""Tests for SFSurrogate GP surrogate (surrogate_fea.py)."""

from __future__ import annotations

import numpy as np
import pytest

from weapon_designer.surrogate_fea import SFSurrogate

# Skip all tests requiring sklearn if it is not installed
try:
    import sklearn  # noqa: F401
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

requires_sklearn = pytest.mark.skipif(
    not _HAS_SKLEARN,
    reason="scikit-learn is not installed",
)


@requires_sklearn
def test_sf_surrogate_fit_predict():
    """Fit GP on synthetic data; predict returns (float, float >= 0)."""
    rng = np.random.default_rng(42)
    N, d = 20, 4
    params = rng.random((N, d))
    sf_values = 1.5 + rng.random(N) * 2.0  # SF in [1.5, 3.5]

    surr = SFSurrogate()
    surr.fit(params, sf_values)

    x = rng.random(d)
    mu, sigma = surr.predict_sf(x)
    assert isinstance(mu, float)
    assert isinstance(sigma, float)
    assert sigma >= 0.0


@requires_sklearn
def test_sf_surrogate_p_violated():
    """P(SF violated) lies in [0, 1] for any input."""
    rng = np.random.default_rng(0)
    N, d = 15, 3
    params = rng.random((N, d))
    # Uniformly high SF values → surrogate should predict low p_violated
    sf_values = np.full(N, 3.0)

    surr = SFSurrogate()
    surr.fit(params, sf_values)

    x = rng.random(d)
    p = surr.p_violated(x, sf_min=1.5)
    assert 0.0 <= p <= 1.0


@requires_sklearn
def test_sf_surrogate_p_violated_high_for_low_sf():
    """When all training SF values are well below sf_min, p_violated should be high."""
    rng = np.random.default_rng(7)
    N, d = 15, 3
    params = rng.random((N, d))
    # Very low SF values
    sf_values = np.full(N, 0.5)

    surr = SFSurrogate()
    surr.fit(params, sf_values)

    x = rng.random(d)
    p = surr.p_violated(x, sf_min=1.5)
    assert p > 0.5


@requires_sklearn
def test_sf_surrogate_save_load(tmp_path):
    """Saved and reloaded surrogate produces identical predictions."""
    rng = np.random.default_rng(1)
    params = rng.random((10, 3))
    sf_values = rng.random(10) * 3.0 + 0.5

    surr = SFSurrogate()
    surr.fit(params, sf_values)

    path = tmp_path / "test_surrogate.pkl"
    surr.save(path)
    surr2 = SFSurrogate.load(path)

    x = rng.random(3)
    mu1, sigma1 = surr.predict_sf(x)
    mu2, sigma2 = surr2.predict_sf(x)
    assert abs(mu1 - mu2) < 1e-6
    assert abs(sigma1 - sigma2) < 1e-6


def test_fea_surrogate_alias():
    """FEASurrogate is a backward-compat alias for SFSurrogate."""
    from weapon_designer.surrogate_fea import FEASurrogate
    assert FEASurrogate is SFSurrogate


def test_sf_surrogate_unfitted_raises():
    """Predicting on unfitted surrogate raises RuntimeError."""
    surr = SFSurrogate()
    with pytest.raises(RuntimeError, match="must be fitted"):
        surr.predict_sf(np.array([0.1, 0.2, 0.3]))


@requires_sklearn
def test_sf_surrogate_acquisition_score():
    """acquisition_score returns non-negative float <= analytical_score."""
    rng = np.random.default_rng(3)
    N, d = 12, 4
    params = rng.random((N, d))
    sf_values = 2.0 + rng.random(N)

    surr = SFSurrogate()
    surr.fit(params, sf_values)

    x = rng.random(d)
    analytical = 150.0  # Joules
    acq = surr.acquisition_score(x, analytical_score=analytical, sf_min=1.5)
    assert isinstance(acq, float)
    assert 0.0 <= acq <= analytical + 1e-6


def test_sf_surrogate_summary_unfitted():
    """summary() returns 'unfitted' string before fitting."""
    surr = SFSurrogate()
    assert "unfitted" in surr.summary()


@requires_sklearn
def test_sf_surrogate_summary_fitted():
    """summary() includes n_train after fitting."""
    surr = SFSurrogate()
    rng = np.random.default_rng(5)
    surr.fit(rng.random((8, 3)), rng.random(8) * 3.0)
    s = surr.summary()
    assert "8" in s
