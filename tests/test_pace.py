"""Tests for the PACE module."""

import jax
import numpy as np
import pace_jbss
from numpy.testing import assert_allclose
from scipy.stats import zscore

jax.config.update("jax_platform_name", "cpu")


def _evaluate_reconstruction(estimated, true, atol=1e-5):
    # Compare estimated with true sources. Sources appear in a random order though.
    assert len(estimated) == len(true)
    n_sources = len(estimated)
    cross_corr = np.abs(np.corrcoef(estimated, true)[:n_sources, n_sources:])
    ordering = cross_corr.argmax(axis=0)  # find best matching source
    assert len(np.setdiff1d(ordering, np.arange(n_sources))) == 0
    cross_corr = np.diag(cross_corr[ordering])
    assert_allclose(cross_corr, 1, atol=atol)  # correlation should be >0.98


def test_basic_demixing():
    """A basic demixing test on generated data."""
    rng = np.random.default_rng(1)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)

    s11 = np.sin(2 * time) * np.sin(40 * time)
    s12 = np.sin(3 * time) ** 5
    s13 = rng.laplace(size=s11.shape)

    s21 = s11**2
    s22 = s12**2
    s23 = s13**2

    S1 = np.c_[s11, s12, s13]
    S2 = np.c_[s21, s22, s23]

    # Standardize data.
    for S in [S1, S2]:
        S /= S.std(axis=0)

    # Mixing matrices.
    A1 = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
    A2 = np.array([[0.5, 2, 1.0], [1, 2, 1], [1.0, 1.0, 2.0]])

    # Generate observations.
    X1 = S1 @ A1.T
    X2 = S2 @ A2.T

    pace = pace_jbss.PaCE(
        n_components=3,
        lam=0,
        scalarization=lambda X, Y: Y - X**2,
        seed=0,
    )
    S1_c, S2_c = pace.fit_transform(X1, X2)  # Reconstruct signals
    _evaluate_reconstruction(S1_c.T, S1.T, atol=0.02)
    _evaluate_reconstruction(S2_c.T, S2.T, atol=0.01)


def test_oscillatory():
    """Test de-mixing oscillatory signals."""
    freqs = np.array([32, 13, 41.0])
    n_sources = len(freqs)
    times = np.linspace(0.0, 1.0, 10_000, endpoint=False)
    sources = np.sin(2.0 * np.pi * freqs[:, None] * times[None, :])

    rng = np.random.default_rng(1)
    mixing1 = rng.random((n_sources, n_sources))
    mixing2 = rng.random((n_sources, n_sources))
    # Avoid pathological nearly singular mixing matrices by resampling.
    assert np.linalg.cond(mixing1) < 1e6
    assert np.linalg.cond(mixing2) < 1e6

    X1 = (mixing1 @ sources).T
    X2 = (mixing2 @ sources).T

    pace = pace_jbss.PaCE(seed=0, lam=1.0)
    S1_c, S2_c = pace.fit_transform(X1, X2)  # Reconstruct signals
    _evaluate_reconstruction(S1_c.T, sources)
    _evaluate_reconstruction(S2_c.T, sources)
