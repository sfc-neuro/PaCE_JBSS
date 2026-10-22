"""Tests for the PACE module."""

import pace_jbss
import jax
import numpy as np
from numpy.testing import assert_allclose

jax.config.update("jax_platform_name", "cpu")


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
    X1 = np.dot(S1, A1.T)
    X2 = np.dot(S2, A2.T)

    pace = pace_jbss.PaCE(
        n_components=3,
        lam=0,
        scalarization=lambda X, Y: Y - X**2,
    )
    S1_c, S2_c = pace.fit_transform(X1, X2)  # Reconstruct signals

    # Compare estimated with true sources. Sources appear in a random order though.
    S1_cross_corr = np.abs(np.corrcoef(S1_c.T, S1.T)[:3, 3:])
    S1_ordering = S1_cross_corr.argmax(axis=0)  # find best matching source
    assert len(np.setdiff1d(S1_ordering, [0, 1, 2])) == 0
    S1_cross_corr = S1_cross_corr[:, S1_ordering]
    assert_allclose(np.diag(S1_cross_corr), 1, atol=0.02)  # correlation should be >0.98

    S2_cross_corr = np.abs(np.corrcoef(S2_c.T, S2.T)[:3, 3:])
    S2_ordering = S2_cross_corr.argmax(axis=0)  # find best matching source.
    assert len(np.setdiff1d(S2_ordering, [0, 1, 2])) == 0
    S2_cross_corr = S2_cross_corr[:, S2_ordering]
    assert_allclose(np.diag(S2_cross_corr), 1, atol=0.01)  # Correlation should be >0.99
