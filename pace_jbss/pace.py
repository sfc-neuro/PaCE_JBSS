"""Parallel Component Extraction: PaCE.

Finds common components across two datasets based on mutual information.
Based on the python implementation of the fast ICA algorithm in scikit-learn.

Reference: Cotroneo et al. (2026)

Authors: Silvia Cotroneo <silvia.cotroneo@aalto.fi>
         Marijn van Vliet <marijn.vanvliet@aalto.fi>

SPDX-License-Identifier: BSD-3-Clause
"""

import warnings
from numbers import Integral, Real
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils import check_array
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data


# --------------------------------------------------
# Stable log(cosh)
# --------------------------------------------------
def _log_cosh(u):
    #return jnp.logaddexp(u, -u)
    return jnp.logaddexp(u, -u) - jnp.log(2.0)


# --------------------------------------------------
# Other functions (G(u)) to implement
# --------------------------------------------------
def _exp(x):
    return -jnp.exp(-(x**2) / 2)


def _cube(x):
    return x**3

def _quadratic(x):
    return x**4


# --------------------------------------------------
# Orthogonal projection / normalization
# --------------------------------------------------
def _project_deflation_np(w, Wprev, eps=1e-12):
    """Project w onto the orthogonal complement of rows in Wprev.

    Wprev shape: (n_prev, dim)
    """
    w = np.asarray(w, dtype=float)
    if Wprev is not None and len(Wprev) > 0:
        Wprev = np.asarray(Wprev, dtype=float)
        w = w - Wprev.T @ (Wprev @ w)
    nrm = np.linalg.norm(w)
    if nrm < eps:
        raise FloatingPointError("Vector collapsed during deflation projection.")
    return w / nrm


def _project_deflation_jax(w, Wprev, eps=1e-12):
    if Wprev is not None and Wprev.size > 0:
        w = w - Wprev.T @ (Wprev @ w)
    return w / (jnp.linalg.norm(w) + eps)


# --------------------------------------------------
# Build one-pair objective with deflation constraints
# --------------------------------------------------
def _build_objective_prewhitened_deflation(
    X0,
    Y0,
    lam,
    scalarization,
    G,
    c_G,
    Wx_prev=None,
    Wy_prev=None,
    n_ref=200000,
    seed=0,
    cov_eps=1e-8,
):
    """X0, Y0 assumed already whitened and centered.

    Shapes:
        X0: (n_samples, dx)
        Y0: (n_samples, dy)

    Wx_prev: previously extracted weight vectors for X-view
             shape (n_prev, dx)
    Wy_prev: previously extracted weight vectors for Y-view
             shape (n_prev, dy)
    """
    key = jax.random.PRNGKey(seed)
    g_ref = jax.random.normal(key, shape=(n_ref,))
    c1 = jnp.mean(G(g_ref))  # E[G(nu_1)], nu_1 ~ N(0,1)

    X0 = jnp.asarray(X0)
    Y0 = jnp.asarray(Y0)

    dx = X0.shape[0]
    dy = Y0.shape[0]

    if Wx_prev is None:
        Wx_prev = jnp.zeros((0, dx))
    else:
        Wx_prev = jnp.asarray(Wx_prev)

    if Wy_prev is None:
        Wy_prev = jnp.zeros((0, dy))
    else:
        Wy_prev = jnp.asarray(Wy_prev)

    def unpack(theta):
        wx = theta[:dx]
        wy = theta[dx : dx + dy]

        # Deflation projection + normalization
        wx = _project_deflation_jax(wx, Wx_prev)
        wy = _project_deflation_jax(wy, Wy_prev)
        return wx, wy

    def eval_all(theta, scalarization):

        wx, wy = unpack(theta)
        x = wx @ X0
        y = wy @ Y0

        z = scalarization(x, y)

        mz = jnp.mean(z)
        var_z = jnp.mean((z - mz) ** 2)
        std_z = jnp.sqrt(var_z + 1e-12)

        cz = jnp.mean(G(std_z * g_ref))

        mx = jnp.mean(G(x)) - c1
        my = jnp.mean(G(y)) - c1
        mz_term = jnp.mean(G(z - mz)) - cz

        C1 = 0.5 * jnp.log(2 * jnp.pi * jnp.exp(1))
        C2 = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * (var_z + 1e-12))
        F = (2 * lam - 1) * (C1 - mx**2 - my**2) + (lam - 1) * (-C2 / c_G + mz_term**2)

        return {
            "F": F,
            "wx": wx,
            "wy": wy,
            "mx": mx,
            "my": my,
            "mz_term": mz_term,
            "var_z": var_z,
            "std_z": std_z,
        }

    def F(theta):
        return eval_all(theta, scalarization)["F"]

    # THIS IS WHY WE NEED JAX
    grad_F = jax.grad(F)

    def f(theta):
        return np.array(F(theta), dtype=float)

    def g(theta):
        return np.array(grad_F(theta), dtype=float)

    def eval_numpy(theta):
        out = eval_all(theta, scalarization)
        return {k: np.array(v) for k, v in out.items()}

    return f, g, eval_numpy, dx, dy


# --------------------------------------------------
# Optimize one pair
# --------------------------------------------------
def _optimize_one_pair_deflation(
    X0,
    Y0,
    scalarization,
    Wx_prev=None,
    Wy_prev=None,
    lam=0.5,
    G=_log_cosh,
    c_G=79.0156,
    seed=0,
    maxiter=500,
):
    """Bla."""
    history = {
        "iter": [],
        "wx": [],
        "wy": [],
        "grad_norm": [],
    }

    f, g, eval_numpy, dx, dy = _build_objective_prewhitened_deflation(
        X0=X0,
        Y0=Y0,
        scalarization=scalarization,
        Wx_prev=Wx_prev,
        Wy_prev=Wy_prev,
        lam=lam,
        G=G,
        c_G=c_G,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    theta0 = rng.standard_normal(dx + dy)

    # make initial point feasible-ish
    wx0 = _project_deflation_np(theta0[:dx], Wx_prev if Wx_prev is not None else [])
    wy0 = _project_deflation_np(
        theta0[dx : dx + dy], Wy_prev if Wy_prev is not None else []
    )
    theta0 = np.concatenate([wx0, wy0])

    iteration = {"k": 0}

    def callback(theta):
        vals = eval_numpy(theta)
        grad = g(theta)

        history["iter"].append(iteration["k"])
        history["wx"].append(vals["wx"].copy())
        history["wy"].append(vals["wy"].copy())
        history["grad_norm"].append(float(np.linalg.norm(grad)))

        iteration["k"] += 1

    # store initial point too
    callback(theta0)

    res = minimize(
        f,
        theta0,
        jac=g,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": maxiter},
    )

    theta = res.x
    wx = _project_deflation_np(theta[:dx], Wx_prev if Wx_prev is not None else [])
    wy = _project_deflation_np(
        theta[dx : dx + dy], Wy_prev if Wy_prev is not None else []
    )

    return wx, wy, res, history


# --------------------------------------------------
# Full deflation extraction
# --------------------------------------------------
def _optimize_multiple_pairs_deflation(
    X0,
    Y0,
    scalarization,
    lam,
    G=_log_cosh,
    c_G=79.0156,
    seed=0,
    maxiter=500,
):
    """Extract multiple pairs one by one with deflation.

    X0, Y0 already whitened and centered.
    Shapes:
        X0: (n_samples, dx)
        Y0: (n_samples, dy)
    """
    X0 = np.asarray(X0, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)

    dx = X0.shape[0]
    dy = Y0.shape[0]

    n_components = min(dx, dy)

    Wx = []
    Wy = []
    results = []
    n_iter = []

    histories = []
    for i in range(n_components):
        wx, wy, res, hist = _optimize_one_pair_deflation(
            X0=X0,
            Y0=Y0,
            scalarization=scalarization,
            Wx_prev=np.array(Wx) if len(Wx) > 0 else None,
            Wy_prev=np.array(Wy) if len(Wy) > 0 else None,
            lam=lam,
            G=G,
            c_G=c_G,
            seed=seed + i,
            maxiter=maxiter,
        )

        histories.append(hist)
        Wx.append(wx)
        Wy.append(wy)
        results.append(res)
        n_iter.append(res["nit"])

    Wx = np.vstack(Wx)  # shape (n_components, dx)
    Wy = np.vstack(Wy)  # shape (n_components, dy)

    return (
        Wx,
        Wy,
        max(n_iter),
        results,
        histories,
    )


class PaCE(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """PaCE: Parallelized Component Extraction.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to use. If ``None`` is passed, all are used.

    fun : {'logcosh', 'exp'}, default='logcosh'
        The functional form of the G function used in the approximation to neg-entropy.
        Could be either ``'logcosh'`` or ``'exp'``.

    max_iter : int, default=200
        Maximum number of iterations during fitting of the model.

    tol : float, default=1e-4
        A positive scalar giving the tolerance at which the un-mixing matrices are
        considered to have converged.

    scalarization : callable | None
        Optimizes the un-mixing matrices such that they maximize either the entropy of
        the sum or that of the difference of the sources.

    lam : float, default=0.5
        Tradeoff between component dependence between datasets and component
        independence within the dataset.

    seed : int, default=None
        Used to initialize the weights with a normal distribution. Pass an ``int`` for
        reproducible results.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components1_ : ndarray of shape (n_components, n_features)
        The linear operator to apply to ``X1`` to get the independent sources.
        This is equal to ``self.unmixing1_ @ self.whitening1_``.

    components2_ : ndarray of shape (n_components, n_features)
        The linear operator to apply to ``X2`` to get the independent sources.
        This is equal to ``self.unmixing2_ @ self.whitening2_``.

    mixing1_ : ndarray of shape (n_features, n_components)
        The pseudo-inverse of ``components1_``. It is the linear operator
        that maps independent sources to  ``self.whitening1_ @ X1``.

    mixing2_ : ndarray of shape (n_features, n_components)
        The pseudo-inverse of ``components2_``. It is the linear operator
        that maps independent sources to  ``self.whitening2_ @ X2``.

    unmixing1_ : ndarray of shape (n_features, n_components)
        The unmixing matrix for ``components1_``. This is the linear operator that maps
        ``self.whitening1_ @ X1`` to the independent sources.

    unmixing2_ : ndarray of shape (n_features, n_components)
        The unmixing matrix for ``components2_``. This is the linear operator that maps
        ``self.whitening2_ @ X2`` to the independent sources.

    mean1_ : ndarray of shape(n_features,)
        The mean over features in ``X1``.

    mean2_ : ndarray of shape(n_features,)
        The mean over features in ``X2``.

    n_iter_ : int
        Number of iterations it took to reach convergence.

    whitening1_ : ndarray of shape (n_components, n_features)
        This is the pre-whitening matrix that projects ``X1`` onto the first
        ``n_components`` principal components.

    whitening2_ : ndarray of shape (n_components, n_features)
        This is the pre-whitening matrix that projects ``X2`` onto the first
        ``n_components`` principal components.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "fun": [StrOptions({"logcosh", "exp"})],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "scalarization": [callable, None],
        "seed": [int, None],
    }

    def __init__(
        self,
        n_components=None,  # TODO: does this actually work?
        *,
        lam=0.5,
        max_iter=200,
        tol=1e-14,
        scalarization=None,
        G="logcosh",
        c_G=None,
        seed=None,
    ):
        super().__init__()

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.lam = lam

        if scalarization is None:
            # TODO: this should change when we scale to more than 2 datasets
            # it should be in the form of f(*X)
            def scalarization(*X):
                return X[0] + X[1]

        if not callable(scalarization):
            raise TypeError("scalarization must be a callable of the form f(*X) -> Y")

        self.n_components = n_components
        self.fun = fun
        self._c_G = 79.016 if fun == "logcosh" else 33.669
        self.max_iter = max_iter
        self.tol = tol
        self.lam = lam
        self.scalarization = scalarization

        if G == "logcosh":
            self.G = _log_cosh
            self.c_G = c_G if c_G is not None else 79.016
        elif G == "exp":
            self.G = _exp
            self.c_G = c_G if c_G is not None else 33.669
        elif callable(G):
            if c_G is None:
                raise ValueError("c_G must be supplied when G is a custom callable.")
            self.c_G = c_G

        self.seed = int(time()) if seed is None else seed

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X1, X2):
        """Fit the model.

        Parameters
        ----------
        X1 : array-like of shape (n_samples, n_features)
            The first dataset, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        X2 : array-like of shape (n_samples, n_features)
            The second dataset, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        """
        X1T = validate_data(
            self,
            X1,
            copy=True,
            dtype=[np.float64, np.float32],
            ensure_min_samples=2,
        ).T
        X2T = validate_data(
            self,
            X2,
            copy=True,
            dtype=[np.float64, np.float32],
            ensure_min_samples=2,
        ).T

        n_features, n_samples = X1T.shape
        n_components = self.n_components
        if n_components is None:
            n_components = min(n_samples, n_features)
        if n_components > min(n_samples, n_features):
            n_components = min(n_samples, n_features)
            warnings.warn(
                "n_components is too large: it will be set to %s" % n_components
            )

        # Centering the features of X
        X_mean1 = X1T.mean(axis=-1)
        X1T -= X_mean1[:, np.newaxis]
        X1 -= X_mean1[np.newaxis, :]

        X_mean2 = X2T.mean(axis=-1)
        X2T -= X_mean2[:, np.newaxis]
        X2 -= X_mean2[np.newaxis, :]

        u1, d1 = linalg.svd(X1T, full_matrices=False, check_finite=False)[:2]
        u2, d2 = linalg.svd(X2T, full_matrices=False, check_finite=False)[:2]

        # Give consistent eigenvectors for both svd solvers
        u1 *= np.sign(u1[0])
        u2 *= np.sign(u2[0])

        K1 = (u1 / d1).T[:n_components]  # see (6.33) p.140
        K2 = (u2 / d2).T[:n_components]  # see (6.33) p.140

        del u1, d1, u2, d2
        X11 = K1 @ X1T
        X12 = K2 @ X2T

        # see (13.6) p.267 Here X1 is white and data
        # in X has been projected onto a subspace by PCA
        X11 *= np.sqrt(n_samples)
        X12 *= np.sqrt(n_samples)

        W1, W2, n_iter, res, histories = _optimize_multiple_pairs_deflation(
            X0=X11,
            Y0=X12,
            scalarization=self.scalarization,
            lam=self.lam,
            seed=self.seed,
            maxiter=self.max_iter,
        )
        del X11, X12

        self.n_iter_ = n_iter

        S1 = np.linalg.multi_dot([W1, K1, X1T]).T
        S2 = np.linalg.multi_dot([W2, K2, X2T]).T

        intermediate_sources1 = []
        intermediate_sources2 = []

        for history in histories:
            intermediate_source1 = []
            intermediate_source2 = []
            for wkn1, wkn2 in zip(history["wx"], history["wy"]):
                intermediate_source1.append(np.linalg.multi_dot([wkn1, K1, X1T]).T)
                intermediate_source2.append(np.linalg.multi_dot([wkn2, K2, X2T]).T)

            intermediate_sources1.append(np.array(intermediate_source1))
            intermediate_sources2.append(np.array(intermediate_source2))

        w1s = intermediate_sources1
        w2s = intermediate_sources2

        S_std1 = np.std(S1, axis=0, keepdims=True)
        S_std2 = np.std(S2, axis=0, keepdims=True)

        S1 /= S_std1
        W1 /= S_std1.T

        S2 /= S_std2
        W2 /= S_std2.T

        self.components1_ = W1 @ K1
        self.mean1_ = X_mean1
        self.whitening1_ = K1
        self.components2_ = W2 @ K2
        self.mean2_ = X_mean2
        self.whitening2_ = K2

        self.mixing1_ = linalg.pinv(self.components1_, check_finite=False)
        self.unmixing1_ = W1
        self.mixing2_ = linalg.pinv(self.components2_, check_finite=False)
        self.unmixing2_ = W2

        self._steps_w1 = w1s
        self._steps_w2 = w2s

        return self

    def transform(self, X1, X2):
        """Recover the sources from X (apply the unmixing matrix).

        Parameters
        ----------
        X1 : array-like of shape (n_samples, n_features)
            The first dataset, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        X2 : array-like of shape (n_samples, n_features)
            The second dataset, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        S1 : ndarray of shape (n_samples, n_components)
            Independenct sources for ``X1``.

        S2 : ndarray of shape (n_samples, n_components)
            Independenct sources for ``X2``.
        """
        check_is_fitted(self)

        X1 = validate_data(
            self,
            X1,
            copy=True,
            dtype=[np.float64, np.float32],
            reset=False,
        )
        X2 = validate_data(
            self,
            X2,
            copy=True,
            dtype=[np.float64, np.float32],
            reset=False,
        )

        X1 -= self.mean1_
        X2 -= self.mean2_

        return X1 @ self.components1_.T, X2 @ self.components2_.T

    def fit_transform(self, X1, X2):
        """Fit the model and recover the sources from X (apply the unmixing matrix).

        Parameters
        ----------
        X1 : array-like of shape (n_samples, n_features)
            The first dataset, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        X2 : array-like of shape (n_samples, n_features)
            The second dataset, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        S1 : ndarray of shape (n_samples, n_components)
            Independenct sources for ``X1``.

        S2 : ndarray of shape (n_samples, n_components)
            Independenct sources for ``X2``.
        """
        self.fit(X1, X2)
        return self.transform(X1, X2)

    def inverse_transform(self, S1, S2):
        """Transform the sources back to the mixed data (apply mixing matrix).

        Parameters
        ----------
        S1 : ndarray of shape (n_samples, n_components)
            Independenct sources for ``X1``.

        S2 : ndarray of shape (n_samples, n_components)
            Independenct sources for ``X2``.

        Returns
        -------
        X1 : ndarray of shape (n_samples, n_features)
            Reconstructed data for ``S1`` obtained with the mixing matrix.

        X2 : ndarray of shape (n_samples, n_features)
            Reconstructed data for ``S2`` obtained with the mixing matrix.
        """
        check_is_fitted(self)

        X1 = check_array(S1, dtype=[np.float64, np.float32])
        X2 = check_array(S2, dtype=[np.float64, np.float32])

        X1 = S1 @ self.mixing1_.T
        X2 = S2 @ self.mixing2_.T

        X1 += self.mean1_
        X2 += self.mean2_

        return X1, X2

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.components1_.shape[0]

    def __sklearn_tags__(self):
        """Tags for scikit-learn API."""
        tags = super().__sklearn_tags__()
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags
