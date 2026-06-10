#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fitting functions for use in The Cannon.

This module has been rewritten to use JAX. The numerical core (chi-squared,
the regularized pixel objective, the linear-algebra theta estimate, the noise
scatter fit, the per-pixel training optimization, and the per-spectrum label
inference) all run on ``jax.numpy`` with pure-JAX optimizers from ``jaxopt``.
Analytic gradients and Jacobians that were previously hand-written are now
obtained by automatic differentiation.

The public functions preserve their signatures and return numpy-compatible
results, so the rest of The Cannon (and existing user code) continues to work.
The per-pixel and per-spectrum cores are written as pure functions so they can
be ``jax.vmap``-ed and ``jax.jit``-ed by :mod:`thecannon.model`.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["fit_spectrum", "fit_pixel_fixed_scatter", "fit_theta_by_linalg",
    "chi_sq", "L1Norm_variation"]

import logging
import numpy as np
import jax
import jax.numpy as jnp
from functools import lru_cache
from jax import lax
from time import time

from jaxopt import LBFGS, LBFGSB, ProximalGradient, LevenbergMarquardt
from jaxopt.prox import prox_lasso

logger = logging.getLogger(__name__)


# Default optimizer settings. These are deliberately tight so that the JAX
# optimizers converge to (essentially) the same optima as the previous
# scipy-based implementation.
_TRAIN_MAXITER = 500
_TRAIN_TOL = 1e-8
_TEST_MAXITER = 200
_TEST_TOL = 1e-10


# --------------------------------------------------------------------------- #
#  Core objective pieces                                                       #
# --------------------------------------------------------------------------- #

def chi_sq(theta, design_matrix, flux, ivar, axis=None, gradient=True):
    """
    Calculate the chi-squared difference between the spectral model and flux.

    :param theta:
        The theta coefficients.

    :param design_matrix:
        The model design matrix.

    :param flux:
        The normalized flux values.

    :param ivar:
        The inverse variances of the normalized flux values.

    :param axis: [optional]
        The axis to sum the chi-squared values across.

    :param gradient: [optional]
        Return the chi-squared value and its derivatives (Jacobian).

    :returns:
        The chi-squared difference between the spectral model and flux, and
        optionally, the Jacobian.
    """
    residuals = jnp.dot(theta, design_matrix.T) - flux

    ivar_residuals = ivar * residuals
    f = jnp.sum(ivar_residuals * residuals, axis=axis)
    if not gradient:
        return f

    g = 2.0 * jnp.dot(design_matrix.T, ivar_residuals)
    return (f, g)


def _chi_sq_only(theta, design_matrix, flux, ivar):
    """ The (smooth) chi-squared value only; used as the optimizer objective. """
    residuals = jnp.dot(theta, design_matrix.T) - flux
    return jnp.sum(ivar * residuals * residuals)


def L1Norm_variation(theta):
    """
    Return the L1 norm of theta (except the first entry) and its derivative.

    :param theta:
        An array of finite values.

    :returns:
        A two-length tuple containing: the L1 norm of theta (except the first
        entry), and the derivative of the L1 norm of theta.
    """

    return (jnp.sum(jnp.abs(theta[1:])),
            jnp.hstack([0.0, jnp.sign(theta[1:])]))


def _pixel_objective_function_fixed_scatter(theta, design_matrix, flux, ivar,
    regularization, gradient=True):
    """
    The objective function for a single regularized pixel with fixed scatter.

    :param theta:
        The spectral coefficients.

    :param design_matrix:
        The design matrix for the model.

    :param flux:
        The normalized flux values for a single pixel across many stars.

    :param ivar:
        The adjusted inverse variance of the normalized flux values.

    :param regularization:
        The regularization term to scale the L1 norm of theta with.

    :param gradient: [optional]
        Also return the analytic derivative of the objective function.
    """

    if gradient:
        csq, d_csq = chi_sq(theta, design_matrix, flux, ivar, gradient=True)
        L1, d_L1 = L1Norm_variation(theta)

        f = csq + regularization * L1
        g = d_csq + regularization * d_L1

        return (f, g)

    else:
        csq = chi_sq(theta, design_matrix, flux, ivar, gradient=False)
        L1, d_L1 = L1Norm_variation(theta)

        return csq + regularization * L1


# --------------------------------------------------------------------------- #
#  Linear algebra theta estimate                                              #
# --------------------------------------------------------------------------- #

def fit_theta_by_linalg(flux, ivar, s2, design_matrix):
    """
    Fit theta coefficients to a set of normalized fluxes for a single pixel.

    :param flux:
        The normalized fluxes for a single pixel (across many stars).

    :param ivar:
        The inverse variance of the normalized flux values for a single pixel
        across many stars.

    :param s2:
        The noise residual (squared scatter term) to adopt in the pixel.

    :param design_matrix:
        The model design matrix.

    :returns:
        The label vector coefficients for the pixel, and the inverse variance
        matrix.
    """

    flux = jnp.asarray(flux)
    ivar = jnp.asarray(ivar)
    design_matrix = jnp.asarray(design_matrix)
    N = design_matrix.shape[1]

    adjusted_ivar = ivar / (1. + ivar * s2)
    CiA = design_matrix * adjusted_ivar[:, None]
    ATCiAinv = jnp.linalg.inv(jnp.dot(design_matrix.T, CiA))
    ATY = jnp.dot(design_matrix.T, flux * adjusted_ivar)
    theta = jnp.dot(ATCiAinv, ATY)

    # JAX does not raise on a singular matrix (it returns inf/nan), so detect
    # that case and fall back to the fiducial value, matching the original
    # behaviour which caught `numpy.linalg.LinAlgError`.
    ok = jnp.all(jnp.isfinite(theta)) & jnp.all(jnp.isfinite(ATCiAinv))
    theta_fallback = jnp.concatenate([jnp.ones(1), jnp.zeros(N - 1)])
    cov_fallback = jnp.inf * jnp.eye(N)

    theta = jnp.where(ok, theta, theta_fallback)
    cov = jnp.where(ok, ATCiAinv, cov_fallback)

    return (theta, cov)


# --------------------------------------------------------------------------- #
#  Noise scatter fit                                                          #
# --------------------------------------------------------------------------- #

def _scatter_objective_function(scatter, residuals_squared, ivar):
    """ Legacy-compatible scalar scatter objective (kept for completeness). """
    adjusted_ivar = ivar / (1.0 + ivar * scatter ** 2)
    chi_sq_value = residuals_squared * adjusted_ivar
    return (jnp.median(chi_sq_value) - 1.0) ** 2


def _fit_scatter(residuals_squared, ivar, n_iter=80):
    """
    Solve for the squared scatter ``s2 = scatter**2 >= 0`` such that the median
    of ``residuals_squared * ivar / (1 + ivar * s2)`` equals one.

    The median is a monotonically decreasing function of ``s2``, so we use a
    bracketing bisection. If the median is already <= 1 at ``s2 = 0`` then no
    positive scatter is required and ``s2 = 0`` is returned. This reproduces the
    behaviour of the original Nelder-Mead minimization of ``(median - 1)**2``
    starting from zero, while being jit/vmap-safe.
    """

    def median_of(u):
        adjusted_ivar = ivar / (1.0 + ivar * u)
        return jnp.median(residuals_squared * adjusted_ivar)

    m0 = median_of(0.0)

    # Each term a_i/(1 + b_i u) <= 1 once u >= residuals_squared_i, so the median
    # is guaranteed <= 1 by u = max(residuals_squared); that brackets the root.
    upper = jnp.max(residuals_squared) + 1.0

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        need_larger = median_of(mid) > 1.0  # median decreasing -> go right
        lo = jnp.where(need_larger, mid, lo)
        hi = jnp.where(need_larger, hi, mid)
        return (lo, hi)

    lo, hi = lax.fori_loop(0, n_iter, body, (0.0 * upper, upper))
    root = 0.5 * (lo + hi)

    return jnp.where(m0 > 1.0, root, 0.0)


# --------------------------------------------------------------------------- #
#  Per-pixel training fit                                                      #
# --------------------------------------------------------------------------- #

def make_pixel_fitter(op_method="l_bfgs_b", maxiter=_TRAIN_MAXITER,
    tol=_TRAIN_TOL, bounds=None):
    """
    Build a pure, ``jax.vmap``-able function that fits the theta coefficients
    and noise residual for a single pixel.

    Unbounded fitters are memoized on ``(op_method, maxiter, tol)`` so that
    repeated calls (e.g. one per cross-validation fold or sweep grid point)
    return the *same* function object and JAX's jit cache can reuse the
    compiled program instead of recompiling.

    :param op_method: [optional]
        The optimization method. ``"l_bfgs_b"`` (default) minimizes the
        combined ``chi_sq + lambda * ||theta[1:]||_1`` objective with L-BFGS
        (closest to the original scipy behaviour). ``"proximal"`` solves the
        lasso with proximal gradient (``prox_lasso``), giving exact zeros.

    :param bounds: [optional]
        A two-length tuple ``(lower, upper)`` of arrays of shape ``(T,)`` giving
        box constraints on the theta coefficients. When provided, the bounded
        L-BFGS-B solver is used. Use ``+/- inf`` for unconstrained coefficients.

    :returns:
        A function ``fit(flux, ivar, init_stack, design_matrix, regularization,
        column_mask)`` returning ``(theta, s2, fopt)``.

        - ``init_stack`` is a ``(n_init, T)`` array of initial theta guesses.
        - ``column_mask`` is a ``(T,)`` boolean array; ``False`` marks censored
          coefficients that are held at zero.
    """

    op_method = (op_method or "l_bfgs_b").lower()
    if op_method == "powell":
        logger.warn("op_method='powell' is not supported by the JAX backend; "
                    "using L-BFGS instead.")
        op_method = "l_bfgs_b"
    if op_method not in ("l_bfgs_b", "proximal"):
        raise ValueError("unknown optimization method '{}' -- 'l_bfgs_b' or "
                         "'proximal' are available".format(op_method))

    if bounds is None:
        # Bounds are arrays (unhashable), so only the unbounded fitters are
        # memoized; bounded models (RestrictedCannonModel) rebuild each time.
        return _make_unbounded_pixel_fitter(op_method, maxiter, tol)

    return _build_pixel_fitter(op_method, maxiter, tol, bounds)


@lru_cache(maxsize=None)
def _make_unbounded_pixel_fitter(op_method, maxiter, tol):
    return _build_pixel_fitter(op_method, maxiter, tol, None)


def _build_pixel_fitter(op_method, maxiter, tol, bounds):

    if bounds is not None:
        lower, upper = (jnp.asarray(bounds[0]), jnp.asarray(bounds[1]))
        if op_method == "proximal":
            logger.warn("theta bounds are not supported with op_method="
                        "'proximal'; using bounded L-BFGS-B instead.")

    def fit(flux, ivar, init_stack, design_matrix, regularization, column_mask):

        T = design_matrix.shape[1]
        mask = column_mask.astype(design_matrix.dtype)

        # No information in this pixel -> fiducial theta with infinite scatter.
        no_info = jnp.sum(ivar) < ivar.size

        # Zero out censored columns so they cannot contribute.
        dm = design_matrix * mask[None, :]

        def smooth_objective(theta):
            return _chi_sq_only(theta, dm, flux, ivar) \
                + regularization * jnp.sum(jnp.abs(theta[1:]))

        # Choose the best starting point by objective value.
        feval = jax.vmap(smooth_objective)(init_stack)
        feval = jnp.where(jnp.isnan(feval), jnp.inf, feval)
        best_init = init_stack[jnp.argmin(feval)]

        if bounds is not None:
            # Do not constrain censored coefficients (they are masked to zero).
            lower_eff = jnp.where(column_mask, lower, -jnp.inf)
            upper_eff = jnp.where(column_mask, upper, jnp.inf)
            best_init = jnp.clip(best_init, lower_eff, upper_eff)
            solver = LBFGSB(fun=smooth_objective, maxiter=maxiter, tol=tol)
            theta = solver.run(best_init, bounds=(lower_eff, upper_eff)).params
        elif op_method == "proximal":
            # Per-coordinate L1 weights: do not regularize theta[0] (continuum),
            # and do not regularize censored coefficients (held at zero anyway).
            l1reg = jnp.full((T,), regularization).at[0].set(0.0) * mask
            solver = ProximalGradient(
                fun=lambda th: _chi_sq_only(th, dm, flux, ivar),
                prox=prox_lasso, maxiter=maxiter, tol=tol)
            theta = solver.run(best_init, l1reg).params
        else:
            solver = LBFGS(fun=smooth_objective, maxiter=maxiter, tol=tol)
            theta = solver.run(best_init).params

        # Censored coefficients are exactly zero.
        theta = theta * mask

        residuals_squared = (flux - jnp.dot(theta, dm.T)) ** 2
        s2 = _fit_scatter(residuals_squared, ivar)
        fopt = smooth_objective(theta)

        fiducial = jnp.concatenate([jnp.ones(1), jnp.zeros(T - 1)])
        theta = jnp.where(no_info, fiducial, theta)
        s2 = jnp.where(no_info, jnp.inf, s2)
        fopt = jnp.where(no_info, jnp.nan, fopt)

        return (theta, s2, fopt)

    return fit


@lru_cache(maxsize=None)
def make_pixel_closed_form():
    """
    Build a pure, ``jax.vmap``-able function that fits a single pixel in closed
    form. With no regularization and no box constraints the pixel objective is a
    convex quadratic (weighted least squares), so its minimum is the
    normal-equations solution and the iterative optimizer is unnecessary.
    Memoized: every call returns the same function object so JAX's jit cache
    can reuse the compiled program across folds and sweep grid points.

    :returns:
        A function ``fit(flux, ivar, design_matrix, column_mask)`` returning
        ``(theta, s2, fopt)`` with the same conventions as the optimizer built
        by :func:`make_pixel_fitter` (censored coefficients are exactly zero;
        information-free pixels return the fiducial theta with infinite scatter).
    """

    def fit(flux, ivar, design_matrix, column_mask):

        T = design_matrix.shape[1]
        mask = column_mask.astype(design_matrix.dtype)

        # No information in this pixel -> fiducial theta with infinite scatter.
        no_info = jnp.sum(ivar) < ivar.size

        # Zero out censored columns so they cannot contribute.
        dm = design_matrix * mask[None, :]

        # Normal equations A^T C A theta = A^T C y (with C = diag(ivar)). A
        # censored column is zeroed, leaving a zero row/column in the Gram
        # matrix; adding a unit diagonal there makes it non-singular and forces
        # that coefficient to exactly zero without coupling to the active ones.
        CiA = dm * ivar[:, None]
        gram = jnp.dot(dm.T, CiA) + jnp.diag(1.0 - mask)
        rhs = jnp.dot(dm.T, flux * ivar)
        theta = jnp.linalg.solve(gram, rhs) * mask

        fiducial = jnp.concatenate([jnp.ones(1), jnp.zeros(T - 1)])
        ok = jnp.all(jnp.isfinite(theta))
        theta = jnp.where(ok, theta, fiducial)

        residuals_squared = (flux - jnp.dot(theta, dm.T)) ** 2
        s2 = _fit_scatter(residuals_squared, ivar)
        fopt = _chi_sq_only(theta, dm, flux, ivar)

        theta = jnp.where(no_info, fiducial, theta)
        s2 = jnp.where(no_info, jnp.inf, s2)
        fopt = jnp.where(no_info, jnp.nan, fopt)

        return (theta, s2, fopt)

    return fit


# --------------------------------------------------------------------------- #
#  Errors-in-variables (uncertain labels) per-pixel fits                       #
# --------------------------------------------------------------------------- #
#
#  The standard Cannon treats the training-set labels (the features that build
#  the design matrix) as exact. When the labels carry uncertainty, propagate it
#  to first order through the model: for star ``s`` and pixel coefficients
#  ``theta``, the model prediction ``m = theta . v(x_s)`` has variance
#
#      Var(m_s) ~= (theta . J_v[s])^T Sigma_s (theta . J_v[s])
#
#  where ``J_v[s] = d v / d x`` is the vectorizer Jacobian at star ``s`` and
#  ``Sigma_s`` is the (scaled) label covariance. With diagonal label errors this
#  is ``sum_l (theta . J_v[s])_l^2 * var_label[s, l]``. The propagated variance
#  is folded into the per-star weight exactly as the scatter term is, via
#  ``ivar_eff = ivar / (1 + ivar * var_label)``. Because the weight then depends
#  on ``theta``, the weighted least-squares solve is iterated to a fixed point
#  (iteratively reweighted least squares); a handful of iterations suffices.


def _label_variance_term(theta, label_jac, label_var):
    """
    First-order propagated model variance from diagonal label errors.

    :param theta:
        The pixel coefficients, shape ``(T,)``.

    :param label_jac:
        The vectorizer Jacobian ``d v / d x`` at every star, shape
        ``(S, T, L)`` (in the scaled label space the design matrix uses).

    :param label_var:
        The per-star variance of the scaled labels, shape ``(S, L)``.

    :returns:
        The propagated model variance for every star, shape ``(S,)``.
    """
    g = jnp.einsum("t,stl->sl", theta, label_jac)   # d(theta . v)/dx per star
    return jnp.sum(g * g * label_var, axis=1)


def make_pixel_closed_form_eiv(label_jac, label_var, n_irls=5):
    """
    Closed-form per-pixel fit that accounts for diagonal label uncertainties via
    iteratively reweighted least squares. Same conventions and return signature
    as :func:`make_pixel_closed_form`; ``label_jac`` (S, T, L) and ``label_var``
    (S, L) are shared across pixels and closed over.
    """

    def fit(flux, ivar, design_matrix, column_mask):

        T = design_matrix.shape[1]
        mask = column_mask.astype(design_matrix.dtype)
        no_info = jnp.sum(ivar) < ivar.size
        dm = design_matrix * mask[None, :]
        fiducial = jnp.concatenate([jnp.ones(1), jnp.zeros(T - 1)])

        def solve(w):
            # Weighted normal equations with per-star weights ``w``. The unit on
            # the censored diagonal keeps the Gram matrix non-singular and pins
            # those coefficients to exactly zero (see make_pixel_closed_form).
            CiA = dm * w[:, None]
            gram = jnp.dot(dm.T, CiA) + jnp.diag(1.0 - mask)
            rhs = jnp.dot(dm.T, flux * w)
            return jnp.linalg.solve(gram, rhs) * mask

        def reweight(theta):
            v_label = _label_variance_term(theta, label_jac, label_var)
            return ivar / (1.0 + ivar * v_label)

        def step(_, theta):
            return solve(reweight(theta))

        # Start from the exact-label solution, then reweight to a fixed point.
        theta = lax.fori_loop(0, n_irls, step, solve(ivar))

        ok = jnp.all(jnp.isfinite(theta))
        theta = jnp.where(ok, theta, fiducial)

        w = reweight(theta)
        residuals_squared = (flux - jnp.dot(theta, dm.T)) ** 2
        s2 = _fit_scatter(residuals_squared, w)
        fopt = _chi_sq_only(theta, dm, flux, w)

        theta = jnp.where(no_info, fiducial, theta)
        s2 = jnp.where(no_info, jnp.inf, s2)
        fopt = jnp.where(no_info, jnp.nan, fopt)

        return (theta, s2, fopt)

    return fit


def make_pixel_fitter_eiv(label_jac, label_var, n_irls=5, op_method="l_bfgs_b",
    maxiter=_TRAIN_MAXITER, tol=_TRAIN_TOL, bounds=None):
    """
    Regularized/bounded per-pixel fit that accounts for diagonal label
    uncertainties. Same conventions and return signature as the optimizer built
    by :func:`make_pixel_fitter`, wrapped in an outer iteratively-reweighted
    loop: each iteration fixes the per-star weights from the current ``theta``
    and re-solves the (regularized) weighted problem. ``label_jac`` (S, T, L)
    and ``label_var`` (S, L) are shared across pixels and closed over.
    """

    op_method = (op_method or "l_bfgs_b").lower()
    if op_method == "powell":
        op_method = "l_bfgs_b"
    if op_method not in ("l_bfgs_b", "proximal"):
        raise ValueError("unknown optimization method '{}' -- 'l_bfgs_b' or "
                         "'proximal' are available".format(op_method))
    if bounds is not None:
        lower, upper = (jnp.asarray(bounds[0]), jnp.asarray(bounds[1]))

    def fit(flux, ivar, init_stack, design_matrix, regularization, column_mask):

        T = design_matrix.shape[1]
        mask = column_mask.astype(design_matrix.dtype)
        no_info = jnp.sum(ivar) < ivar.size
        dm = design_matrix * mask[None, :]
        fiducial = jnp.concatenate([jnp.ones(1), jnp.zeros(T - 1)])

        def objective(theta, w):
            return _chi_sq_only(theta, dm, flux, w) \
                + regularization * jnp.sum(jnp.abs(theta[1:]))

        # Weighted solve, warm-started from ``start``, with per-star weights ``w``.
        if bounds is not None:
            lower_eff = jnp.where(column_mask, lower, -jnp.inf)
            upper_eff = jnp.where(column_mask, upper, jnp.inf)

            def solve(start, w):
                s = jnp.clip(start, lower_eff, upper_eff)
                solver = LBFGSB(fun=lambda th: objective(th, w),
                                maxiter=maxiter, tol=tol)
                return solver.run(s, bounds=(lower_eff, upper_eff)).params
        elif op_method == "proximal":
            def solve(start, w):
                l1reg = jnp.full((T,), regularization).at[0].set(0.0) * mask
                solver = ProximalGradient(
                    fun=lambda th: _chi_sq_only(th, dm, flux, w),
                    prox=prox_lasso, maxiter=maxiter, tol=tol)
                return solver.run(start, l1reg).params
        else:
            def solve(start, w):
                solver = LBFGS(fun=lambda th: objective(th, w),
                               maxiter=maxiter, tol=tol)
                return solver.run(start).params

        def reweight(theta):
            v_label = _label_variance_term(theta, label_jac, label_var)
            return ivar / (1.0 + ivar * v_label)

        # Choose the best starting point against the raw-ivar objective.
        feval = jax.vmap(lambda th: objective(th, ivar))(init_stack)
        feval = jnp.where(jnp.isnan(feval), jnp.inf, feval)
        best_init = init_stack[jnp.argmin(feval)]

        # First solve with raw weights, then reweight to a fixed point.
        theta0 = solve(best_init, ivar)

        def step(_, theta):
            return solve(theta, reweight(theta))

        theta = lax.fori_loop(0, n_irls, step, theta0) * mask

        w = reweight(theta)
        residuals_squared = (flux - jnp.dot(theta, dm.T)) ** 2
        s2 = _fit_scatter(residuals_squared, w)
        fopt = objective(theta, w)

        theta = jnp.where(no_info, fiducial, theta)
        s2 = jnp.where(no_info, jnp.inf, s2)
        fopt = jnp.where(no_info, jnp.nan, fopt)

        return (theta, s2, fopt)

    return fit


def fit_pixel_fixed_scatter(flux, ivar, initial_thetas, design_matrix,
    regularization, censoring_mask, **kwargs):
    """
    Fit theta coefficients and noise residual for a single pixel, using
    an initially fixed scatter value.

    :param flux:
        The normalized flux values.

    :param ivar:
        The inverse variance array for the normalized fluxes.

    :param initial_thetas:
        A list of initial theta values to start from, and their source. For
        example: ``[(theta_0, "guess"), (theta_1, "old_theta")]``.

    :param design_matrix:
        The model design matrix. Censored coefficients may be indicated by
        columns that are entirely non-finite (the historical convention).

    :param regularization:
        The regularization strength to apply during optimization (Lambda).

    :param censoring_mask:
        A per-label censoring mask for each pixel. (Unused directly here; the
        censored coefficients are inferred from the design matrix.)

    :keyword op_method:
        The optimization method to use. Valid options are: ``l_bfgs_b``,
        ``proximal``. (``powell`` is accepted for backwards compatibility but
        falls back to ``l_bfgs_b``.)

    :returns:
        The optimized theta coefficients, the noise residual ``s2``, and
        metadata related to the optimization process.
    """

    flux = jnp.asarray(flux)
    ivar = jnp.asarray(ivar)
    design_matrix = jnp.asarray(design_matrix)

    T = design_matrix.shape[1]

    # Censored coefficients are those whose design-matrix column is entirely
    # non-finite (the original convention used `numpy.nan` as a fill value).
    column_mask = jnp.any(jnp.isfinite(design_matrix), axis=0)
    # Replace non-finite entries so the masked-out columns are safe to use.
    design_matrix = jnp.where(jnp.isfinite(design_matrix), design_matrix, 0.0)

    # Stack the candidate initial thetas into a fixed (n_init, T) array.
    init_stack = jnp.atleast_2d(
        jnp.asarray([np.asarray(theta) for theta, _ in initial_thetas]))

    op_method = kwargs.get("op_method", "l_bfgs_b")
    op_kwds = kwargs.get("op_kwds", {}) or {}
    maxiter = op_kwds.get("maxiter", _TRAIN_MAXITER)
    tol = op_kwds.get("tol", _TRAIN_TOL)

    t_init = time()
    fitter = make_pixel_fitter(op_method=op_method, maxiter=maxiter, tol=tol)
    theta, s2, fopt = fitter(
        flux, ivar, init_stack, design_matrix, regularization, column_mask)

    theta = np.asarray(theta)
    s2 = float(np.asarray(s2))

    # Determine which starting point was selected, for metadata.
    metadata = dict(
        op_method=("l_bfgs_b" if (op_method or "l_bfgs_b").lower() == "powell"
                   else (op_method or "l_bfgs_b").lower()),
        op_time=time() - t_init,
        fopt=float(np.asarray(fopt)),
        initial_theta=np.asarray(init_stack[0]),
        initial_theta_source=initial_thetas[0][1] if initial_thetas else None)

    return (theta, s2, metadata)


# --------------------------------------------------------------------------- #
#  Per-spectrum label inference (test step)                                    #
# --------------------------------------------------------------------------- #

def _freeze_terms(terms):
    """ A hashable (nested-tuple) copy of a vectorizer's ``terms`` structure. """
    return tuple(tuple(tuple(t) for t in term) for term in terms)


def spectrum_fitter_core(vectorizer, maxiter=_TEST_MAXITER, tol=_TEST_TOL):
    """
    Return the pure, ``jax.vmap``-able single-spectrum fitter

        ``core(flux, ivar, initial_labels, theta, s2, fiducials, scales)``

    for the given vectorizer. The trained-model arrays are call-time arguments
    (not closed-over constants), and the core is memoized on the vectorizer's
    ``terms``, so models that share a vectorizer structure -- e.g. every
    cross-validation fold of a sweep grid point -- reuse one compiled program
    instead of recompiling per trained model.
    """
    key = (type(vectorizer).__name__, _freeze_terms(vectorizer.terms),
           maxiter, tol)
    try:
        return _SPECTRUM_FITTER_CACHE[key]
    except KeyError:
        core = _build_spectrum_fitter(vectorizer, maxiter, tol)
        _SPECTRUM_FITTER_CACHE[key] = core
        return core


_SPECTRUM_FITTER_CACHE = {}


def make_spectrum_fitter(vectorizer, theta, s2, fiducials, scales,
    maxiter=_TEST_MAXITER, tol=_TEST_TOL):
    """
    Build a pure, ``jax.vmap``-able function that infers stellar labels for a
    single spectrum via Levenberg-Marquardt nonlinear least squares.

    :returns:
        A function ``core(flux, ivar, initial_labels)`` returning
        ``(op_labels, cov, chi_sq, model_flux, n_use)`` where ``initial_labels``
        is a ``(n_init, L)`` array of starting points (the best is selected).
    """

    fitter = spectrum_fitter_core(vectorizer, maxiter=maxiter, tol=tol)
    theta = jnp.asarray(theta)
    s2 = jnp.asarray(s2)
    fiducials = jnp.asarray(fiducials)
    scales = jnp.asarray(scales)

    def core(flux, ivar, initial_labels):
        return fitter(flux, ivar, initial_labels, theta, s2, fiducials, scales)

    return core


def _build_spectrum_fitter(vectorizer, maxiter, tol):

    def core(flux, ivar, initial_labels, theta, s2, fiducials, scales):

        adjusted_ivar = ivar / (1. + ivar * s2)

        # Exclude non-finite / zero-information pixels by zero-weighting them
        # (rather than removing them) so the residual vector keeps a fixed size.
        use = jnp.isfinite(flux * adjusted_ivar) & (adjusted_ivar > 0)
        weights = jnp.sqrt(jnp.where(use, adjusted_ivar, 0.0))
        flux_safe = jnp.where(use, flux, 0.0)
        safe_theta = jnp.where(use[:, None], jnp.nan_to_num(theta), 0.0)

        def model_flux(scaled_params):
            return jnp.dot(safe_theta, vectorizer(scaled_params))[:, 0]

        def residuals(scaled_params):
            return weights * (model_flux(scaled_params) - flux_safe)

        lm = LevenbergMarquardt(
            residual_fun=residuals, maxiter=maxiter, tol=tol, xtol=tol,
            gtol=tol)

        def solve_one(x0):
            scaled_x0 = (x0 - fiducials) / scales
            params = lm.run(scaled_x0).params
            r = residuals(params)
            return params, jnp.sum(r ** 2)

        params_all, chi_sqs = jax.vmap(solve_one)(initial_labels)
        chi_sqs = jnp.where(jnp.isnan(chi_sqs), jnp.inf, chi_sqs)
        best = jnp.argmin(chi_sqs)
        params = params_all[best]
        chi_sq_best = chi_sqs[best]

        # Covariance = inv(J^T J), where J is the Jacobian of the (weighted)
        # residuals. This matches scipy.optimize.leastsq's `cov_x`.
        J = jax.jacfwd(residuals)(params)
        cov = jnp.linalg.inv(jnp.dot(J.T, J))

        op_labels = params * scales + fiducials
        return (op_labels, cov, chi_sq_best, model_flux(params), jnp.sum(use))

    return core


def fit_spectrum(flux, ivar, initial_labels, vectorizer, theta, s2, fiducials,
    scales, dispersion=None, use_derivatives=True, op_kwds=None):
    """
    Fit a single spectrum by least-squares fitting to solve for labels.

    :param flux:
        The normalized flux values.

    :param ivar:
        The inverse variance array for the normalized fluxes.

    :param initial_labels:
        The point(s) to initialize optimization from.

    :param vectorizer:
        The vectorizer to use when fitting the data.

    :param theta:
        The theta coefficients (spectral derivatives) of the trained model.

    :param s2:
        The pixel scatter (s^2) array for each pixel.

    :param fiducials:
        The fiducial label values used to scale the labels.

    :param scales:
        The scale values used to normalize the labels.

    :param dispersion: [optional]
        The dispersion (e.g., wavelength) points for the normalized fluxes.

    :param use_derivatives: [optional]
        Retained for API compatibility. The Levenberg-Marquardt optimizer now
        always uses Jacobians obtained by automatic differentiation.

    :param op_kwds: [optional]
        Optimization keywords. ``maxiter`` and ``tol`` are honoured.

    :returns:
        A three-length tuple containing: the optimized labels, the covariance
        matrix, and metadata associated with the optimization.
    """

    op_kwds = op_kwds or {}
    maxiter = op_kwds.get("maxiter", _TEST_MAXITER)
    tol = op_kwds.get("tol", _TEST_TOL)

    L = len(vectorizer.label_names)

    flux = jnp.asarray(flux)
    ivar = jnp.asarray(ivar)
    adjusted_ivar = ivar / (1. + ivar * jnp.asarray(s2))
    if not bool(jnp.any(jnp.isfinite(flux * adjusted_ivar)
                        & (adjusted_ivar > 0))):
        logger.warn("No information in spectrum!")
        return (np.nan * np.ones(L), None,
                {"fail_message": "Pixels contained no information"})

    core = make_spectrum_fitter(
        vectorizer, theta, s2, fiducials, scales, maxiter=maxiter, tol=tol)

    initial_labels = jnp.atleast_2d(jnp.asarray(initial_labels))
    op_labels, cov, chi_sq_value, model_flux, n_use = core(
        flux, ivar, initial_labels)

    op_labels = np.asarray(op_labels)
    cov = np.asarray(cov)
    chi_sq_value = float(np.asarray(chi_sq_value))
    n_use = int(np.asarray(n_use))

    if cov is None or not np.any(np.isfinite(cov)):
        logger.warn("Non-finite covariance matrix returned!")
        if cov is None:
            cov = np.ones((L, L))

    meta = {
        "chi_sq": chi_sq_value,
        "r_chi_sq": chi_sq_value / max(1, (n_use - L - 1)),
        "model_flux": np.asarray(model_flux),
        "method": "levenberg_marquardt",
        "label_names": vectorizer.label_names,
        "derivatives_used": True,
        "maxiter": maxiter,
        "tol": tol,
    }

    return (op_labels, cov, meta)
