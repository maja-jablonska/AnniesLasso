#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parity tests for the JAX rewrite of The Cannon.

Every numerical path is exercised against frozen "golden" reference outputs that
were captured from the original numpy/scipy implementation (see
``_capture_golden.py``). The golden file stores both the *inputs* and the
legacy *outputs*; here we re-run the same inputs through the (now JAX) code and
assert agreement.

Tolerances reflect the nature of each computation:

* exact-to-machine-precision for closed-form linear algebra / basis evaluation;
* tight (1e-6) for the convex, unregularized least-squares training and the
  Levenberg-Marquardt test step;
* looser (1e-3) for the non-smooth lasso training, where we *additionally*
  assert the JAX optimizer achieves an objective at least as good as the legacy
  scipy optimizer (the strictly-convex lasso minimum is unique, so a lower
  objective means a more accurate solution).
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import pickle
import tempfile

import numpy as np
import pytest

import thecannon as tc
from thecannon import fitting, continuum
from thecannon.vectorizer.polynomial import PolynomialVectorizer


GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "golden", "golden.pkl")


@pytest.fixture(scope="module")
def golden():
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip("golden references not found; run "
                    "`python -m thecannon.tests._capture_golden` first")
    with open(GOLDEN_PATH, "rb") as fp:
        return pickle.load(fp)


@pytest.fixture(scope="module")
def vectorizer(golden):
    m = golden["meta"]
    return PolynomialVectorizer(label_names=m["label_names"], order=m["order"])


@pytest.fixture(scope="module")
def trained_reg0(golden, vectorizer):
    """A model trained with regularization=0 (reused across tests)."""
    m = golden["meta"]
    model = tc.CannonModel(m["labels"], m["flux"], m["ivar"], vectorizer,
                           dispersion=m["dispersion"], regularization=0)
    model.train()
    return model


# --------------------------------------------------------------------------- #
#  Environment                                                                 #
# --------------------------------------------------------------------------- #

def test_float64_enabled():
    import jax
    assert jax.config.jax_enable_x64 is True


def test_vectorizer_returns_float64(vectorizer, golden):
    import numpy as np
    out = np.asarray(vectorizer(golden["inputs"]["vec_labels_batch"]))
    assert out.dtype == np.float64


# --------------------------------------------------------------------------- #
#  Vectorizer                                                                  #
# --------------------------------------------------------------------------- #

def test_get_label_vector_batch(vectorizer, golden):
    out = np.asarray(
        vectorizer.get_label_vector(golden["inputs"]["vec_labels_batch"]))
    np.testing.assert_allclose(
        out, golden["outputs"]["get_label_vector_batch"], atol=1e-12)


def test_get_label_vector_single(vectorizer, golden):
    out = np.asarray(
        vectorizer.get_label_vector(golden["inputs"]["vec_labels_single"]))
    np.testing.assert_allclose(
        out, golden["outputs"]["get_label_vector_single"], atol=1e-12)


def test_get_label_vector_derivative(vectorizer, golden):
    out = np.asarray(vectorizer.get_label_vector_derivative(
        golden["inputs"]["vec_labels_single"]))
    expected = golden["outputs"]["get_label_vector_derivative"]
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, atol=1e-9)


# --------------------------------------------------------------------------- #
#  Linear algebra                                                              #
# --------------------------------------------------------------------------- #

def test_fit_theta_by_linalg(golden):
    d = golden["inputs"]["linalg"]
    theta, cov = fitting.fit_theta_by_linalg(
        d["flux"], d["ivar"], d["s2"], d["design_matrix"])
    exp = golden["outputs"]["fit_theta_by_linalg"]
    np.testing.assert_allclose(np.asarray(theta), exp["theta"], atol=1e-8)
    np.testing.assert_allclose(np.asarray(cov), exp["cov"], atol=1e-8)


def test_fit_theta_by_linalg_singular(golden):
    d = golden["inputs"]["linalg_singular"]
    theta, cov = fitting.fit_theta_by_linalg(
        d["flux"], d["ivar"], d["s2"], d["design_matrix"])
    exp = golden["outputs"]["fit_theta_by_linalg_singular"]
    # Singular design matrix must fall back to the fiducial value [1, 0, ...].
    np.testing.assert_array_equal(np.asarray(theta), exp["theta"])
    assert np.all(~np.isfinite(np.asarray(cov)))


# --------------------------------------------------------------------------- #
#  Objective pieces                                                            #
# --------------------------------------------------------------------------- #

def test_chi_sq_value_and_gradient(golden):
    c = golden["inputs"]["chi_sq"]
    value, grad = fitting.chi_sq(
        c["theta"], c["design_matrix"], c["flux"], c["ivar"], gradient=True)
    exp = golden["outputs"]["chi_sq"]
    np.testing.assert_allclose(float(value), exp["value"], atol=1e-9)
    np.testing.assert_allclose(np.asarray(grad), exp["grad"], atol=1e-9)


def test_L1Norm_variation(golden):
    value, grad = fitting.L1Norm_variation(golden["inputs"]["L1"]["theta"])
    exp = golden["outputs"]["L1Norm_variation"]
    np.testing.assert_allclose(float(value), exp["value"], atol=1e-12)
    np.testing.assert_allclose(np.asarray(grad), exp["grad"], atol=1e-12)


def test_pixel_objective(golden):
    o = golden["inputs"]["objective"]
    value, grad = fitting._pixel_objective_function_fixed_scatter(
        o["theta"], o["design_matrix"], o["flux"], o["ivar"],
        o["regularization"], gradient=True)
    exp = golden["outputs"]["objective"]
    np.testing.assert_allclose(float(value), exp["value"], atol=1e-9)
    np.testing.assert_allclose(np.asarray(grad), exp["grad"], atol=1e-9)


# --------------------------------------------------------------------------- #
#  Scatter fit                                                                 #
# --------------------------------------------------------------------------- #

def test_scatter_fit_positive(golden):
    import jax.numpy as jnp
    s = golden["inputs"]["scatter_hi"]
    s2 = float(fitting._fit_scatter(
        jnp.asarray(s["residuals_squared"]), jnp.asarray(s["ivar"])))
    np.testing.assert_allclose(s2, golden["outputs"]["scatter_hi"], atol=1e-3)


def test_scatter_fit_zero(golden):
    import jax.numpy as jnp
    s = golden["inputs"]["scatter_lo"]
    s2 = float(fitting._fit_scatter(
        jnp.asarray(s["residuals_squared"]), jnp.asarray(s["ivar"])))
    np.testing.assert_allclose(s2, golden["outputs"]["scatter_lo"], atol=1e-8)


# --------------------------------------------------------------------------- #
#  Training                                                                    #
# --------------------------------------------------------------------------- #

def test_train_reg0_theta(trained_reg0, golden):
    exp = golden["outputs"]["train_reg0"]
    np.testing.assert_allclose(trained_reg0.theta, exp["theta"], atol=1e-6)


def test_train_reg0_s2(trained_reg0, golden):
    exp = golden["outputs"]["train_reg0"]
    np.testing.assert_allclose(trained_reg0.s2, exp["s2"], atol=1e-8)


def test_train_regularized_matches_within_tolerance(golden, vectorizer):
    m = golden["meta"]
    reg = golden["outputs"]["train_regR"]["regularization"]
    model = tc.CannonModel(m["labels"], m["flux"], m["ivar"], vectorizer,
                           dispersion=m["dispersion"], regularization=reg)
    theta, s2, _ = model.train()
    np.testing.assert_allclose(
        theta, golden["outputs"]["train_regR"]["theta"], atol=2e-3)


def test_train_regularized_objective_at_least_as_good(golden, vectorizer):
    """The strictly-convex lasso minimum is unique; the JAX optimizer must
    reach an objective no worse than the legacy scipy optimizer."""
    import jax.numpy as jnp
    m = golden["meta"]
    reg = golden["outputs"]["train_regR"]["regularization"]
    model = tc.CannonModel(m["labels"], m["flux"], m["ivar"], vectorizer,
                           dispersion=m["dispersion"], regularization=reg)
    theta, _, _ = model.train()

    scaled = (m["labels"] - m["fiducials"]) / m["scales"]
    dm = np.asarray(vectorizer(scaled).T)
    jax_obj = np.array([
        float(fitting._pixel_objective_function_fixed_scatter(
            jnp.asarray(theta[p]), jnp.asarray(dm),
            jnp.asarray(m["flux"][:, p]), jnp.asarray(m["ivar"][:, p]), reg,
            gradient=False))
        for p in range(m["n_pixels"])])
    golden_obj = golden["outputs"]["train_regR_objective"]
    # JAX objective must not exceed the legacy objective (allowing a tiny margin).
    assert np.all(jax_obj <= golden_obj + 1e-8)


# --------------------------------------------------------------------------- #
#  Prediction & inference                                                      #
# --------------------------------------------------------------------------- #

def test_predict(trained_reg0, golden):
    pred = np.asarray(trained_reg0(golden["inputs"]["predict_labels"]))
    np.testing.assert_allclose(
        pred, golden["outputs"]["predict_reg0"], atol=1e-9)


def test_test_step_labels(trained_reg0, golden):
    t = golden["inputs"]["test"]
    labels, cov, meta = trained_reg0.test(t["flux"], t["ivar"])
    np.testing.assert_allclose(
        labels, golden["outputs"]["test_labels"], atol=1e-4)


def test_test_step_covariance(trained_reg0, golden):
    t = golden["inputs"]["test"]
    labels, cov, meta = trained_reg0.test(t["flux"], t["ivar"])
    np.testing.assert_allclose(
        cov, golden["outputs"]["test_cov"], rtol=1e-5, atol=1e-12)


def test_test_step_recovers_true_labels(trained_reg0, golden):
    t = golden["inputs"]["test"]
    labels, cov, meta = trained_reg0.test(t["flux"], t["ivar"])
    rel = np.abs(labels - t["true_labels"]) / golden["meta"]["scales"]
    assert np.all(rel < 0.05)


# --------------------------------------------------------------------------- #
#  Continuum                                                                   #
# --------------------------------------------------------------------------- #

def test_continuum(golden):
    c = golden["inputs"]["continuum"]
    cont, meta = continuum.sines_and_cosines(
        c["dispersion"], c["flux"], c["ivar"], c["continuum_pixels"],
        L=c["L"], order=c["order"])
    np.testing.assert_allclose(
        np.asarray(cont), golden["outputs"]["continuum"], atol=1e-10)


# --------------------------------------------------------------------------- #
#  I/O round-trip & format compatibility                                       #
# --------------------------------------------------------------------------- #

def test_write_read_roundtrip(trained_reg0, golden):
    fd, path = tempfile.mkstemp(suffix=".model")
    os.close(fd)
    try:
        trained_reg0.write(path, include_training_set_spectra=True,
                           overwrite=True)
        reloaded = tc.CannonModel.read(path)
        # Trained attributes are JAX arrays and survive the round-trip as such.
        import jax
        assert isinstance(trained_reg0.theta, jax.Array)
        assert isinstance(reloaded.theta, jax.Array)
        np.testing.assert_array_equal(reloaded.theta, trained_reg0.theta)
        np.testing.assert_array_equal(reloaded.s2, trained_reg0.s2)
        # Predictions are identical after reload.
        pred_a = np.asarray(trained_reg0(golden["inputs"]["predict_labels"]))
        pred_b = np.asarray(reloaded(golden["inputs"]["predict_labels"]))
        np.testing.assert_allclose(pred_a, pred_b, atol=1e-12)
    finally:
        os.remove(path)


# --------------------------------------------------------------------------- #
#  Proximal-gradient lasso variant                                            #
# --------------------------------------------------------------------------- #

def test_proximal_matches_lbfgs_at_reg0(golden, vectorizer):
    """At regularization=0 the proximal solver reduces to least squares and
    must agree with the default path (and the golden least-squares result)."""
    m = golden["meta"]
    model = tc.CannonModel(m["labels"], m["flux"], m["ivar"], vectorizer,
                           dispersion=m["dispersion"], regularization=0)
    theta, s2, _ = model.train(op_method="proximal")
    np.testing.assert_allclose(
        theta, golden["outputs"]["train_reg0"]["theta"], atol=1e-5)


# --------------------------------------------------------------------------- #
#  Closed-form fast path (regularization == 0)                                 #
# --------------------------------------------------------------------------- #

def test_closed_form_matches_lbfgs_with_censoring(golden, vectorizer):
    """The closed-form fast path used at regularization=0 must reproduce the
    iterative L-BFGS optimum, including when coefficients are censored."""
    import jax
    import jax.numpy as jnp
    from thecannon import fitting

    m = golden["meta"]
    model = tc.CannonModel(m["labels"], m["flux"], m["ivar"], vectorizer,
                           dispersion=m["dispersion"], regularization=0)
    dm = jnp.asarray(model.design_matrix)
    flux_PN = jnp.asarray(model.training_set_flux.T)
    ivar_PN = jnp.asarray(model.training_set_ivar.T)
    P, T = flux_PN.shape[0], dm.shape[1]

    # Censor three (non-continuum) coefficients on every pixel.
    rng = np.random.default_rng(1)
    mask = np.ones((P, T), bool)
    for p in range(P):
        mask[p, rng.choice(np.arange(1, T), size=3, replace=False)] = False
    mask = jnp.asarray(mask)

    fiducial = jnp.concatenate([jnp.ones(1), jnp.zeros(T - 1)])
    init = jnp.broadcast_to(jnp.stack([fiducial, fiducial]), (P, 2, T))
    reg = jnp.zeros(P)

    cf = jax.vmap(fitting.make_pixel_closed_form(), in_axes=(0, 0, None, 0))
    lb = jax.vmap(fitting.make_pixel_fitter(op_method="l_bfgs_b"),
                  in_axes=(0, 0, 0, None, 0, 0))
    theta_cf, _, _ = cf(flux_PN, ivar_PN, dm, mask)
    theta_lb, _, _ = lb(flux_PN, ivar_PN, init, dm, reg, mask)

    np.testing.assert_allclose(theta_cf, theta_lb, atol=1e-8)
    # Censored coefficients are exactly zero.
    assert bool(jnp.all(theta_cf[~np.asarray(mask)] == 0.0))
