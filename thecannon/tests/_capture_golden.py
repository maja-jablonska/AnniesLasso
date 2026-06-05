#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Capture golden reference inputs+outputs from the *legacy* numpy/scipy
implementation of The Cannon, so the JAX rewrite can be verified to yield the
same results.

Run this ONCE against the numpy/scipy code (before the JAX port), e.g.:

    python -m thecannon.tests._capture_golden

It writes ``thecannon/tests/golden/golden.pkl`` containing a dictionary with
``inputs`` (everything needed to re-run each numerical path) and ``outputs``
(the legacy results). The parity tests in ``test_jax_parity.py`` load this file,
re-run the same inputs through the (now JAX) code, and assert agreement.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import pickle
import numpy as np

import thecannon as tc
from thecannon import fitting, continuum
from thecannon.vectorizer.polynomial import PolynomialVectorizer


HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_PATH = os.path.join(HERE, "golden", "golden.pkl")

SEED = 20240601
LABEL_NAMES = ("TEFF", "LOGG", "FEH")
ORDER = 2
N_STARS = 40
N_PIXELS = 60


def build_synthetic():
    """Build a small, deterministic synthetic training set."""
    rng = np.random.RandomState(SEED)

    vectorizer = PolynomialVectorizer(label_names=LABEL_NAMES, order=ORDER)
    T = 1 + len(vectorizer.terms)  # number of theta coefficients per pixel

    # Physically-ish ranges, then we work in scaled space internally.
    teff = rng.uniform(4000.0, 6500.0, size=N_STARS)
    logg = rng.uniform(1.0, 4.5, size=N_STARS)
    feh = rng.uniform(-1.5, 0.4, size=N_STARS)
    labels = np.vstack([teff, logg, feh]).T  # (N_STARS, 3)

    # Fiducials/scales the way CannonModel computes them.
    scales = np.ptp(np.percentile(labels, [2.5, 97.5], axis=0), axis=0)
    fiducials = np.percentile(labels, 50, axis=0)
    scaled = (labels - fiducials) / scales

    design_matrix = vectorizer(scaled).T  # (N_STARS, T)

    # A "true" theta with a sensible continuum (~1) and clear label structure
    # (boost the linear terms so the inverse/inference problem is well posed).
    true_theta = 0.05 * rng.randn(N_PIXELS, T)
    true_theta[:, 0] = 1.0
    K = len(LABEL_NAMES)
    true_theta[:, 1:1 + K] += 0.3 * rng.randn(N_PIXELS, K)

    clean_flux = design_matrix @ true_theta.T  # (N_STARS, N_PIXELS)
    sigma = 0.01
    noise = sigma * rng.randn(N_STARS, N_PIXELS)
    flux = clean_flux + noise
    ivar = np.ones_like(flux) / (sigma ** 2)

    dispersion = np.linspace(15000.0, 15000.0 + N_PIXELS - 1, N_PIXELS)

    return dict(
        vectorizer=vectorizer, labels=labels, scaled=scaled,
        scales=scales, fiducials=fiducials, design_matrix=design_matrix,
        true_theta=true_theta, flux=flux, ivar=ivar, dispersion=dispersion,
        T=T, sigma=sigma)


def capture():
    rng = np.random.RandomState(SEED + 1)
    syn = build_synthetic()
    vectorizer = syn["vectorizer"]
    design_matrix = syn["design_matrix"]
    flux, ivar = syn["flux"], syn["ivar"]
    scaled = syn["scaled"]
    T = syn["T"]

    inputs = {}
    outputs = {}

    # ---- 1. Vectorizer ----------------------------------------------------
    # Batched label vector.
    inputs["vec_labels_batch"] = scaled.copy()
    outputs["get_label_vector_batch"] = vectorizer.get_label_vector(scaled)

    # Single label vector.
    single = scaled[0].copy()
    inputs["vec_labels_single"] = single
    outputs["get_label_vector_single"] = vectorizer.get_label_vector(single)

    # Analytic derivative (single 1-D labels -> (T, L)).
    outputs["get_label_vector_derivative"] = \
        vectorizer.get_label_vector_derivative(single)

    # ---- 2. fit_theta_by_linalg ------------------------------------------
    pix = 3
    f0, i0 = flux[:, pix].copy(), ivar[:, pix].copy()
    inputs["linalg"] = dict(flux=f0, ivar=i0, s2=0.0,
                            design_matrix=design_matrix.copy())
    th, cov = fitting.fit_theta_by_linalg(f0, i0, 0.0, design_matrix.copy())
    outputs["fit_theta_by_linalg"] = dict(theta=th, cov=cov)

    # Singular case: a rank-deficient design matrix should hit the fallback.
    singular_dm = np.ones((N_STARS, T))  # all columns identical -> singular
    th_s, cov_s = fitting.fit_theta_by_linalg(f0, i0, 0.0, singular_dm.copy())
    inputs["linalg_singular"] = dict(flux=f0, ivar=i0, s2=0.0,
                                     design_matrix=singular_dm)
    outputs["fit_theta_by_linalg_singular"] = dict(theta=th_s, cov=cov_s)

    # ---- 3. chi_sq + L1Norm_variation + objective ------------------------
    theta_test = th.copy()
    inputs["chi_sq"] = dict(theta=theta_test, design_matrix=design_matrix.copy(),
                            flux=f0, ivar=i0)
    csq_val, csq_grad = fitting.chi_sq(theta_test, design_matrix, f0, i0,
                                       gradient=True)
    outputs["chi_sq"] = dict(value=np.asarray(csq_val), grad=np.asarray(csq_grad))

    l1_val, l1_grad = fitting.L1Norm_variation(theta_test)
    inputs["L1"] = dict(theta=theta_test)
    outputs["L1Norm_variation"] = dict(value=np.asarray(l1_val),
                                       grad=np.asarray(l1_grad))

    reg = 10.0
    inputs["objective"] = dict(theta=theta_test,
                               design_matrix=design_matrix.copy(),
                               flux=f0, ivar=i0, regularization=reg)
    obj_val, obj_grad = fitting._pixel_objective_function_fixed_scatter(
        theta_test, design_matrix, f0, i0, reg, gradient=True)
    outputs["objective"] = dict(value=np.asarray(obj_val),
                                grad=np.asarray(obj_grad))

    # ---- 4. scatter fit ---------------------------------------------------
    # Build residuals that *require* a positive scatter (median > 1).
    resid_sq_hi = (rng.rand(N_STARS) * 5.0) ** 2
    ivar_sc = np.ones(N_STARS) * 50.0
    import scipy.optimize as op
    sc_hi = op.fmin(fitting._scatter_objective_function, 0.0,
                    args=(resid_sq_hi, ivar_sc), disp=False)
    inputs["scatter_hi"] = dict(residuals_squared=resid_sq_hi, ivar=ivar_sc)
    outputs["scatter_hi"] = float(np.asarray(sc_hi).ravel()[0] ** 2)  # s2

    # And residuals where median already <= 1 (expect ~0 scatter).
    resid_sq_lo = (rng.rand(N_STARS) * 0.01) ** 2
    sc_lo = op.fmin(fitting._scatter_objective_function, 0.0,
                    args=(resid_sq_lo, ivar_sc), disp=False)
    inputs["scatter_lo"] = dict(residuals_squared=resid_sq_lo, ivar=ivar_sc)
    outputs["scatter_lo"] = float(np.asarray(sc_lo).ravel()[0] ** 2)

    # ---- 5. Training: regularization = 0 (exact least squares) -----------
    model0 = tc.CannonModel(syn["labels"], flux, ivar, vectorizer,
                            dispersion=syn["dispersion"], regularization=0)
    theta0, s2_0, _ = model0.train(threads=1)
    outputs["train_reg0"] = dict(theta=np.asarray(theta0),
                                 s2=np.asarray(s2_0))

    # ---- 6. Training: regularization > 0 (lasso) -------------------------
    modelR = tc.CannonModel(syn["labels"], flux, ivar, vectorizer,
                            dispersion=syn["dispersion"], regularization=reg)
    thetaR, s2_R, _ = modelR.train(threads=1)
    outputs["train_regR"] = dict(theta=np.asarray(thetaR),
                                 s2=np.asarray(s2_R),
                                 regularization=reg)
    # The achieved objective per pixel (for optimizer-quality comparison).
    obj_per_pixel = np.array([
        fitting._pixel_objective_function_fixed_scatter(
            thetaR[p], design_matrix, flux[:, p], ivar[:, p], reg,
            gradient=False)
        for p in range(N_PIXELS)])
    outputs["train_regR_objective"] = obj_per_pixel

    # ---- 7. Prediction (__call__) ----------------------------------------
    pred_labels = syn["labels"][:5].copy()
    inputs["predict_labels"] = pred_labels
    outputs["predict_reg0"] = np.asarray(model0(pred_labels))

    # ---- 8. Test step (inference) ----------------------------------------
    n_test = 5
    test_flux = flux[:n_test].copy()
    test_ivar = ivar[:n_test].copy()
    inputs["test"] = dict(flux=test_flux, ivar=test_ivar,
                          true_labels=syn["labels"][:n_test].copy())
    rec_labels, rec_cov, _ = model0.test(test_flux, test_ivar, threads=1)
    outputs["test_labels"] = np.asarray(rec_labels)
    outputs["test_cov"] = np.asarray(rec_cov)

    # ---- 9. Continuum normalization --------------------------------------
    cont_disp = np.linspace(15000.0, 16000.0, 60)
    cont_rng = np.random.RandomState(SEED + 2)
    cont_flux = 1.0 + 0.1 * np.sin(cont_disp / 50.0) \
        + 0.02 * cont_rng.randn(3, cont_disp.size)
    cont_ivar = np.ones_like(cont_flux) / (0.02 ** 2)
    continuum_pixels = np.arange(0, cont_disp.size, 2)
    inputs["continuum"] = dict(dispersion=cont_disp, flux=cont_flux,
                               ivar=cont_ivar,
                               continuum_pixels=continuum_pixels,
                               L=1400, order=3)
    cont, _ = continuum.sines_and_cosines(
        cont_disp, cont_flux, cont_ivar, continuum_pixels, L=1400, order=3)
    outputs["continuum"] = np.asarray(cont)

    # ---- meta -------------------------------------------------------------
    meta = dict(seed=SEED, label_names=LABEL_NAMES, order=ORDER,
                n_stars=N_STARS, n_pixels=N_PIXELS, T=T,
                scales=syn["scales"], fiducials=syn["fiducials"],
                true_theta=syn["true_theta"], labels=syn["labels"],
                flux=flux, ivar=ivar, dispersion=syn["dispersion"])

    return dict(inputs=inputs, outputs=outputs, meta=meta)


def main():
    golden = capture()
    with open(GOLDEN_PATH, "wb") as fp:
        pickle.dump(golden, fp, protocol=2)
    print("Wrote golden references to {}".format(GOLDEN_PATH))
    print("Captured outputs: {}".format(sorted(golden["outputs"].keys())))


if __name__ == "__main__":
    main()
