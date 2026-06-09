#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-end smoke test of the full Cannon pipeline on a tiny synthetic data set,
driven entirely with JAX arrays.

This mirrors ``notebooks/start.ipynb`` -- continuum-normalize, build a
``CannonModel`` with a quadratic ``PolynomialVectorizer``, ``train`` it, then run
the ``test`` step and check the labels are recovered -- but on a self-contained,
deterministic toy data set so it needs no external files and runs in seconds.

Crucially every input array is a ``jax.numpy`` array (as in the notebook after
the ``jnp.array(...)`` conversion cell). This is exactly the path that used to
break: ``continuum.normalize`` did in-place ``x[mask] = 1.0`` assignment, which
JAX arrays do not support. The test asserts the pipeline runs and that the
quantities that should stay on-device (normalized flux/ivar, trained theta/s2)
are genuine ``jax.Array`` objects.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import thecannon as tc
from thecannon import continuum
from thecannon.vectorizer.polynomial import PolynomialVectorizer


LABEL_NAMES = ["teff", "logg", "feh"]
# Physical-ish centres and half-ranges, so the labels span realistic magnitudes
# (the model rescales them internally; this just exercises that scaling).
LABEL_CENTERS = np.array([4800.0, 2.5, -0.2])
LABEL_RANGES = np.array([600.0, 1.0, 0.4])

N_TRAIN = 48
N_VAL = 16
N_PIXELS = 60
SIGMA = 0.002                      # per-pixel flux noise (high S/N)


def _make_dataset():
    """
    Build a tiny synthetic spectral library with a known label dependence.

    Even-indexed pixels are flat "continuum" pixels (flux == 1 + noise); these
    are the pixels handed to the continuum fit, so the fitted continuum is ~1
    everywhere and normalization is ~identity. Odd-indexed "feature" pixels
    carry a smooth quadratic dependence on the (scaled) labels, which is exactly
    what an order-2 polynomial Cannon can represent -- so training fits it well
    and the test step inverts it back to the labels.

    Returns plain numpy; callers convert to JAX at the boundary.
    """
    rng = np.random.RandomState(20240609)

    # Scaled labels x ~ U(-1, 1); raw labels are an affine map of x.
    x_all = rng.uniform(-1.0, 1.0, size=(N_TRAIN + N_VAL, len(LABEL_NAMES)))
    labels_all = LABEL_CENTERS + LABEL_RANGES * x_all

    dispersion = np.linspace(15100.0, 16900.0, N_PIXELS)

    feature = np.zeros(N_PIXELS, dtype=bool)
    feature[1::2] = True                      # odd pixels carry label info
    continuum_pixels = np.where(~feature)[0].astype(int)

    # Per-feature-pixel linear (A) and quadratic (B) coefficients.
    A = np.zeros((N_PIXELS, len(LABEL_NAMES)))
    B = np.zeros((N_PIXELS, len(LABEL_NAMES)))
    A[feature] = rng.normal(0.0, 0.06, size=(feature.sum(), len(LABEL_NAMES)))
    B[feature] = rng.normal(0.0, 0.015, size=(feature.sum(), len(LABEL_NAMES)))

    # flux = 1 + A.x + B.x^2  (continuum pixels have A = B = 0 -> flux == 1)
    flux = 1.0 + x_all @ A.T + (x_all ** 2) @ B.T
    flux = flux + rng.normal(0.0, SIGMA, size=flux.shape)
    ivar = np.full_like(flux, 1.0 / SIGMA ** 2)

    train = np.arange(N_TRAIN)
    val = np.arange(N_TRAIN, N_TRAIN + N_VAL)
    return dict(dispersion=dispersion, flux=flux, ivar=ivar,
                continuum_pixels=continuum_pixels, labels=labels_all,
                train=train, val=val)


@pytest.fixture(scope="module")
def pipeline():
    """Run normalize -> build -> train -> test once, all with JAX arrays."""
    data = _make_dataset()

    # --- promote every input to JAX, as the notebook does before normalize ---
    dispersion = jnp.asarray(data["dispersion"])
    flux = jnp.asarray(data["flux"])
    ivar = jnp.asarray(data["ivar"])
    continuum_pixels = jnp.asarray(data["continuum_pixels"])

    normalized_flux, normalized_ivar, cont, meta = continuum.normalize(
        dispersion, flux, ivar, continuum_pixels,
        L=1400, order=3, regions=[(15100, 16900)], progressbar=False)

    train, val = data["train"], data["val"]
    labels = data["labels"]            # numpy (the CannonModel label branch)

    vectorizer = PolynomialVectorizer(label_names=LABEL_NAMES, order=2)
    model = tc.CannonModel(
        labels[train],
        normalized_flux[train],
        normalized_ivar[train],
        vectorizer,
        dispersion=dispersion,
        regularization=0)
    theta, s2, train_meta = model.train(progressbar=False)

    val_labels, val_cov, val_meta = model.test(
        normalized_flux[val], normalized_ivar[val], progressbar=False)

    return dict(model=model, theta=theta, s2=s2,
                normalized_flux=normalized_flux, normalized_ivar=normalized_ivar,
                continuum=cont, val_labels=np.asarray(val_labels),
                val_cov=np.asarray(val_cov), truth=labels[val])


# --------------------------------------------------------------------------- #
#  Normalization stays on-device and is sane                                  #
# --------------------------------------------------------------------------- #

def test_normalize_returns_jax_arrays(pipeline):
    # The exact path that previously raised on `x[mask] = 1.0`.
    assert isinstance(pipeline["normalized_flux"], jax.Array)
    assert isinstance(pipeline["normalized_ivar"], jax.Array)


def test_normalized_flux_is_finite(pipeline):
    assert bool(jnp.all(jnp.isfinite(pipeline["normalized_flux"])))
    assert bool(jnp.all(jnp.isfinite(pipeline["normalized_ivar"])))


def test_continuum_is_near_unity(pipeline):
    # Spectra are built around a flat continuum of 1, so normalization is ~no-op.
    nf = pipeline["normalized_flux"]
    assert abs(float(jnp.median(nf)) - 1.0) < 0.02


# --------------------------------------------------------------------------- #
#  Training stays on-device and is well-formed                                #
# --------------------------------------------------------------------------- #

def test_model_is_trained(pipeline):
    assert pipeline["model"].is_trained


def test_trained_quantities_are_jax_arrays(pipeline):
    assert isinstance(pipeline["theta"], jax.Array)
    assert isinstance(pipeline["s2"], jax.Array)


def test_theta_and_scatter_well_formed(pipeline):
    theta, s2 = pipeline["theta"], pipeline["s2"]
    # 10 terms for an order-2 polynomial in 3 labels (1 + 3 linear + 6 quadratic).
    n_terms = pipeline["model"].design_matrix.shape[1]
    assert theta.shape == (N_PIXELS, n_terms)
    assert s2.shape == (N_PIXELS,)
    assert bool(jnp.all(jnp.isfinite(theta)))
    assert bool(jnp.all(s2 >= 0.0))


# --------------------------------------------------------------------------- #
#  The test step inverts the model back to the true labels                    #
# --------------------------------------------------------------------------- #

def test_recovers_validation_labels(pipeline):
    pred = pipeline["val_labels"]
    truth = pipeline["truth"]
    assert pred.shape == truth.shape == (N_VAL, len(LABEL_NAMES))

    # Compare in scaled-label units so all three labels share one tolerance.
    rel = np.abs(pred - truth) / LABEL_RANGES
    assert np.all(np.isfinite(rel))
    assert rel.mean() < 0.05, f"mean scaled error too high: {rel.mean():.4f}"
    assert rel.max() < 0.15, f"max scaled error too high: {rel.max():.4f}"


def test_forward_model_shapes(pipeline):
    """The trained model can predict flux for new labels (single and batch)."""
    model = pipeline["model"]
    one = model(pipeline["truth"][0])
    many = model(pipeline["truth"][:3])
    assert np.asarray(one).shape == (N_PIXELS,)
    assert np.asarray(many).shape == (3, N_PIXELS)
