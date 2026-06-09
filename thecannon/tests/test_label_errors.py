#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for training with label (feature) uncertainties.

The standard Cannon treats the training-set labels as exact. When
``training_set_label_err`` is supplied, ``train`` propagates the label
uncertainties into the per-pixel weights and refines them with iteratively
reweighted least squares (errors-in-variables). These tests check that:

  * supplying ``None`` reproduces the exact-label fit bit-for-bit (no regression);
  * large per-star label errors down-weight the stars whose labels we do not
    trust, recovering coefficients a naive exact-label fit gets wrong;
  * the regularized path also runs and stays finite;
  * label errors survive a save/read round-trip;
  * construction validates the error array's shape and values.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import tempfile

import numpy as np
import pytest

import thecannon as tc
from thecannon.vectorizer.polynomial import PolynomialVectorizer


def _vec(order=2):
    return PolynomialVectorizer(label_names=["a", "b", "c"], order=order)


def _toy(seed=1, S=60, P=40, L=3):
    rng = np.random.RandomState(seed)
    labels = rng.normal(size=(S, L))
    flux = 1.0 + 0.02 * rng.normal(size=(S, P))
    ivar = np.full((S, P), 1e4)
    return labels, flux, ivar


# --------------------------------------------------------------------------- #
#  No regression: label_err=None is identical to the exact-label fit          #
# --------------------------------------------------------------------------- #

def test_none_matches_exact_label_fit():
    labels, flux, ivar = _toy()

    base = tc.CannonModel(labels, flux, ivar, _vec())
    t0, s0, _ = base.train(progressbar=False)

    same = tc.CannonModel(labels, flux, ivar, _vec(),
                          training_set_label_err=None)
    t1, s1, _ = same.train(progressbar=False)

    assert np.allclose(np.asarray(t0), np.asarray(t1))
    assert np.allclose(np.asarray(s0), np.asarray(s1))


# --------------------------------------------------------------------------- #
#  EIV runs, stays finite, and changes the answer                             #
# --------------------------------------------------------------------------- #

def test_eiv_closed_form_runs_and_differs():
    labels, flux, ivar = _toy()
    lerr = np.full_like(labels, 0.1)

    exact = tc.CannonModel(labels, flux, ivar, _vec())
    t_exact = np.asarray(exact.train(progressbar=False)[0])

    eiv = tc.CannonModel(labels, flux, ivar, _vec(),
                         training_set_label_err=lerr)
    theta, s2, _ = eiv.train(progressbar=False)
    theta, s2 = np.asarray(theta), np.asarray(s2)

    assert np.all(np.isfinite(theta))
    assert np.all(s2 >= 0.0)
    assert not np.allclose(theta, t_exact)


def test_eiv_regularized_runs():
    labels, flux, ivar = _toy()
    lerr = np.full_like(labels, 0.1)
    eiv = tc.CannonModel(labels, flux, ivar, _vec(), regularization=10.0,
                         training_set_label_err=lerr)
    theta, s2, _ = eiv.train(progressbar=False, n_irls=3)
    assert np.all(np.isfinite(np.asarray(theta)))
    assert np.all(np.asarray(s2) >= 0.0)


# --------------------------------------------------------------------------- #
#  Mechanism: untrusted labels are down-weighted                              #
# --------------------------------------------------------------------------- #

def test_large_label_errors_downweight_untrusted_stars():
    rng = np.random.RandomState(3)
    S, L = 200, 2
    x_true = rng.normal(size=(S, L))
    theta_true = np.array([0.5, -0.3, 0.8])           # [pivot, a, b], order-1
    flux = (theta_true[0] + x_true @ theta_true[1:]).reshape(S, 1)
    ivar = np.full((S, 1), 1e6)

    # Corrupt the recorded labels of some stars and flag them as untrusted.
    x_obs = x_true.copy()
    bad = rng.choice(S, 30, replace=False)
    x_obs[bad] += rng.normal(scale=5.0, size=(30, L))
    lerr = np.full((S, L), 1e-3)
    lerr[bad] = 50.0

    def predict(model, theta):
        scaled = (x_true - model._fiducials) / model._scales
        dm = np.asarray(model.vectorizer(scaled).T)
        return dm @ np.asarray(theta)

    vec = lambda: PolynomialVectorizer(label_names=["a", "b"], order=1)
    naive = tc.CannonModel(x_obs, flux, ivar, vec())
    t_naive = np.asarray(naive.train(progressbar=False)[0])[0]

    eiv = tc.CannonModel(x_obs, flux, ivar, vec(),
                         training_set_label_err=lerr)
    t_eiv = np.asarray(eiv.train(progressbar=False, n_irls=8)[0])[0]

    rms_naive = np.sqrt(np.mean((predict(naive, t_naive) - flux[:, 0]) ** 2))
    rms_eiv = np.sqrt(np.mean((predict(eiv, t_eiv) - flux[:, 0]) ** 2))

    # The naive fit trusts the corrupted labels and is biased; EIV all but
    # eliminates the error by down-weighting the flagged stars.
    assert rms_eiv < rms_naive / 5.0


# --------------------------------------------------------------------------- #
#  Serialization and validation                                               #
# --------------------------------------------------------------------------- #

def test_label_err_survives_round_trip():
    labels, flux, ivar = _toy()
    lerr = np.abs(np.random.RandomState(7).normal(size=labels.shape)) + 0.01
    model = tc.CannonModel(labels, flux, ivar, _vec(),
                           training_set_label_err=lerr)
    model.train(progressbar=False)

    path = os.path.join(tempfile.mkdtemp(), "model.model")
    model.write(path, overwrite=True)
    reloaded = tc.CannonModel.read(path)

    assert reloaded.training_set_label_err is not None
    assert np.allclose(np.asarray(reloaded.training_set_label_err), lerr)


def test_label_err_validates_shape():
    labels, flux, ivar = _toy()
    with pytest.raises(ValueError):
        tc.CannonModel(labels, flux, ivar, _vec(),
                       training_set_label_err=np.ones((labels.shape[0], 99)))


def test_label_err_rejects_negative():
    labels, flux, ivar = _toy()
    bad = np.ones_like(labels)
    bad[0, 0] = -1.0
    with pytest.raises(ValueError):
        tc.CannonModel(labels, flux, ivar, _vec(),
                       training_set_label_err=bad)
