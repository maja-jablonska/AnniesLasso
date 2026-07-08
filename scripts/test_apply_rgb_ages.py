#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the RGB/HeB evolutionary-state classifier in
:mod:`scripts.apply_rgb_ages`, on synthetic data built so the label features
are uninformative while a molecular-band-like spectral feature separates the
classes -- the regime where spectral features must beat label-only ones.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import pandas as pd
import pytest

try:
    from scripts.apply_rgb_ages import (RGB, HEB, clean_flux,
                                        classifier_matrix,
                                        fit_state_classifier, seismic_state,
                                        select_state)
except ImportError:
    from apply_rgb_ages import (RGB, HEB, clean_flux, classifier_matrix,
                                fit_state_classifier, seismic_state,
                                select_state)


N_PIXELS = 240
BAND = slice(40, 60)


def make_dataset(n=1000, seed=0):
    """
    ``(table, normalized_flux, normalized_ivar, y, labeled)`` where the label
    columns are drawn identically for both classes (uninformative) and the
    RGB stars have a deeper synthetic CN-like band in the spectra. Half the
    stars carry a seismic ``EvoState``; the other half are unknown.
    """
    rng = np.random.RandomState(seed)
    y = np.where(rng.random_sample(n) < 0.5, RGB, HEB)

    table = pd.DataFrame({
        "raw_teff": rng.normal(4800.0, 120.0, n),
        "raw_logg": rng.normal(2.45, 0.12, n),
        "raw_fe_h": rng.normal(0.0, 0.2, n),
        "c_fe": rng.normal(0.0, 0.05, n),
        "n_fe": rng.normal(0.0, 0.05, n),
        "mg_fe": rng.normal(0.0, 0.05, n),
    })

    flux = 1.0 + rng.normal(0.0, 0.005, (n, N_PIXELS))
    flux[:, BAND] -= np.where(y == RGB, 0.12, 0.02)[:, None]
    ivar = np.full((n, N_PIXELS), 1.0e4)
    dead = rng.random_sample(flux.shape) < 0.01
    flux[dead] = np.nan
    ivar[dead] = 0.0

    labeled = rng.random_sample(n) < 0.5
    table["EvoState"] = np.where(labeled, y, np.nan)
    return table, flux, ivar, y, labeled


def test_seismic_state_reads_string_evolst():
    # The APOKASC prep notebook writes string 'RGB'/'RC' in EvolSt; anything
    # ambiguous (or missing) must stay NaN so those stars get classified.
    table = pd.DataFrame(
        {"EvolSt": ["RGB", "RC", " rc ", "HeB", "RC/RGB?", None]})
    state = seismic_state(table)
    assert state.tolist()[:4] == [RGB, HEB, HEB, HEB]
    assert state[4:].isna().all()


def test_seismic_state_prefers_numeric_evostate():
    table = pd.DataFrame({"EvoState": [1, 2, np.nan, 7],
                          "EvolSt": ["RC", "RGB", "RGB", "RGB"]})
    state = seismic_state(table)
    assert state.tolist()[:2] == [RGB, HEB]
    assert state[2:].isna().all()


def test_seismic_state_without_state_columns_is_all_nan():
    table = pd.DataFrame({"raw_teff": [4800.0, 4900.0]})
    assert seismic_state(table).isna().all()


def test_clean_flux_fills_bad_pixels_and_clips():
    flux = np.array([[1.0, np.nan, 5.0], [0.8, 1.2, -1.0]])
    ivar = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    out = clean_flux(flux, ivar)
    assert np.isfinite(out).all()
    assert out[0, 1] == 1.0          # NaN flux -> continuum
    assert out[1, 0] == 1.0          # zero ivar -> continuum
    assert out.min() >= 0.0 and out.max() <= 3.0
    # the input is not modified in place
    assert np.isnan(flux[0, 1])


def test_classifier_matrix_modes():
    table, flux, ivar, _, _ = make_dataset(n=50)
    n_labels = 6
    assert classifier_matrix(table, features="labels").shape == (50, n_labels)
    assert classifier_matrix(table, flux, ivar, "spectra").shape \
        == (50, N_PIXELS)
    assert classifier_matrix(table, flux, ivar, "both").shape \
        == (50, n_labels + N_PIXELS)
    with pytest.raises(ValueError):
        classifier_matrix(table, None, None, "spectra")
    with pytest.raises(ValueError):
        classifier_matrix(table, flux, ivar, "bogus")


def test_spectral_features_recover_rgb_where_labels_cannot():
    table, flux, ivar, y, labeled = make_dataset()
    unknown = ~labeled

    clf = fit_state_classifier(table, flux, ivar, features="both")
    assert clf is not None and clf.features_used_ == "both"
    is_rgb, source, proba = select_state(table, clf, 0.9, flux, ivar)

    # Seismic rows are taken at face value, never classified.
    assert (is_rgb[labeled] == (y[labeled] == RGB)).all()
    assert set(source[labeled]) == {"seismic"}
    assert np.isnan(proba[labeled]).all()

    # Spectral features confidently recover the unknown RGB stars...
    true_rgb = unknown & (y == RGB)
    completeness = is_rgb[true_rgb].mean()
    accepted = unknown & is_rgb
    purity = (y[accepted] == RGB).mean()
    assert completeness > 0.9
    assert purity > 0.95
    assert set(source[accepted]) == {"classified"}

    # ...whereas the label-only classifier is guessing on the same data:
    # boosted trees are overconfident on uninformative features, so its
    # failure mode is coin-flip contamination among the accepted stars.
    clf_labels = fit_state_classifier(table, features="labels")
    assert clf_labels.features_used_ == "labels"
    is_rgb_labels, _, _ = select_state(table, clf_labels, 0.9)
    accepted_labels = unknown & is_rgb_labels
    purity_labels = ((y[accepted_labels] == RGB).mean()
                     if accepted_labels.any() else 1.0)
    assert purity_labels < 0.7


def test_spectra_only_mode_works():
    table, flux, ivar, y, labeled = make_dataset(seed=1)
    clf = fit_state_classifier(table, flux, ivar, features="spectra")
    assert clf.features_used_ == "spectra"
    is_rgb, _, _ = select_state(table, clf, 0.9, flux, ivar)
    unknown = ~labeled
    assert is_rgb[unknown & (y == RGB)].mean() > 0.9


def test_missing_training_spectra_falls_back_to_labels():
    table, flux, ivar, _, _ = make_dataset(seed=2)
    clf = fit_state_classifier(table, features="both")
    assert clf is not None and clf.features_used_ == "labels"
    # And selection then needs no spectra either.
    is_rgb, _, _ = select_state(table, clf, 0.9)
    assert is_rgb.dtype == bool


def test_select_state_requires_spectra_for_spectral_classifier():
    table, flux, ivar, _, _ = make_dataset(seed=3)
    clf = fit_state_classifier(table, flux, ivar, features="both")
    with pytest.raises(ValueError):
        select_state(table, clf, 0.9)


def test_heb_target_selects_the_clump():
    table, flux, ivar, y, labeled = make_dataset(seed=4)
    clf = fit_state_classifier(table, flux, ivar, features="both",
                               target=HEB)
    is_heb, source, proba = select_state(table, clf, 0.9, flux, ivar,
                                         target=HEB)
    unknown = ~labeled

    # Seismic HeB rows are taken at face value; RGB rows are excluded.
    assert (is_heb[labeled] == (y[labeled] == HEB)).all()

    # The classifier recovers the unknown HeB stars, and what it accepts
    # is HeB, not RGB.
    assert is_heb[unknown & (y == HEB)].mean() > 0.9
    accepted = unknown & is_heb
    assert (y[accepted] == HEB).mean() > 0.95


def test_too_few_labeled_stars_returns_none():
    table, flux, ivar, _, _ = make_dataset(n=100)
    assert fit_state_classifier(table, flux, ivar, features="both") is None
