#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the high-level ``thecannon.apogee`` API.

These exercise the APOGEE-specific glue end to end without any external data:
write a synthetic ``apStar``-format FITS file, read it back, pseudo-continuum-
normalize it, train a small ``CannonModel`` on a synthetic reference set, and
fit a held-out star with ``apogee.fit``. The point is to lock in the I/O layout,
the wavelength solution, and the shape contracts of ``read_spectrum`` /
``normalize`` / ``process`` / ``fit``.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import pandas as pd
import pytest

import thecannon as tc
from thecannon import apogee

pytest.importorskip("astropy.io.fits")


def test_wavelength_grid_and_continuum_pixels():
    wl = apogee.wavelength_grid()
    assert wl.shape == (apogee.N_PIXELS,)
    # The standard APOGEE grid spans ~15100-17000 Angstrom and is increasing.
    assert 15090 < wl[0] < 15110
    assert 16990 < wl[-1] < 17010
    assert np.all(np.diff(wl) > 0)

    mask = apogee.continuum_pixel_mask(wl)
    assert mask.shape == wl.shape
    assert mask.dtype == bool
    assert 100 < mask.sum() < apogee.N_PIXELS


def test_read_spectrum_roundtrip(tmp_path):
    path = str(tmp_path / "apStar-demo.fits")
    apogee.make_example_spectrum(path, teff=4800, logg=2.5, fe_h=-0.2, seed=0)

    disp, flux, ivar = apogee.read_spectrum(path)
    assert disp.shape == flux.shape == ivar.shape == (apogee.N_PIXELS,)
    # Wavelength solution should match the canonical grid the writer used.
    np.testing.assert_allclose(disp, apogee.wavelength_grid(), rtol=1e-6)
    assert np.all(np.isfinite(flux))
    assert np.all(ivar >= 0)
    assert np.any(ivar > 0)


def test_normalize_preserves_dimensionality(tmp_path):
    path = str(tmp_path / "apStar-demo.fits")
    apogee.make_example_spectrum(path, seed=1)
    disp, flux, ivar = apogee.read_spectrum(path)

    nf, niv, cont, meta = apogee.normalize(disp, flux, ivar)
    # A single (1-D) spectrum in -> single (1-D) spectrum out.
    assert nf.shape == flux.shape
    assert niv.shape == ivar.shape
    assert cont.shape == flux.shape
    # Normalized continuum sits near unity.
    assert 0.8 < np.nanmedian(nf) < 1.2


def test_process_stacks_multiple(tmp_path):
    paths = []
    for k in range(3):
        p = str(tmp_path / f"apStar-{k}.fits")
        apogee.make_example_spectrum(p, teff=4500 + 100 * k, seed=k)
        paths.append(p)

    disp, flux, ivar = apogee.process(paths)
    assert disp.shape == (apogee.N_PIXELS,)
    assert flux.shape == (3, apogee.N_PIXELS)
    assert ivar.shape == (3, apogee.N_PIXELS)


def test_fit_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    n = 24
    ref = pd.DataFrame({
        "TEFF": rng.uniform(4200, 5200, n),
        "LOGG": rng.uniform(1.8, 3.2, n),
        "FE_H": rng.uniform(-0.8, 0.3, n),
    })
    ref_paths = []
    for k, row in ref.iterrows():
        p = str(tmp_path / f"ref-{k}.fits")
        apogee.make_example_spectrum(p, teff=row.TEFF, logg=row.LOGG,
                                     fe_h=row.FE_H, snr=200, seed=100 + k)
        ref_paths.append(p)

    disp, flux, ivar = apogee.process(ref_paths)
    vectorizer = tc.vectorizer.PolynomialVectorizer(("TEFF", "LOGG", "FE_H"), 2)
    model = tc.CannonModel(ref, flux, ivar, vectorizer=vectorizer,
                           dispersion=disp)
    model.train()
    assert model.is_trained

    star = str(tmp_path / "star.fits")
    apogee.make_example_spectrum(star, teff=4800, logg=2.5, fe_h=-0.2,
                                 snr=200, seed=999)
    labels, cov, meta, table = apogee.fit(model, star, as_table=True,
                                          progressbar=False)

    assert labels.shape == (1, 3)
    assert cov.shape == (1, 3, 3)
    assert list(table.columns) == ["TEFF", "LOGG", "FE_H", "r_chi_sq"]
    # Surface gravity and metallicity are well constrained by the synthetic
    # lines; assert a loose recovery so the test is not brittle.
    assert abs(float(table["LOGG"][0]) - 2.5) < 0.4
    assert abs(float(table["FE_H"][0]) - (-0.2)) < 0.3
    assert np.isfinite(meta[0]["r_chi_sq"])


def test_fit_requires_trained_model(tmp_path):
    star = str(tmp_path / "star.fits")
    apogee.make_example_spectrum(star, seed=2)

    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"TEFF": rng.uniform(4200, 5200, 6),
                        "LOGG": rng.uniform(1.8, 3.2, 6),
                        "FE_H": rng.uniform(-0.8, 0.3, 6)})
    disp, flux, ivar = apogee.process([star] * 6)
    vectorizer = tc.vectorizer.PolynomialVectorizer(("TEFF", "LOGG", "FE_H"), 2)
    model = tc.CannonModel(ref, flux, ivar, vectorizer=vectorizer,
                           dispersion=disp)
    with pytest.raises(ValueError):
        apogee.fit(model, star)
