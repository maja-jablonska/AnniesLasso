#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A high-level API for running The Cannon on APOGEE spectra.

The rest of ``thecannon`` is deliberately data-agnostic: :class:`CannonModel`
takes flux/ivar arrays and a label table, and :mod:`thecannon.continuum` knows
how to pseudo-continuum-normalize *any* spectrum given a set of continuum
pixels. This module supplies the APOGEE-specific glue that sits in front of
those primitives, namely:

* the standard 8575-pixel APOGEE ``apStar`` log-linear wavelength grid;
* the blue/green/red detector regions used to normalize each chip separately;
* the canonical set of continuum pixels (shipped as package data);
* readers for ``apStar`` and ``aspcapStar`` FITS files;
* a one-call :func:`process` step that turns a list of files into the
  ``(dispersion, normalized_flux, normalized_ivar)`` arrays the model wants;
* :func:`fit`, which reads, normalizes and runs the test step against a trained
  model in a single call; and
* :func:`make_example_spectrum`, which writes a synthetic ``apStar``-format file
  so the pipeline can be exercised without downloading data.

Typical usage::

    import thecannon as tc
    from thecannon import apogee

    model = tc.CannonModel.read("apogee_dr17.model")
    labels, cov, meta = apogee.fit(model, ["aspcapStar-r12-2M00....fits"])

Nothing here imports anything heavy at module load beyond numpy; ``astropy`` is
imported lazily inside the FITS readers so the rest of the package keeps working
in environments without it.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from . import continuum

logger = logging.getLogger(__name__)

__all__ = [
    "N_PIXELS",
    "DEFAULT_REGIONS",
    "wavelength_grid",
    "continuum_pixel_indices",
    "continuum_pixel_mask",
    "read_spectrum",
    "read_spectra",
    "normalize",
    "process",
    "fit",
    "make_example_spectrum",
]


# -----------------------------------------------------------------------------
# The APOGEE wavelength solution.
#
# Combined ``apStar`` / ``aspcapStar`` spectra are sampled on a fixed log-linear
# grid of 8575 pixels. The FITS WCS keywords encode it as
#     log10(lambda / Angstrom) = CRVAL1 + CDELT1 * (pixel - (CRPIX1 - 1))
# with CRVAL1 = 4.179, CDELT1 = 6e-6, CRPIX1 = 1. We keep these as the default
# and still honour whatever is in a file's header when reading.
# -----------------------------------------------------------------------------

N_PIXELS = 8575
_LOG10_WL_0 = 4.179
_LOG10_WL_STEP = 6e-6

# Blue, green and red detector spans (Angstrom). The Cannon normalizes each
# chip independently because the blaze function and inter-chip gaps make a
# single global continuum fit a poor description of the data.
DEFAULT_REGIONS = (
    (15090.0, 15822.0),
    (15823.0, 16451.0),
    (16452.0, 16971.0),
)

_CONTINUUM_PIXELS_FILE = os.path.join(
    os.path.dirname(__file__), "data", "apogee_continuum_pixels.list")


def wavelength_grid(n_pixels=N_PIXELS, log10_wl_0=_LOG10_WL_0,
    log10_wl_step=_LOG10_WL_STEP):
    """
    Return the standard APOGEE log-linear dispersion array (in Angstrom).

    :param n_pixels: [optional]
        The number of pixels in the grid. Defaults to the combined-spectrum
        length of 8575.

    :param log10_wl_0: [optional]
        ``log10`` of the wavelength at the first pixel (the ``CRVAL1`` keyword).

    :param log10_wl_step: [optional]
        The ``log10`` wavelength step per pixel (the ``CDELT1`` keyword).

    :returns:
        A ``(n_pixels, )`` array of wavelengths in Angstrom.
    """
    return 10.0 ** (log10_wl_0 + log10_wl_step * np.arange(n_pixels))


def _wavelength_from_header(header, n_pixels):
    """
    Build a dispersion array from FITS WCS keywords, falling back to the
    canonical APOGEE grid when the keywords are missing.
    """
    try:
        crval1 = float(header["CRVAL1"])
        cdelt1 = float(header.get("CDELT1", header.get("CD1_1")))
    except (KeyError, TypeError, ValueError):
        logger.debug("No usable WCS in header; using the default APOGEE grid.")
        return wavelength_grid(n_pixels)

    crpix1 = float(header.get("CRPIX1", 1.0))
    pixels = np.arange(n_pixels) - (crpix1 - 1.0)
    log_wl = crval1 + cdelt1 * pixels

    ctype1 = str(header.get("CTYPE1", "LOG-LINEAR")).upper()
    dcflag = int(header.get("DC-FLAG", 1))
    if "LOG" in ctype1 or dcflag == 1:
        return 10.0 ** log_wl
    # Linear dispersion (rare for combined APOGEE products, but be permissive).
    return log_wl


def continuum_pixel_indices(path=None):
    """
    Return the integer pixel indices flagged as continuum.

    These are the APOGEE pixels that ASPCAP/The Cannon treat as line-free enough
    to anchor the pseudo-continuum. They index into the 8575-pixel grid.

    :param path: [optional]
        Read the indices from this file instead of the packaged list. The file
        is parsed with ``numpy.loadtxt`` and may contain ``#`` comments.
    """
    path = path or _CONTINUUM_PIXELS_FILE
    indices = np.loadtxt(path, dtype=int, comments="#")
    return np.atleast_1d(indices)


def continuum_pixel_mask(dispersion=None, path=None):
    """
    Return a boolean continuum mask aligned to ``dispersion``.

    :param dispersion: [optional]
        The dispersion array the mask should match. Defaults to the standard
        APOGEE grid, giving a length-8575 mask.

    :param path: [optional]
        An alternative continuum-pixel index file (see
        :func:`continuum_pixel_indices`).

    :returns:
        A boolean array, ``True`` at continuum pixels.
    """
    if dispersion is None:
        n_pixels = N_PIXELS
    else:
        n_pixels = len(dispersion)

    indices = continuum_pixel_indices(path)
    indices = indices[(indices >= 0) & (indices < n_pixels)]

    mask = np.zeros(n_pixels, dtype=bool)
    mask[indices] = True
    return mask


# -----------------------------------------------------------------------------
# Reading APOGEE FITS spectra.
# -----------------------------------------------------------------------------

def _coerce_1d(array, row):
    """Pick a single 1-D spectrum out of a possibly-2D HDU."""
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        # ``apStar`` stacks combined spectra and individual visits along axis 0;
        # row 0 is the pixel-weighted combination. ``aspcapStar`` is already
        # 1-D. Clip the requested row into range to stay robust.
        return array[min(row, array.shape[0] - 1)]
    raise ValueError(
        "expected a 1-D or 2-D flux array, got shape {}".format(array.shape))


def read_spectrum(path, row=0, hdu_flux=1, hdu_error=2):
    """
    Read a single APOGEE ``apStar`` or ``aspcapStar`` FITS file.

    The two products share a layout: HDU1 holds flux and HDU2 holds the flux
    uncertainty (1-sigma), both sampled on the log-linear wavelength grid whose
    WCS lives in the HDU1 header. ``apStar`` flux is in physical units and is
    *not* continuum-normalized (feed it to :func:`normalize`), whereas
    ``aspcapStar`` flux is already pseudo-continuum-normalized.

    :param path:
        Path to the FITS file.

    :param row: [optional]
        For ``apStar`` files whose HDUs stack several spectra, the row to read.
        Row 0 (the default) is the pixel-weighted combined spectrum.

    :param hdu_flux: [optional]
        Index of the flux HDU. Default 1.

    :param hdu_error: [optional]
        Index of the uncertainty (1-sigma) HDU. Default 2.

    :returns:
        A 3-tuple ``(dispersion, flux, ivar)`` of length-``N_PIXELS`` arrays.
        Pixels with a non-positive or non-finite uncertainty get ``ivar = 0``.
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        flux = _coerce_1d(hdul[hdu_flux].data, row)
        error = _coerce_1d(hdul[hdu_error].data, row)
        header = hdul[hdu_flux].header
        dispersion = _wavelength_from_header(header, flux.size)

    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ivar = 1.0 / (error ** 2)

    bad = ~np.isfinite(ivar) | ~np.isfinite(flux) | (error <= 0)
    flux = np.where(bad, 0.0, flux)
    ivar = np.where(bad, 0.0, ivar)

    return dispersion, flux, ivar


def read_spectra(paths, row=0, hdu_flux=1, hdu_error=2):
    """
    Read several APOGEE files and stack them.

    All files are assumed to share the standard APOGEE wavelength grid (this is
    true for combined ``apStar``/``aspcapStar`` products). The dispersion of the
    first file is returned.

    :param paths:
        An iterable of FITS file paths.

    :returns:
        ``(dispersion, flux, ivar)`` where ``flux`` and ``ivar`` are
        ``(n_stars, n_pixels)`` arrays.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("no spectrum paths were given")

    dispersion = None
    flux_rows, ivar_rows = [], []
    for path in paths:
        d, f, i = read_spectrum(path, row=row, hdu_flux=hdu_flux,
            hdu_error=hdu_error)
        if dispersion is None:
            dispersion = d
        elif f.size != dispersion.size:
            raise ValueError(
                "{} has {} pixels but the first spectrum has {}".format(
                    path, f.size, dispersion.size))
        flux_rows.append(f)
        ivar_rows.append(i)

    return dispersion, np.vstack(flux_rows), np.vstack(ivar_rows)


# -----------------------------------------------------------------------------
# Processing and fitting.
# -----------------------------------------------------------------------------

def normalize(dispersion, flux, ivar, continuum_pixels=None,
    regions=DEFAULT_REGIONS, continuum_pixels_path=None, **kwargs):
    """
    Pseudo-continuum-normalize APOGEE spectra with sensible APOGEE defaults.

    This is a thin wrapper around :func:`thecannon.continuum.normalize` that
    fills in the APOGEE continuum pixels and per-chip ``regions`` so callers do
    not have to remember them.

    :param dispersion:
        The ``(n_pixels, )`` dispersion array.

    :param flux:
        Flux values, ``(n_pixels, )`` or ``(n_stars, n_pixels)``.

    :param ivar:
        Inverse variances, the same shape as ``flux``.

    :param continuum_pixels: [optional]
        A boolean mask or integer indices selecting continuum pixels. Defaults
        to the packaged APOGEE continuum-pixel list aligned to ``dispersion``.

    :param regions: [optional]
        ``[(start, end), ...]`` wavelength spans fitted independently. Defaults
        to the three APOGEE detector regions.

    :param continuum_pixels_path: [optional]
        Override the packaged continuum-pixel file.

    :returns:
        ``(normalized_flux, normalized_ivar, continuum, metadata)`` exactly as
        :func:`thecannon.continuum.normalize` returns them.
    """
    if continuum_pixels is None:
        continuum_pixels = continuum_pixel_mask(dispersion,
            path=continuum_pixels_path)
    continuum_pixels = np.asarray(continuum_pixels)
    if continuum_pixels.dtype == bool:
        continuum_pixels = np.where(continuum_pixels)[0]

    was_1d = np.ndim(flux) == 1
    norm_flux, norm_ivar, cont, metadata = continuum.normalize(
        dispersion, flux, ivar, continuum_pixels, regions=regions, **kwargs)

    if was_1d:
        # ``continuum.normalize`` works in 2-D internally; collapse the star
        # axis back so a single spectrum in yields a single spectrum out.
        norm_flux = np.asarray(norm_flux)[0]
        norm_ivar = np.asarray(norm_ivar)[0]
        cont = np.asarray(cont)[0]

    return norm_flux, norm_ivar, cont, metadata


def process(paths, row=0, continuum_pixels=None, regions=DEFAULT_REGIONS,
    continuum_pixels_path=None, **kwargs):
    """
    Read APOGEE files and pseudo-continuum-normalize them in one call.

    :param paths:
        A single FITS path or an iterable of paths.

    :returns:
        ``(dispersion, normalized_flux, normalized_ivar)`` ready to hand to
        :meth:`thecannon.CannonModel.test`. ``normalized_flux`` and
        ``normalized_ivar`` are 2-D, ``(n_stars, n_pixels)``.
    """
    if isinstance(paths, str):
        paths = [paths]

    dispersion, flux, ivar = read_spectra(paths, row=row)
    norm_flux, norm_ivar, _continuum, _meta = normalize(
        dispersion, flux, ivar, continuum_pixels=continuum_pixels,
        regions=regions, continuum_pixels_path=continuum_pixels_path, **kwargs)
    return dispersion, np.atleast_2d(norm_flux), np.atleast_2d(norm_ivar)


def fit(model, paths, row=0, initial_labels=None, continuum_pixels=None,
    regions=DEFAULT_REGIONS, continuum_pixels_path=None, normalize_kwds=None,
    as_table=False, **test_kwds):
    """
    Read, normalize and fit APOGEE spectra against a trained Cannon model.

    This is the end-to-end convenience entry point: it runs :func:`process` and
    then the model's test step, returning the optimized labels.

    :param model:
        A trained :class:`thecannon.CannonModel`.

    :param paths:
        A single FITS path or an iterable of paths.

    :param initial_labels: [optional]
        Initial label guesses passed through to
        :meth:`thecannon.CannonModel.test`. Defaults to the model fiducials.

    :param as_table: [optional]
        If ``True``, also return a ``pandas.DataFrame`` of the labels keyed by
        the model's label names, with a ``r_chi_sq`` column appended (requires
        pandas).

    Remaining keyword arguments are forwarded to
    :meth:`thecannon.CannonModel.test`.

    :returns:
        ``(labels, cov, meta)`` from the test step, or
        ``(labels, cov, meta, table)`` when ``as_table=True``.
    """
    if not getattr(model, "is_trained", False):
        raise ValueError("the model must be trained before fitting spectra")

    normalize_kwds = normalize_kwds or {}
    _dispersion, norm_flux, norm_ivar = process(
        paths, row=row, continuum_pixels=continuum_pixels, regions=regions,
        continuum_pixels_path=continuum_pixels_path, **normalize_kwds)

    labels, cov, meta = model.test(norm_flux, norm_ivar,
        initial_labels=initial_labels, **test_kwds)

    if as_table:
        import pandas as pd
        names = list(model.vectorizer.label_names)
        table = pd.DataFrame(np.asarray(labels), columns=names)
        table["r_chi_sq"] = [m["r_chi_sq"] for m in meta]
        return labels, cov, meta, table

    return labels, cov, meta


# -----------------------------------------------------------------------------
# Synthetic data, so the pipeline can be demonstrated/tested offline.
# -----------------------------------------------------------------------------

def make_example_spectrum(path, teff=4800.0, logg=2.5, fe_h=-0.2, snr=120.0,
    seed=0, overwrite=True):
    """
    Write a synthetic ``apStar``-format FITS file for demos and tests.

    The flux is *not* a physical model; it is a smooth pseudo-continuum (a slow
    blaze-like curve) imprinted with a handful of Gaussian absorption lines
    whose depths scale loosely with the labels, plus photon noise set by
    ``snr``. The point is to produce a file with the correct HDU layout and WCS
    so that :func:`read_spectrum`, :func:`normalize` and the model test step can
    be run without downloading real data.

    :param path:
        Where to write the FITS file.

    :param teff, logg, fe_h: [optional]
        Labels that modulate the synthetic line depths.

    :param snr: [optional]
        Approximate per-pixel signal-to-noise ratio.

    :param seed: [optional]
        Seed for the additive noise so output is reproducible.

    :returns:
        The ``path`` that was written.
    """
    from astropy.io import fits

    rng = np.random.default_rng(seed)
    dispersion = wavelength_grid()
    n = dispersion.size

    # A slow, blaze-like pseudo-continuum (what apStar flux looks like before
    # normalization): a low-order curve scaled to a plausible flux level.
    x = np.linspace(-1.0, 1.0, n)
    continuum_shape = 1.0 - 0.25 * x ** 2 + 0.05 * x
    amplitude = 1.0e-16 * 10 ** (-0.4 * (logg - 2.5))
    continuum_flux = amplitude * continuum_shape

    # A few absorption lines; deeper for cooler / more metal-rich stars.
    line_centers = np.array(
        [15200, 15330, 15770, 16150, 16400, 16720, 16810.0])
    metal_depth = 0.35 * (1.0 + fe_h) * (4800.0 / teff)
    depths = metal_depth * np.array(
        [0.9, 0.6, 1.0, 0.7, 0.5, 0.8, 0.6])
    sigma = 1.2  # Angstrom

    absorption = np.ones(n)
    for center, depth in zip(line_centers, depths):
        absorption -= depth * np.exp(
            -0.5 * ((dispersion - center) / sigma) ** 2)
    absorption = np.clip(absorption, 0.02, None)

    clean = continuum_flux * absorption
    noise_sigma = clean / float(snr)
    flux = clean + rng.normal(scale=noise_sigma)
    error = np.where(noise_sigma > 0, noise_sigma, np.nanmedian(noise_sigma))

    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "synthetic"
    primary.header["SNR"] = float(snr)
    primary.header["TEFF"] = float(teff)
    primary.header["LOGG"] = float(logg)
    primary.header["FE_H"] = float(fe_h)

    flux_hdu = fits.ImageHDU(flux.astype(np.float32), name="FLUX")
    for hdu in (flux_hdu,):
        hdu.header["CRVAL1"] = _LOG10_WL_0
        hdu.header["CDELT1"] = _LOG10_WL_STEP
        hdu.header["CRPIX1"] = 1.0
        hdu.header["CTYPE1"] = "LOG-LINEAR"
        hdu.header["DC-FLAG"] = 1
    error_hdu = fits.ImageHDU(error.astype(np.float32), name="ERROR")

    fits.HDUList([primary, flux_hdu, error_hdu]).writeto(
        path, overwrite=overwrite)
    return path
