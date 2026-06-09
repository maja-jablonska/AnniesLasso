#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Continuum-normalization.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["normalize", "sines_and_cosines"]

import logging
import numpy as np
import jax
import jax.numpy as jnp


@jax.jit
def _continuum_amplitudes(continuum_flux, continuum_ivar, M, region_matrix,
    scalar):
    """
    Solve the (eigenvalue-regularized) weighted normal equations for the sine-
    and-cosine amplitudes and evaluate the continuum, vectorized over stars.

    All stars in a region share the same design matrices ``M`` (at the continuum
    pixels) and ``region_matrix`` (at every region pixel), so the per-star solve
    is mapped with ``jax.vmap`` and the whole thing JIT-compiled. JIT recompiles
    once per distinct design-matrix shape (i.e. once per region).

    :param continuum_flux:
        Continuum-pixel fluxes, shape ``(n_stars, n_continuum_pixels)``.

    :param continuum_ivar:
        Inverse variances matching ``continuum_flux``.

    :param M:
        Continuum-pixel design matrix, shape ``(n_terms, n_continuum_pixels)``.

    :param region_matrix:
        Region design matrix, shape ``(n_terms, n_region_pixels)``.

    :param scalar:
        The magic eigenvalue-regularization scalar.

    :returns:
        A tuple of (continuum over the region pixels, amplitudes, condition
        number), each with a leading ``n_stars`` axis.
    """
    def _one(cfl, civ):
        MTM = jnp.dot(M, civ[:, None] * M.T)
        MTy = jnp.dot(M, civ * cfl)

        eigenvalues = jnp.linalg.eigvalsh(MTM)
        MTM = MTM + jnp.eye(MTM.shape[0]) * (scalar * jnp.max(eigenvalues))
        eigenvalues = jnp.linalg.eigvalsh(MTM)
        condition_number = jnp.max(eigenvalues) / jnp.min(eigenvalues)

        amplitudes = jnp.linalg.solve(MTM, MTy)
        return jnp.dot(region_matrix.T, amplitudes), amplitudes, condition_number

    return jax.vmap(_one)(continuum_flux, continuum_ivar)


def _continuum_design_matrix(dispersion, L, order):
    """
    Build a design matrix for the continuum determination, using sines and
    cosines.

    :param dispersion:
        An array of dispersion points.

    :param L:
        The length-scale for the sine and cosine functions.

    :param order:
        The number of sines and cosines to use in the fit.
    """

    L, dispersion = float(L), np.array(dispersion)
    scale = 2 * (np.pi / L)
    return np.vstack([
        np.ones_like(dispersion).reshape((1, -1)), 
        np.array([
            [np.cos(o * scale * dispersion), np.sin(o * scale * dispersion)] \
            for o in range(1, order + 1)]).reshape((2 * order, dispersion.size))
        ])


def sines_and_cosines(dispersion, flux, ivar, continuum_pixels, L=1400, order=3,
    regions=None, fill_value=1.0, progressbar=True, **kwargs):
    """
    Fit the flux values of pre-defined continuum pixels using a sum of sine and
    cosine functions.

    :param dispersion:
        The dispersion values.

    :param flux:
        The flux values for all pixels, as they correspond to the `dispersion`
        array.

    :param ivar:
        The inverse variances for all pixels, as they correspond to the
        `dispersion` array.

    :param continuum_pixels:
        A mask that selects pixels that should be considered as 'continuum'.

    :param L: [optional]
        The length scale for the sines and cosines.

    :param order: [optional]
        The number of sine/cosine functions to use in the fit.

    :param regions: [optional]
        Specify sections of the spectra that should be fitted separately in each
        star. This may be due to gaps between CCDs, or some other physically-
        motivated reason. These values should be specified in the same units as
        the `dispersion`, and should be given as a list of `[(start, end), ...]`
        values. For example, APOGEE spectra have gaps near the following
        wavelengths which could be used as `regions`:

        >> regions = ([15090, 15822], [15823, 16451], [16452, 16971])

    :param fill_value: [optional]
        The continuum value to use for when no continuum was calculated for that
        particular pixel (e.g., the pixel is outside of the `regions`).

    :param full_output: [optional]
        If set as True, then a metadata dictionary will also be returned.

    :returns:
        The continuum values for all pixels, and a dictionary that contains 
        metadata about the fit.
    """

    scalar = kwargs.pop("__magic_scalar", 1e-6) # MAGIC
    flux, ivar = np.atleast_2d(flux), np.atleast_2d(ivar)

    if regions is None:
        regions = [(dispersion[0], dispersion[-1])]

    region_masks = []
    region_matrices = []
    continuum_masks = []
    continuum_matrices = []
    pixel_included_in_regions = np.zeros_like(flux).astype(int)
    for start, end in regions:

        # Build the masks for this region.
        si, ei = np.searchsorted(dispersion, (start, end))
        region_mask = (end >= dispersion) * (dispersion >= start)
        region_masks.append(region_mask)
        pixel_included_in_regions[:, region_mask] += 1

        continuum_masks.append(continuum_pixels[
            (ei >= continuum_pixels) * (continuum_pixels >= si)])

        # Build the design matrices for this region.
        region_matrices.append(
            _continuum_design_matrix(dispersion[region_masks[-1]], L, order))
        continuum_matrices.append(
            _continuum_design_matrix(dispersion[continuum_masks[-1]], L, order))

        # TODO: ISSUE: Check for overlapping regions and raise an warning.

    # Check for non-zero pixels (e.g. ivar > 0) that are not included in a
    # region. We should warn about this very loudly!
    warn_on_pixels = (pixel_included_in_regions == 0) * (ivar > 0)
    if np.any(warn_on_pixels):
        n_affected = int(np.any(warn_on_pixels, axis=1).sum())
        logging.warn("Some pixels have measured flux values (e.g., ivar > 0) "
                     "but are not included in any specified continuum region. "
                     "These pixels won't be continuum-normalised ({0} spectra "
                     "affected).".format(n_affected))

    S = flux.shape[0]
    continuum = np.ones_like(flux) * fill_value
    metadata = [[] for _ in range(S)]

    # Each region is fit for every star at once: the design matrices are shared
    # across stars, so the per-star normal-equation solve is vmapped and JIT-
    # compiled (see `_continuum_amplitudes`). The host loop below only walks the
    # handful of regions, so a plain tqdm bar over regions is the right tool --
    # the jax-tqdm bars used in CannonModel.train/test only work inside JAX
    # loops, and here there is no per-star Python loop left to track.
    regions_iter = list(
        zip(region_masks, region_matrices, continuum_masks, continuum_matrices))
    if progressbar:
        try:
            from tqdm.auto import tqdm
            regions_iter = tqdm(regions_iter, desc="Normalizing", unit="region")
        except ImportError:
            pass

    for region_mask, region_matrix, continuum_mask, continuum_matrix \
    in regions_iter:
        if continuum_mask.size == 0:
            # No continuum pixels in this region; leave it at the fill value.
            for s in range(S):
                metadata[s].append([order, L, fill_value, scalar, [], None])
            continue

        # Solve for the amplitudes (linear algebra performed in JAX, vmapped
        # over all stars and JIT-compiled).
        region_continuum, amplitudes, condition_number = _continuum_amplitudes(
            jnp.asarray(flux[:, continuum_mask]),
            jnp.asarray(ivar[:, continuum_mask]),
            jnp.asarray(continuum_matrix),
            jnp.asarray(region_matrix),
            float(scalar))

        continuum[:, region_mask] = np.asarray(region_continuum)

        amplitudes = np.asarray(amplitudes)
        condition_number = np.asarray(condition_number)
        for s in range(S):
            metadata[s].append((order, L, fill_value, scalar,
                                amplitudes[s], float(condition_number[s])))

    return (continuum, metadata)
    

def normalize(dispersion, flux, ivar, continuum_pixels, L=1400, order=3,
    regions=None, fill_value=1.0, progressbar=True, **kwargs):
    """
    Pseudo-continuum-normalize the flux using a defined set of continuum pixels
    and a sum of sine and cosine functions.

    :param dispersion:
        The dispersion values.

    :param flux:
        The flux values for all pixels, as they correspond to the `dispersion`
        array.

    :param ivar:
        The inverse variances for all pixels, as they correspond to the
        `dispersion` array.

    :param continuum_pixels:
        A mask that selects pixels that should be considered as 'continuum'.

    :param L: [optional]
        The length scale for the sines and cosines.

    :param order: [optional]
        The number of sine/cosine functions to use in the fit.

    :param regions: [optional]
        Specify sections of the spectra that should be fitted separately in each
        star. This may be due to gaps between CCDs, or some other physically-
        motivated reason. These values should be specified in the same units as
        the `dispersion`, and should be given as a list of `[(start, end), ...]`
        values. For example, APOGEE spectra have gaps near the following
        wavelengths which could be used as `regions`:

        >> regions = ([15090, 15822], [15823, 16451], [16452, 16971])

    :param fill_value: [optional]
        The continuum value to use for when no continuum was calculated for that
        particular pixel (e.g., the pixel is outside of the `regions`).

    :param full_output: [optional]
        If set as True, then a metadata dictionary will also be returned.

    :returns:
        The continuum values for all pixels, and a dictionary that contains 
        metadata about the fit.
    """
    continuum, metadata = sines_and_cosines(dispersion, flux, ivar,
        continuum_pixels, L=L, order=order, regions=regions,
        fill_value=fill_value, progressbar=progressbar, **kwargs)

    normalized_flux = flux/continuum
    normalized_ivar = continuum * ivar * continuum
    normalized_flux = jnp.where(normalized_ivar == 0, 1.0, normalized_flux)

    non_finite_pixels = ~jnp.isfinite(normalized_flux)
    normalized_flux = jnp.where(non_finite_pixels, 1.0, normalized_flux)
    normalized_ivar = jnp.where(non_finite_pixels, 0.0, normalized_ivar)

    return (normalized_flux, normalized_ivar, continuum, metadata)


