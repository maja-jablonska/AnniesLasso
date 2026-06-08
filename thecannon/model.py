#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["CannonModel"]

import logging
import multiprocessing as mp
import numpy as np
import jax
import jax.numpy as jnp
import os
import pickle
from datetime import datetime
from functools import wraps
from sys import version_info
from scipy.spatial import Delaunay

from .vectorizer.base import BaseVectorizer
from . import (censoring, fitting, utils, vectorizer as vectorizer_module, __version__)


logger = logging.getLogger(__name__)


def requires_training(method):
    """
    A decorator for model methods that require training before being run.

    :param method:
        A method belonging to CannonModel.
    """
    @wraps(method)
    def wrapper(model, *args, **kwargs):
        if not model.is_trained:
            raise TypeError("the model requires training first")
        return method(model, *args, **kwargs)
    return wrapper


class CannonModel(object):
    """
    A model for The Cannon which includes L1 regularization and pixel censoring.

    :param training_set_labels:
        A set of objects with labels known to high fidelity. This can be 
        given as a numpy structured array, or an astropy table.

    :param training_set_flux:
        An array of normalised fluxes for stars in the labelled set, given 
        as shape `(num_stars, num_pixels)`. The `num_stars` should match the
        number of rows in `training_set_labels`.

    :param training_set_ivar:
        An array of inverse variances on the normalized fluxes for stars in 
        the training set. The shape of the `training_set_ivar` array should
        match that of `training_set_flux`.

    :param vectorizer:
        A vectorizer to take input labels and produce a design matrix. This
        should be a sub-class of `vectorizer.BaseVectorizer`.

    :param dispersion: [optional]
        The dispersion values corresponding to the given pixels. If provided, 
        this should have a size of `num_pixels`.
    
    :param regularization: [optional]
        The strength of the L1 regularization. This should either be `None`,
        a float-type value for single regularization strength for all pixels,
        or a float-like array of length `num_pixels`.

    :param censors: [optional]
        A dictionary containing label names as keys and boolean censoring
        masks as values.
    """

    _data_attributes = \
        ("training_set_labels", "training_set_flux", "training_set_ivar")

    # Descriptive attributes are needed to train *and* test the model.
    _descriptive_attributes = \
        ("vectorizer", "censors", "regularization", "dispersion")

    # Trained attributes are set only at training time.
    _trained_attributes = ("theta", "s2")
    
    def __init__(self, training_set_labels, training_set_flux, training_set_ivar,
        vectorizer, dispersion=None, regularization=None, censors=None, **kwargs):

        # Save the vectorizer.
        if not isinstance(vectorizer, BaseVectorizer):
            raise TypeError(
                "vectorizer must be a sub-class of vectorizer.BaseVectorizer")
        
        self._vectorizer = vectorizer
        
        if training_set_flux is None and training_set_ivar is None:

            # Must be reading in a model that does not have the training set
            # spectra saved.
            self._training_set_flux = None
            self._training_set_ivar = None
            self._training_set_labels = training_set_labels

        else:
            self._training_set_flux = np.atleast_2d(training_set_flux)
            self._training_set_ivar = np.atleast_2d(training_set_ivar)
            
            if isinstance(training_set_labels, np.ndarray) \
            and training_set_labels.shape[0] == self._training_set_flux.shape[0] \
            and training_set_labels.shape[1] == len(vectorizer.label_names):
                # A valid array was given as the training set labels, not a table.
                self._training_set_labels = training_set_labels
            else: 
                self._training_set_labels = np.array(
                    [training_set_labels[ln] for ln in vectorizer.label_names]).T
            
            # Check that the flux and ivar are valid.
            self._verify_training_data(**kwargs)

        # Set regularization, censoring, dispersion.
        self.regularization = regularization
        self.censors = censors
        self.dispersion = dispersion

        # Set useful private attributes.
        __scale_labels_function = kwargs.get("__scale_labels_function", 
            lambda l: np.ptp(np.percentile(l, [2.5, 97.5], axis=0), axis=0))
        __fiducial_labels_function = kwargs.get("__fiducial_labels_function",
            lambda l: np.percentile(l, 50, axis=0))

        self._scales = __scale_labels_function(self.training_set_labels)
        self._fiducials = __fiducial_labels_function(self.training_set_labels)
        self._design_matrix = vectorizer(
            (self.training_set_labels - self._fiducials)/self._scales).T

        self.reset()

        return None


    # Representations.


    def __str__(self):
        return "<{module}.{name} of {K} labels {trained}with a training set "\
               "of {N} stars each with {M} pixels>".format(
                    module=self.__module__,
                    name=type(self).__name__,
                    trained="trained " if self.is_trained else "",
                    K=self.training_set_labels.shape[1],
                    N=self.training_set_labels.shape[0], 
                    M=self.training_set_flux.shape[1])


    def __repr__(self):
        return "<{0}.{1} object at {2}>".format(self.__module__, 
            type(self).__name__, hex(id(self)))


    # Model attributes that cannot (well, should not) be changed.


    @property
    def training_set_labels(self):
        """ Return the labels in the training set. """
        return self._training_set_labels


    @property
    def training_set_flux(self):
        """ Return the training set fluxes. """
        return self._training_set_flux


    @property
    def training_set_ivar(self):
        """ Return the inverse variances of the training set fluxes. """
        return self._training_set_ivar


    @property
    def vectorizer(self):
        """ Return the vectorizer for this model. """
        return self._vectorizer


    @property
    def design_matrix(self):
        """ Return the design matrix for this model. """
        return self._design_matrix


    def _censored_design_matrix(self, pixel_index, fill_value=np.nan):
        """
        Return a censored design matrix for the given pixel index, and a mask of
        which theta values to ignore when fitting.
    
        :param pixel_index:
            The zero-indexed pixel.

        :returns:
            A two-length tuple containing the censored design mask for this
            pixel, and a boolean mask of values to exclude when fitting for
            the spectral derivatives.
        """

        if not self.censors or self.censors is None \
        or len(set(self.censors).intersection(self.vectorizer.label_names)) == 0:
            return self.design_matrix

        data = (self.training_set_labels.copy() - self._fiducials)/self._scales
        for i, label_name in enumerate(self.vectorizer.label_names):
            try:
                use = self.censors[label_name][pixel_index]

            except KeyError:
                continue

            if not use:
                data[:, i] = fill_value

        return self.vectorizer(data).T


    @property
    def theta(self):
        """ Return the theta coefficients (spectral model derivatives). """
        return self._theta


    @property
    def s2(self):
        """ Return the intrinsic variance (s^2) for all pixels. """
        return self._s2


    # Model attributes that can be changed after initiation.


    @property
    def censors(self):
        """ Return the wavelength censor masks for the labels. """
        return self._censors


    @censors.setter
    def censors(self, censors):
        """
        Set label censoring masks for each pixel.

        :param censors:
            A dictionary-like object with label names as keys, and boolean arrays
            as values.
        """

        censors = {} if censors is None else censors
        if isinstance(censors, censoring.Censors):
            # Could be a censoring dictionary from a different model,
            # with different label names and pixels.
            
            # But more likely: we are loading a model from disk.
            self._censors = censors

        elif isinstance(censors, dict):
            self._censors = censoring.Censors(
                self.vectorizer.label_names, self.training_set_flux.shape[1],
                censors)

        else:
            raise TypeError(
                "censors must be a dictionary or a censoring.Censors object")


    @property
    def dispersion(self):
        """ Return the dispersion points for all pixels. """
        return self._dispersion


    @dispersion.setter
    def dispersion(self, dispersion):
        """
        Set the dispersion values for all the pixels.

        :param dispersion:
            An array of the dispersion values.
        """
        if dispersion is None:
            self._dispersion = None
            return None

        dispersion = np.array(dispersion).flatten()
        if self.training_set_flux is not None \
        and dispersion.size != self.training_set_flux.shape[1]:
            raise ValueError("dispersion provided does not match the number "
                             "of pixels per star ({0} != {1})".format(
                                dispersion.size, self.training_set_flux.shape[1]))

        if dispersion.dtype.kind not in "iuf":
            raise ValueError("dispersion values are not float-like")

        if not np.all(np.isfinite(dispersion)):
            raise ValueError("dispersion values must be finite")

        self._dispersion = dispersion
        return None


    @property
    def regularization(self):
        """ Return the strength of the L1 regularization for this model. """
        return self._regularization


    @regularization.setter
    def regularization(self, regularization):
        """
        Specify the strength of the regularization for the model, either as a
        single value for all pixels, or a different strength for each pixel.

        :param regularization:
            The L1-regularization strength for the model.
        """

        if regularization is None:
            self._regularization = None
            return None

        regularization = np.array(regularization).flatten()
        if regularization.size == 1:
            regularization = regularization[0]
            if 0 > regularization or not np.isfinite(regularization):
                raise ValueError("regularization must be positive and finite")

        elif regularization.size != self.training_set_flux.shape[1]:
            raise ValueError("regularization array must be of size `num_pixels`")

            if any(0 > regularization) \
            or not np.all(np.isfinite(regularization)):
                raise ValueError("regularization must be positive and finite")

        self._regularization = regularization
        return None


    # Convenient functions and properties.


    @property
    def is_trained(self):
        """ Return true or false for whether the model is trained. """
        return all(getattr(self, attr, None) is not None \
            for attr in self._trained_attributes)


    def reset(self):
        """ Clear any attributes that have been trained. """
        for attribute in self._trained_attributes:
            setattr(self, "_{}".format(attribute), None)
        return None


    def _pixel_access(self, array, index, default=None):
        """
        Safely access a (potentially per-pixel) attribute of the model.
        
        :param array:
            Either `None`, a float value, or an array the size of the dispersion
            array.

        :param index:
            The zero-indexed pixel to attempt to access.

        :param default: [optional]
            The default value to return if `array` is None.
        """

        if array is None:
            return default
        try:
            return array[index]
        except (IndexError, TypeError):
            return array


    def _verify_training_data(self, rho_warning=0.90):
        """
        Verify the training data for the appropriate shape and content.

        :param rho_warning: [optional]
            Maximum correlation value between labels before a warning is given.
        """

        if self.training_set_flux.shape != self.training_set_ivar.shape:
            raise ValueError("the training set flux and inverse variance arrays"
                             " for the labelled set must have the same shape")

        if len(self.training_set_labels) != self.training_set_flux.shape[0]:
            raise ValueError(
                "the first axes of the training set flux array should "
                "have the same shape as the nuber of rows in the labelled set"
                "(N_stars, N_pixels)")

        if not np.all(np.isfinite(self.training_set_labels)):
            raise ValueError("training set labels are not all finite")

        if not np.all(np.isfinite(self.training_set_flux)):
            raise ValueError("training set fluxes are not all finite")

        if not np.all(self.training_set_ivar >= 0) \
        or not np.all(np.isfinite(self.training_set_ivar)):
            raise ValueError("training set ivars are not all positive finite")

        # Look for very high correlation coefficients between labels, which
        # could make the training time very difficult.
        rho = np.corrcoef(self.training_set_labels.T)

        # Set the diagonal indices to zero.
        K = rho.shape[0]
        rho[np.diag_indices(K)] = 0.0
        indices = np.argsort(rho.flatten())[::-1]

        for index in indices:
            x, y = (index % K, int(index / K)) 
            rho_xy = rho[x, y]
            if rho_xy >= rho_warning: 
                if x > y: # One warning per correlated label pair.
                    logger.warn("Labels '{X}' and '{Y}' are highly correlated ("\
                        "rho = {rho_xy:.2}). This may cause very slow training "\
                        "times. Are both labels needed?".format(
                            X=self.vectorizer.label_names[x],
                            Y=self.vectorizer.label_names[y],
                            rho_xy=rho_xy))
            else:
                break
        return None


    def in_convex_hull(self, labels):
        """
        Return whether the provided labels are inside a complex hull constructed
        from the labelled set.

        :param labels:
            A `NxK` array of `N` sets of `K` labels, where `K` is the number of
            labels that make up the vectorizer.

        :returns:
            A boolean array as to whether the points are in the complex hull of
            the labelled set.
        """

        labels = np.atleast_2d(labels)
        if labels.shape[1] != self.training_set_labels.shape[1]:
            raise ValueError("expected {} labels; got {}".format(
                self.training_set_labels.shape[1], labels.shape[1]))

        hull = Delaunay(self.training_set_labels)
        return hull.find_simplex(labels) >= 0


    def write(self, path, include_training_set_spectra=False, overwrite=False,
        protocol=-1):
        """
        Serialise the trained model and save it to disk. This will save all
        relevant training attributes, and optionally, the training data.

        :param path:
            The path to save the model to.

        :param include_training_set_spectra: [optional]
            Save the labelled set, normalised flux and inverse variance used to
            train the model.

        :param overwrite: [optional]
            Overwrite the existing file path, if it already exists.

        :param protocol: [optional]
            The Python pickling protocol to employ. Use 2 for compatibility with
            previous Python releases, -1 for performance.
        """

        if os.path.exists(path) and not overwrite:
            raise IOError("path already exists: {0}".format(path))

        attributes = list(self._descriptive_attributes) \
                   + list(self._trained_attributes) \
                   + list(self._data_attributes)

        if "metadata" in attributes:
            logger.warn("'metadata' is a protected attribute. Ignoring.")
            attributes.remote("metadata")

        # Store up all the trained attributes and a hash of the training set.
        # Only the vectorizer and censors are serialized via their custom
        # `__getstate__` (the vectorizer to a (name, kwds) tuple and the censors
        # to a dict); everything else (numpy arrays, scalars) is stored as-is.
        # NB: numpy>=2.0 defines `ndarray.__getstate__` (returning None), so the
        # historical broad ``value.__getstate__()`` call would corrupt arrays.
        state = {}
        for attribute in attributes:

            value = getattr(self, attribute)

            if attribute in ("vectorizer", "censors") and value is not None \
            and hasattr(value, "__getstate__"):
                value = value.__getstate__()

            state[attribute] = value

        # Create a metadata dictionary.
        state["metadata"] = dict(
            version=__version__,
            model_class=type(self).__name__,
            modified=str(datetime.now()),
            data_attributes=self._data_attributes,
            descriptive_attributes=self._descriptive_attributes,
            trained_attributes=self._trained_attributes,
            training_set_hash=utils.short_hash(
                getattr(self, attr) for attr in self._data_attributes),
        )

        if not include_training_set_spectra:
            state.pop("training_set_flux")
            state.pop("training_set_ivar")

        elif not self.is_trained:
            logger.warn("The training set spectra won't be saved, and this model"\
                        "is not already trained. The saved model will not be "\
                        "able to be trained when loaded!")

        with open(path, "wb") as fp:
            pickle.dump(state, fp, protocol) 
        return None


    @classmethod
    def read(cls, path, **kwargs):
        """
        Read a saved model from disk.

        :param path:
            The path where to load the model from.
        """

        encodings = ("utf-8", "latin-1")
        for encoding in encodings:
            kwds = {"encoding": encoding} if version_info[0] >= 3 else {}
            try:
                with open(path, "rb") as fp:        
                    state = pickle.load(fp, **kwds)

            except UnicodeDecodeError:
                if encoding == encodings:
                    raise

        # Parse the state.
        metadata = state.get("metadata", {})
        version_saved = metadata.get("version", "0.1.0")
        if version_saved >= "0.2.0": # Refactor'd.

            init_attributes = list(metadata["data_attributes"]) \
                            + list(metadata["descriptive_attributes"])

            kwds = dict([(a, state.get(a, None)) for a in init_attributes])

            # Initiate the vectorizer.
            vectorizer_class, vectorizer_kwds = kwds["vectorizer"]
            klass = getattr(vectorizer_module, vectorizer_class)
            kwds["vectorizer"] = klass(**vectorizer_kwds)

            # Initiate the censors.
            kwds["censors"] = censoring.Censors(**kwds["censors"])

            model = cls(**kwds)

            # Set training attributes.
            for attr in metadata["trained_attributes"]:
                setattr(model, "_{}".format(attr), state.get(attr, None))

            return model
            
        else:
            raise NotImplementedError(
                "Cannot auto-convert old model files yet; "
                "contact Andy Casey <andrew.casey@monash.edu> if you need this")


    def train(self, threads=None, op_method=None, op_strict=True, op_kwds=None,
        progressbar=True, batch_size=None, **kwargs):
        """
        Train the model.

        :param threads: [optional]
            The number of parallel threads to use.

        :param op_method: [optional]
            The optimization algorithm to use: l_bfgs_b (default) and powell
            are available.

        :param op_strict: [optional]
            Default to Powell's optimization method if BFGS fails.

        :param op_kwds:
            Keyword arguments to provide directly to the optimization function.

        :param progressbar: [optional]
            Display a progress bar while pixels are being fit (requires
            ``tqdm``; silently disabled if it is not installed).

        :param batch_size: [optional]
            The number of pixels to fit per vectorized batch. Pixels are fit
            independently, so this does not change the result; it only controls
            the granularity of the progress bar and bounds the peak memory and
            compilation cost of the vmapped optimizer. Defaults to a value that
            yields ~50 progress updates with reasonably large batches.

        :returns:
            A three-length tuple containing the spectral coefficients `theta`,
            the squared scatter term at each pixel `s2`, and metadata related to
            the training of each pixel.
        """

        if self.training_set_flux is None or self.training_set_ivar is None:
            raise TypeError(
                "cannot train: training set spectra not saved with the model")

        if threads not in (1, None):
            logger.warn("The `threads` argument is deprecated and ignored: "
                        "training is vectorized with jax.vmap.")

        S, P = self.training_set_flux.shape
        T = self.design_matrix.shape[1]

        logger.info("Training {0}-label {1} with {2} stars and {3} pixels/star"\
            .format(len(self.vectorizer.label_names), type(self).__name__, S, P))

        op_method = op_method or "l_bfgs_b"
        op_kwds = op_kwds or {}
        maxiter = op_kwds.get("maxiter", fitting._TRAIN_MAXITER)
        tol = op_kwds.get("tol", fitting._TRAIN_TOL)

        # Optional box constraints on theta (used by RestrictedCannonModel). The
        # bounds are given as a list of (min, max) tuples per term, with None
        # indicating no limit on that side.
        bounds = op_kwds.get("bounds", None)
        if bounds is not None:
            lower = jnp.asarray(
                [(-jnp.inf if lo is None else lo) for lo, _ in bounds],
                dtype=float)
            upper = jnp.asarray(
                [(jnp.inf if hi is None else hi) for _, hi in bounds],
                dtype=float)
            bounds = (lower, upper)

        # Batched (pixel-major) flux and inverse variance.
        flux_PN = jnp.asarray(self.training_set_flux.T)    # (P, N)
        ivar_PN = jnp.asarray(self.training_set_ivar.T)    # (P, N)
        design_matrix = jnp.asarray(self.design_matrix)    # (N, T)

        fiducial = jnp.concatenate([jnp.ones(1), jnp.zeros(T - 1)])

        # Initial theta guesses for every pixel: a linear-algebra estimate and
        # the fiducial value. The regularized objective is convex, so the
        # optimum is independent of the starting point; these only set where the
        # optimizer begins.
        linalg_theta = jax.vmap(
            lambda f, i: fitting.fit_theta_by_linalg(f, i, 0.0, design_matrix)[0]
        )(flux_PN, ivar_PN)                                 # (P, T)
        finite = jnp.all(jnp.isfinite(linalg_theta), axis=1, keepdims=True)
        linalg_theta = jnp.where(finite, linalg_theta, fiducial[None, :])
        init_stack = jnp.stack(
            [linalg_theta, jnp.broadcast_to(fiducial, (P, T))], axis=1)  # (P,2,T)

        # Per-pixel column mask: True keeps the coefficient, False censors it.
        if self.censors and len(set(self.censors).intersection(
                self.vectorizer.label_names)) > 0:
            column_mask = jnp.asarray(
                censoring.design_matrix_mask(self.censors, self.vectorizer))
        else:
            column_mask = jnp.ones((P, T), dtype=bool)

        # Per-pixel regularization strength.
        reg = 0.0 if self.regularization is None else self.regularization
        reg_P = jnp.broadcast_to(jnp.asarray(reg, dtype=float), (P,))

        fitter = fitting.make_pixel_fitter(
            op_method=op_method, maxiter=maxiter, tol=tol, bounds=bounds)
        batched = jax.jit(jax.vmap(
            fitter, in_axes=(0, 0, 0, None, 0, 0)))

        # Pixels are fit independently, so we run them in fixed-size batches
        # rather than one giant vmap. This lets us report progress (the single
        # fused call is opaque), bounds peak memory, and keeps the compiled
        # program small enough to compile once and reuse for every batch.
        if batch_size is None:
            batch_size = max(512, int(np.ceil(P / 50)))
        batch_size = int(min(max(1, batch_size), P))

        # Pad the pixel axis up to a whole number of equally-sized batches so
        # every call to `batched` has identical shapes and XLA compiles it only
        # once. The padding pixels are duplicates of the last real pixel and are
        # discarded below.
        n_batches = int(np.ceil(P / batch_size))
        pad = n_batches * batch_size - P
        if pad:
            edge = lambda a: jnp.broadcast_to(a[-1:], (pad,) + a.shape[1:])
            flux_PN = jnp.concatenate([flux_PN, edge(flux_PN)], axis=0)
            ivar_PN = jnp.concatenate([ivar_PN, edge(ivar_PN)], axis=0)
            init_stack = jnp.concatenate([init_stack, edge(init_stack)], axis=0)
            reg_P = jnp.concatenate([reg_P, edge(reg_P)], axis=0)
            column_mask = jnp.concatenate([column_mask, edge(column_mask)],
                                          axis=0)

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        pbar = None
        if progressbar and tqdm is not None:
            pbar = tqdm(total=P, desc="Training", unit="px")

        theta_batches, s2_batches, fopt_batches = [], [], []
        for b in range(n_batches):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            th, s2b, fob = batched(
                flux_PN[sl], ivar_PN[sl], init_stack[sl], design_matrix,
                reg_P[sl], column_mask[sl])
            # Block on the device (keeping the results as JAX arrays) so the bar
            # tracks real work rather than JAX's asynchronous dispatch.
            jax.block_until_ready((th, s2b, fob))
            theta_batches.append(th)
            s2_batches.append(s2b)
            fopt_batches.append(fob)
            if pbar is not None:
                pbar.update(min((b + 1) * batch_size, P) - b * batch_size)

        if pbar is not None:
            pbar.close()

        # Concatenate the batches and drop any padding pixels. The trained
        # quantities are kept as JAX arrays end-to-end.
        theta = jnp.concatenate(theta_batches, axis=0)[:P]
        s2 = jnp.concatenate(s2_batches, axis=0)[:P]
        fopt = jnp.concatenate(fopt_batches, axis=0)[:P]

        # A single host transfer for the per-pixel metadata.
        fopt_values = fopt.tolist()
        meta = [dict(op_method=op_method, fopt=fopt_values[p],
                     maxiter=maxiter, tol=tol) for p in range(P)]

        self._theta, self._s2 = (theta, s2)

        return (theta, s2, meta)


    @requires_training
    def __call__(self, labels):
        """
        Return spectral fluxes, given the labels.

        :param labels:
            An array of stellar labels.
        """

        # Scale and offset the labels.
        scaled_labels = (np.atleast_2d(labels) - self._fiducials)/self._scales
        flux = np.dot(self.theta, np.asarray(self.vectorizer(scaled_labels))).T
        return flux[0] if flux.shape[0] == 1 else flux


    @requires_training
    def test(self, flux, ivar, initial_labels=None, threads=None, 
        use_derivatives=True, op_kwds=None):
        """
        Run the test step on spectra.

        :param flux:
            The (pseudo-continuum-normalized) spectral flux.

        :param ivar:
            The inverse variance values for the spectral fluxes.

        :param initial_labels: [optional]
            The initial labels to try for each spectrum. This can be a single
            set of initial values, or one set of initial values for each star.

        :param threads: [optional]
            The number of parallel threads to use.

        :param use_derivatives: [optional]
            Boolean `True` indicating to use analytic derivatives provided by 
            the vectorizer, `None` to calculate on the fly, or a callable
            function to calculate your own derivatives.

        :param op_kwds: [optional]
            Optimization keywords that get passed to `scipy.optimize.leastsq`.
        """

        if flux is None or ivar is None:
            raise ValueError("flux and ivar must not be None")

        if op_kwds is None:
            op_kwds = dict()

        if threads not in (1, None):
            logger.warn("The `threads` argument is deprecated and ignored: the "
                        "test step is vectorized with jax.vmap.")

        flux, ivar = (np.atleast_2d(flux), np.atleast_2d(ivar))
        S, P = flux.shape

        if ivar.shape != flux.shape:
            raise ValueError("flux and ivar arrays must be the same shape")

        L = len(self._fiducials)

        if initial_labels is None:
            initial_labels = self._fiducials

        # Coerce initial labels to shape (S, n_init, L).
        initial_labels = np.atleast_2d(np.asarray(initial_labels, dtype=float))
        if initial_labels.shape == (S, L):
            initial_labels = initial_labels[:, None, :]   # one start per star
        elif initial_labels.ndim == 2:
            initial_labels = np.tile(initial_labels[None], (S, 1, 1))  # shared
        elif initial_labels.ndim == 3 and initial_labels.shape[0] != S:
            initial_labels = np.tile(initial_labels, (S, 1, 1))

        maxiter = op_kwds.get("maxiter", fitting._TEST_MAXITER)
        tol = op_kwds.get("tol", fitting._TEST_TOL)

        core = fitting.make_spectrum_fitter(
            self.vectorizer, self.theta, self.s2, self._fiducials,
            self._scales, maxiter=maxiter, tol=tol)
        batched = jax.jit(jax.vmap(core, in_axes=(0, 0, 0)))

        op_labels, cov, chi_sq, model_flux, n_use = batched(
            jnp.asarray(flux), jnp.asarray(ivar), jnp.asarray(initial_labels))

        op_labels = np.asarray(op_labels)
        cov = np.asarray(cov)
        chi_sq = np.asarray(chi_sq)
        n_use = np.asarray(n_use)

        meta = [dict(chi_sq=float(chi_sq[s]),
                     r_chi_sq=float(chi_sq[s]) / max(1, int(n_use[s]) - L - 1),
                     method="levenberg_marquardt",
                     label_names=self.vectorizer.label_names)
                for s in range(S)]

        return (op_labels, cov, meta)


    def _initial_theta(self, pixel_index, **kwargs):
        """
        Return a list of guesses of the spectral coefficients for the given
        pixel index. Initial values are sourced in the following preference
        order: 

            (1) a previously trained `theta` value for this pixel,
            (2) an estimate of `theta` using linear algebra,
            (3) a neighbouring pixel's `theta` value,
            (4) the fiducial value of [1, 0, ..., 0].

        :param pixel_index:
            The zero-indexed integer of the pixel.

        :returns:
            A list of initial theta guesses, and the source of each guess.
        """

        guesses = []

        if self.theta is not None:
            # Previously trained theta value.
            if np.all(np.isfinite(self.theta[pixel_index])):
                guesses.append((self.theta[pixel_index], "previously_trained"))

        # Estimate from linear algebra.
        theta, cov = fitting.fit_theta_by_linalg(
            self.training_set_flux[:, pixel_index],
            self.training_set_ivar[:, pixel_index],
            s2=kwargs.get("s2", 0.0), design_matrix=self.design_matrix)

        if np.all(np.isfinite(theta)):
            guesses.append((theta, "linear_algebra"))

        if self.theta is not None:
            # Neighbouring pixels value.
            for neighbour_pixel_index in set(np.clip(
                [pixel_index - 1, pixel_index + 1], 
                0, self.training_set_flux.shape[1] - 1)):

                if np.all(np.isfinite(self.theta[neighbour_pixel_index])):
                    guesses.append(
                        (self.theta[neighbour_pixel_index], "neighbour_pixel"))

        # Fiducial value.
        fiducial = np.hstack([1.0, np.zeros(len(self.vectorizer.terms))])
        guesses.append((fiducial, "fiducial"))

        return guesses
