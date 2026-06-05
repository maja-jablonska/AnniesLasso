#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "0.2.93"

import logging
try:
    from numpy import RankWarning
except ImportError:
    # numpy >= 2.0 moved RankWarning to numpy.exceptions.
    from numpy.exceptions import RankWarning
from warnings import simplefilter

# The Cannon performs scientific (double-precision) linear algebra and
# optimization. JAX defaults to 32-bit; enable 64-bit globally *before* any
# submodule imports jax.numpy. NOTE: this is a process-wide setting and will
# affect any other JAX user in the same interpreter. Import thecannon early.
import jax as _jax
_jax.config.update("jax_enable_x64", True)

# Compatibility shim: jaxopt 0.8.3 (unmaintained) still calls jax.tree_map,
# which was deprecated in JAX 0.4.x and removed in JAX 0.6. Restore the alias so
# the jaxopt-based test step (fitting.py) works on modern JAX. Harmless on older
# JAX where the attribute already exists.
if not hasattr(_jax, "tree_map"):
    _jax.tree_map = _jax.tree_util.tree_map

from .model import CannonModel
from . import (censoring, continuum, fitting, plot, utils, vectorizer)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # TODO: Remove this when stable.

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

simplefilter("ignore", RankWarning)
simplefilter("ignore", RuntimeWarning)


def load_model(path, **kwargs):
    """
    Load a Cannon model from an existing filename, regardless of the kind of
    Cannon model sub-class.

    :param path:
        The path where the model has been saved. This saved model must include
        a labelled data set.
    """

    print("deprecated; use CannonModel.read") # TODO
    return CannonModel.read(path, **kwargs)


# Clean up the top-level namespace for this module.
del handler, logger, logging, RankWarning, simplefilter, _jax
