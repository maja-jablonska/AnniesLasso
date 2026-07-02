#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared configuration for the Cannon sweep / selection drivers.

This is the single home for the pieces that would otherwise be copy-pasted
across scripts.run_sweep, scripts.wandb_sweep and scripts.select_labels: the
default label-set-builder knobs, the argparse group that exposes them, and the
loader for the bundled golden test set.

It is deliberately lightweight -- only the standard library and numpy -- so that
importing it never pulls in jax/thecannon/matplotlib. In particular the W&B
Bayesian *controller* (scripts.wandb_sweep --role controller) imports this for
its defaults while staying free of the science stack it does not need.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os

import numpy as np


# --------------------------------------------------------------------------- #
#  Label-set builder defaults (see scripts.run_sweep.build_label_sets)          #
# --------------------------------------------------------------------------- #

DEFAULT_BASE = ["raw_teff", "raw_logg", "raw_fe_h", "mg_fe"]
DEFAULT_AGE_COLS = ["age_Dnu", "age_L"]
DEFAULT_MASS_COL = "mass_L"
DEFAULT_ABUNDANCES = ["ce_fe", "ca_fe", "si_fe", "ni_fe",
                      "mn_fe", "al_fe", "c_fe", "n_fe"]
DEFAULT_LABEL_SET_MODE = "one-at-a-time"
LABEL_SET_MODES = ["one-at-a-time", "cumulative", "minimal"]


def add_label_builder_args(parser):
    """
    Add the shared label-set-builder arguments to an ``argparse`` parser:
    ``--base``, ``--age-cols``, ``--mass-col``, ``--abundances``,
    ``--label-set-mode`` and the repeatable ``--label-set`` override. Used by
    both scripts.run_sweep and scripts.wandb_sweep so the two stay in sync.
    """
    parser.add_argument("--base", type=lambda s: s.split(","),
                        default=list(DEFAULT_BASE),
                        help="comma-separated core labels in every set (the age "
                             "column from --age-cols is appended to this)")
    parser.add_argument("--age-cols", type=lambda s: s.split(","),
                        default=list(DEFAULT_AGE_COLS),
                        help="age column(s); each makes a separate base variant "
                             "(missing columns are skipped)")
    parser.add_argument("--mass-col", default=DEFAULT_MASS_COL,
                        help="mass column added by the 'age and mass' sets "
                             "(empty string to disable)")
    parser.add_argument("--abundances", type=lambda s: s.split(","),
                        default=list(DEFAULT_ABUNDANCES),
                        help="comma-separated abundances to test on top of base "
                             "(the <x>_fe columns are derived from raw_<x>_h - "
                             "raw_fe_h at load time)")
    parser.add_argument("--label-set-mode", default=DEFAULT_LABEL_SET_MODE,
                        choices=LABEL_SET_MODES,
                        help="how extras are combined with each base")
    parser.add_argument("--label-set", dest="label_sets", action="append",
                        type=lambda s: tuple(s.split(",")), default=None,
                        help="explicit label set to try (repeatable); overrides "
                             "the builder when given")
    return parser


# --------------------------------------------------------------------------- #
#  Bundled golden test set                                                     #
# --------------------------------------------------------------------------- #

def golden_path():
    """ Path to the packaged golden test pickle. """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "thecannon", "tests", "golden", "golden.pkl")


def load_golden():
    """
    Load the bundled (already continuum-normalized) golden test set.

    :returns:
        ``(names, label_source, dispersion, flux, ivar)`` where ``names`` is the
        list of label names, ``label_source`` is a ``{name: (N,) array}`` mapping
        of the reference labels, and ``flux``/``ivar`` are ``(N, P)`` arrays.
    """
    import pickle
    with open(golden_path(), "rb") as fp:
        meta = pickle.load(fp)["meta"]
    names = list(meta["label_names"])
    arr = np.atleast_2d(np.asarray(meta["labels"], dtype=float))
    label_source = {name: arr[:, i] for i, name in enumerate(names)}
    return names, label_source, meta["dispersion"], meta["flux"], meta["ivar"]
