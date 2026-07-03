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


def add_filter_arg(parser):
    """
    Add the repeatable ``--filter`` row-selection argument: each value is a
    pandas query on the label table (e.g. ``"Rel_age_Dnu == True"``) that the
    training / cross-validation pool must satisfy. Multiple filters are ANDed.
    """
    parser.add_argument("--filter", dest="filters", action="append",
                        default=None, metavar="QUERY",
                        help="row filter as a pandas query on the label table, "
                             "e.g. \"Rel_age_Dnu == True\" (repeatable; all must "
                             "pass). Applied to the training/CV pool before "
                             "continuum normalization.")
    return parser


def _column_hint(label_source, expr):
    """
    Build a " Did you mean ..." hint for a failed filter: the table columns whose
    name shares a word-part with the query (case-insensitive), so a wrong /
    mis-cased column name is easy to correct. Falls back to a truncated column
    list. Returns an empty string if columns are unavailable.
    """
    import re
    columns = getattr(label_source, "columns", None)
    cols = list(columns) if columns is not None else []
    if not cols:
        return ""
    skip = {"true", "false", "and", "or", "not", "in", "none"}
    parts = set()
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr):
        if tok.lower() in skip:
            continue
        for p in tok.lower().split("_"):
            if len(p) >= 2:
                parts.add(p)
    near = [c for c in cols if any(p in str(c).lower() for p in parts)]
    shown = near if near else cols
    listed = ", ".join(map(str, shown[:40]))
    more = "" if len(shown) <= 40 else " (+{0} more)".format(len(shown) - 40)
    label = "similarly-named columns" if near else "available columns"
    return " {0}: {1}{2}".format(label, listed, more)


# --------------------------------------------------------------------------- #
#  Age-reliability cut (per label set)                                          #
# --------------------------------------------------------------------------- #

# Each age/mass label column is only reliable for stars where its flag column is
# True; a label set is cut to the stars for which every such column it fits is
# reliable (no cut when it fits none). mass_L shares age_L's isochrone fit, so it
# keys off the same RelAge_L flag. Extend this mapping as needed.
AGE_RELIABILITY = {"age_Dnu": "RelAge_Dnu", "age_L": "RelAge_L",
                   "mass_L": "RelAge_L"}


def age_reliability_masks(label_source, mapping=None):
    """
    Return ``{age_col: bool_array}`` giving, for each age column whose flag column
    exists in ``label_source``, the per-row "this age is reliable" mask. Flags may
    be boolean, 0/1, or the strings "True"/"1". Age columns with no flag column
    present are omitted (so they impose no cut).
    """
    mapping = AGE_RELIABILITY if mapping is None else mapping
    columns = getattr(label_source, "columns", None)
    cols = list(columns) if columns is not None else []
    out = {}
    for age_col, rel_col in mapping.items():
        if rel_col in cols:
            col = label_source[rel_col]
            out[age_col] = np.asarray(
                col.isin([True, 1, "True", "true", "1"])
                if hasattr(col, "isin") else np.asarray(col, dtype=bool))
    return out


def _element_of(col):
    """ The element token of an abundance label: ``mg_fe``/``raw_mg_h`` -> "mg"
    (also iron via ``raw_fe_h`` -> "fe"); ``None`` for non-abundance labels. """
    import re
    c = str(col).lower()
    m = re.fullmatch(r"([a-z]{1,2})_fe", c)
    if m:
        return m.group(1)
    m = re.fullmatch(r"raw_([a-z]{1,2})_h", c)
    if m:
        return m.group(1)
    return None


def abundance_flag_masks(label_source):
    """
    Return ``{label_col: bool_array}`` where the mask is True for stars whose
    ASPCAP abundance flag for that element is 0 (unflagged). Recognises the flag
    column ``<EL>_FE_FLAG`` or ``<EL>_H_FLAG`` (case-insensitive); labels with no
    flag column present are omitted (so they impose no cut).
    """
    columns = getattr(label_source, "columns", None)
    cols = list(columns) if columns is not None else []
    if not cols:
        return {}
    lut = {str(c).lower(): c for c in cols}
    out = {}
    for col in cols:
        element = _element_of(col)
        if element is None:
            continue
        for cand in ("{0}_fe_flag".format(element), "{0}_h_flag".format(element)):
            if cand in lut:
                flags = np.asarray(label_source[lut[cand]])
                # NaN -> treat as unflagged (0); otherwise 0 == good.
                flags = np.where(np.equal(flags, flags), flags, 0)
                out[col] = np.asarray(flags).astype("float64") == 0
                break
    return out


def per_label_masks(label_source):
    """
    Combined per-label "OK to use this star" masks: the age/mass reliability
    flags (:func:`age_reliability_masks`) AND the abundance flags
    (:func:`abundance_flag_masks`), keyed by label column. A label set is cut to
    the stars for which every present key it fits is OK (see
    :func:`label_set_row_mask`).
    """
    masks = age_reliability_masks(label_source)
    for col, mask in abundance_flag_masks(label_source).items():
        masks[col] = (masks[col] & mask) if col in masks else mask
    return masks


def label_set_row_mask(label_names, age_masks):
    """
    Boolean row mask selecting stars for which every age column present in
    ``label_names`` is reliable, or ``None`` when the set fits no age column (no
    restriction). ``age_masks`` is the dict from :func:`age_reliability_masks`.
    """
    if not age_masks:
        return None
    masks = [age_masks[c] for c in label_names if c in age_masks]
    if not masks:
        return None
    mask = masks[0].copy()
    for m in masks[1:]:
        mask = mask & m
    return mask


def filter_mask(label_source, filters, log=None):
    """
    Boolean mask over the rows of ``label_source`` (a pandas DataFrame) selecting
    those that pass every pandas query in ``filters`` (ANDed). An all-True mask
    is returned when ``filters`` is empty. Use this when you need the mask itself
    -- e.g. to AND a row cut into a training-set-selection mask while still
    predicting for every star; use :func:`apply_filters` to subset outright.
    """
    mask = np.ones(len(label_source), dtype=bool)
    if not filters:
        return mask

    for expr in filters:
        try:
            m = np.asarray(label_source.eval(expr), dtype=bool)
        except Exception as exc:
            raise ValueError("could not apply --filter {0!r}: {1}.{2}".format(
                expr, exc, _column_hint(label_source, expr)))
        if m.shape != mask.shape:
            raise ValueError(
                "--filter {0!r} did not yield one boolean per row (got shape "
                "{1})".format(expr, m.shape))
        if log is not None:
            log.info("filter %r: %d/%d rows pass", expr, int(m.sum()), len(m))
        mask &= m

    if log is not None:
        log.info("row filters kept %d/%d stars", int(mask.sum()), len(mask))
    return mask


def apply_filters(label_source, flux, ivar, filters, log=None):
    """
    Restrict ``(label_source, flux, ivar)`` to the rows passing every pandas
    query in ``filters``. ``label_source`` is a pandas DataFrame; ``flux`` and
    ``ivar`` are row-aligned ``(N, P)`` arrays. Returns the filtered triple (the
    inputs unchanged when ``filters`` is empty).
    """
    if not filters:
        return label_source, flux, ivar

    mask = filter_mask(label_source, filters, log=log)
    if not mask.any():
        raise ValueError("row filters rejected every star: {0}".format(filters))
    return label_source[mask], flux[mask], ivar[mask]


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
