#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run a cross-validated Cannon hyper-parameter sweep on real APOGEE data.

This is the driver that ties the two pieces together: it loads and
continuum-normalizes the spectra with the helpers from
:mod:`scripts.train_cannon`, then hands the in-memory arrays to
:func:`scripts.sweep_cannon.sweep`, which does the k-fold cross-validation over
a grid of label sets, polynomial orders and regularization strengths (logging
one offline W&B run per grid point).

JAX device selection is via the ``JAX_PLATFORMS`` environment variable.

Usage
-----
::

    python -m scripts.run_sweep \\
        --spectra /path/cleaned_ages.parquet \\
        --continuum-list /path/continuum.list \\
        --labels raw_teff,raw_logg,raw_fe_h,raw_mg_h,raw_ce_h,age_L,mass_L \\
        --label-set raw_teff,raw_logg,raw_fe_h \\
        --label-set raw_teff,raw_logg,raw_fe_h,raw_mg_h,raw_ce_h \\
        --orders 1,2 --regularizations 0,1e2,1e3,1e4 \\
        --n-splits 5 --wandb-project cannon-sweep

Quick end-to-end check on the bundled golden data::

    JAX_PLATFORMS=cpu python -m scripts.run_sweep --demo
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import logging
import os

import numpy as np

# Work both as a package module (`python -m scripts.run_sweep`) and when run
# directly from the scripts/ directory.
try:
    from scripts.train_cannon import (load_spectra, normalize_spectra,
                                       DEFAULT_DATA_DIR, DEFAULT_LABELS)
    from scripts.sweep_cannon import sweep
except ImportError:
    from train_cannon import (load_spectra, normalize_spectra,
                              DEFAULT_DATA_DIR, DEFAULT_LABELS)
    from sweep_cannon import sweep

logger = logging.getLogger("thecannon.run_sweep")


def default_label_sets(labels):
    """ Nested label sets (first 3, first 5, all) -- 'does adding labels help?' """
    sets = []
    for k in (3, 5, len(labels)):
        if 1 <= k <= len(labels):
            sets.append(tuple(labels[:k]))
    return list(dict.fromkeys(sets))            # dedup, preserve order


def finite_label_mapping(label_source, label_union):
    """
    Build a ``{name: (M,) array}`` mapping restricted to stars with finite
    values across *every* label in ``label_union`` (the sweep does no filtering,
    and a NaN label silently poisons that pixel's fit).
    """
    matrix = np.vstack(
        [np.asarray(label_source[name], dtype=float) for name in label_union]).T
    finite = np.isfinite(matrix).all(axis=1)
    logger.info("%d/%d stars retained after dropping non-finite labels",
                int(finite.sum()), len(finite))
    mapping = {name: matrix[finite, i] for i, name in enumerate(label_union)}
    return mapping, finite


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spectra",
                        default=os.path.join(DEFAULT_DATA_DIR,
                                             "cleaned_ages.parquet"),
                        help="parquet table of spectra + labels")
    parser.add_argument("--continuum-list",
                        default=os.path.join(DEFAULT_DATA_DIR, "continuum.list"),
                        help="text file of continuum pixel indices")
    parser.add_argument("--labels", type=lambda s: s.split(","),
                        default=DEFAULT_LABELS,
                        help="comma-separated master list of label columns")
    parser.add_argument("--label-set", dest="label_sets", action="append",
                        type=lambda s: tuple(s.split(",")), default=None,
                        help="a label set to try (repeatable); defaults to "
                             "nested subsets of --labels")
    parser.add_argument("--orders", type=lambda s: [int(x) for x in s.split(",")],
                        default=[1, 2], help="comma-separated polynomial orders")
    parser.add_argument("--regularizations",
                        type=lambda s: [float(x) for x in s.split(",")],
                        default=[0.0, 1e2, 1e3, 1e4],
                        help="comma-separated L1 regularization strengths")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="number of cross-validation folds")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the (shared) fold assignment")
    parser.add_argument("--output", default="sweep_results.csv",
                        help="CSV path for the tidy results table")
    parser.add_argument("--wandb-project", default=None,
                        help="log each grid point to this W&B project")
    parser.add_argument("--wandb-mode", default="offline",
                        choices=["offline", "online", "disabled"],
                        help="W&B run mode (default: offline)")
    parser.add_argument("--demo", action="store_true",
                        help="run on the bundled golden data instead")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable INFO-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.demo:
        import pickle
        golden_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "thecannon", "tests", "golden", "golden.pkl")
        with open(golden_path, "rb") as fp:
            meta = pickle.load(fp)["meta"]
        names = list(meta["label_names"])
        arr = np.atleast_2d(np.asarray(meta["labels"], dtype=float))
        label_source = {name: arr[:, i] for i, name in enumerate(names)}
        dispersion = meta["dispersion"]
        norm_flux, norm_ivar = meta["flux"], meta["ivar"]   # already normalized
        labels_master = names
        label_sets = [tuple(names[:2]), tuple(names)]
        n_splits = min(args.n_splits, 3)
        print("Demo on golden data: {0} stars, {1} pixels, labels {2}".format(
            norm_flux.shape[0], norm_flux.shape[1], names))
    else:
        label_source, dispersion, flux, ivar = load_spectra(args.spectra)
        norm_flux, norm_ivar = normalize_spectra(
            dispersion, flux, ivar, args.continuum_list)
        labels_master = args.labels
        label_sets = args.label_sets or default_label_sets(labels_master)
        n_splits = args.n_splits

    # Union of every label used by any label set; drop non-finite stars once.
    label_union = list(dict.fromkeys(
        name for label_set in label_sets for name in label_set))
    mapping, finite = finite_label_mapping(label_source, label_union)

    logger.info("sweeping %d label sets x %d orders x %d regularizations",
                len(label_sets), len(args.orders), len(args.regularizations))

    sweep(
        mapping, norm_flux[finite], norm_ivar[finite], dispersion,
        label_sets=label_sets,
        orders=args.orders,
        regularizations=args.regularizations,
        n_splits=n_splits,
        seed=args.seed,
        output=args.output,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode)


if __name__ == "__main__":
    main()
