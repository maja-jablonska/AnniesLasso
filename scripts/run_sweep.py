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
        --label-set raw_teff,raw_logg,raw_fe_h \\
        --label-set raw_teff,raw_logg,raw_fe_h,mg_fe,ce_fe \\
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
                                       quality_mask, DEFAULT_DATA_DIR,
                                       DEFAULT_LABELS)
    from scripts.sweep_cannon import sweep
except ImportError:
    from train_cannon import (load_spectra, normalize_spectra, quality_mask,
                              DEFAULT_DATA_DIR, DEFAULT_LABELS)
    from sweep_cannon import sweep

logger = logging.getLogger("thecannon.run_sweep")


def enable_jax_compilation_cache(cache_dir):
    """
    Turn on JAX's persistent (on-disk) compilation cache. The sweep's jitted
    train/test programs take every model-specific array as an argument, so
    grid points that share shapes produce byte-identical HLO -- the first run
    of a sweep pays the XLA compilation cost once per shape, and every later
    fold, regularization strength, restarted job, or re-run hits this cache
    instead of recompiling.
    """
    import jax
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    # Persist every program that takes >= 1 s to compile, regardless of size.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    logger.info("JAX persistent compilation cache: %s", cache_dir)


def build_label_sets(base_core, age_cols, mass_col, abundances, mode):
    """
    Build the grid of label sets from a fixed core, one or more age columns
    (each making a base variant), an optional mass column, and a list of
    abundances. ``mode`` controls how the extras are added on top of each base:

      - ``one-at-a-time``: base, base+mass, base+each abundance, base+all
        abundances, base+mass+all abundances. Isolates each addition's effect.
      - ``cumulative``: base -> +mass -> +abund1 -> +abund2 -> ... (each set a
        superset of the previous).
      - ``minimal``: base, base+all abundances, base+mass, base+mass+all
        abundances.
    """
    def dedup(seq):
        return tuple(dict.fromkeys(seq))        # drop dupes, preserve order

    sets = []
    for age in age_cols:
        base = list(base_core) + ([age] if age else [])
        sets.append(dedup(base))

        if mode == "one-at-a-time":
            if mass_col:
                sets.append(dedup(base + [mass_col]))
            for a in abundances:
                sets.append(dedup(base + [a]))
            if abundances:
                sets.append(dedup(base + list(abundances)))
            if mass_col and abundances:
                sets.append(dedup(base + [mass_col] + list(abundances)))

        elif mode == "cumulative":
            current = list(base)
            if mass_col:
                current.append(mass_col)
                sets.append(dedup(current))
            for a in abundances:
                current.append(a)
                sets.append(dedup(current))

        elif mode == "minimal":
            if abundances:
                sets.append(dedup(base + list(abundances)))
            if mass_col:
                sets.append(dedup(base + [mass_col]))
            if mass_col and abundances:
                sets.append(dedup(base + [mass_col] + list(abundances)))

    return list(dict.fromkeys(sets))            # dedup whole sets across ages


def _has_column(label_source, name):
    """ True if ``name`` is a column of the table / key of the mapping. """
    columns = getattr(label_source, "columns", None)
    return (name in columns) if columns is not None else (name in label_source)


def filter_existing(label_sets, label_source):
    """ Drop label sets that reference a column missing from the data. """
    kept = []
    for label_set in label_sets:
        missing = [n for n in label_set if not _has_column(label_source, n)]
        if missing:
            logger.warning("skipping label set %s (missing columns: %s)",
                           "+".join(label_set), ",".join(missing))
        else:
            kept.append(label_set)
    return kept


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
    parser.add_argument("--base", type=lambda s: s.split(","),
                        default=["raw_teff", "raw_logg", "raw_fe_h", "mg_fe"],
                        help="comma-separated core labels in every set (the age "
                             "column from --age-cols is appended to this)")
    parser.add_argument("--age-cols", type=lambda s: s.split(","),
                        default=["age_Dnu", "age_L"],
                        help="age column(s); each makes a separate base variant "
                             "(missing columns are skipped)")
    parser.add_argument("--mass-col", default="mass_L",
                        help="mass column added by the 'age and mass' sets "
                             "(empty string to disable)")
    parser.add_argument("--abundances", type=lambda s: s.split(","),
                        default=["ce_fe", "ca_fe", "si_fe", "ni_fe",
                                 "mn_fe", "al_fe", "c_fe", "n_fe"],
                        help="comma-separated abundances to test on top of base "
                             "(the <x>_fe columns are derived from raw_<x>_h - "
                             "raw_fe_h at load time)")
    parser.add_argument("--label-set-mode", default="one-at-a-time",
                        choices=["one-at-a-time", "cumulative", "minimal"],
                        help="how extras are combined with each base")
    parser.add_argument("--label-set", dest="label_sets", action="append",
                        type=lambda s: tuple(s.split(",")), default=None,
                        help="explicit label set to try (repeatable); overrides "
                             "the builder when given")
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
    parser.add_argument("--jax-cache-dir",
                        default=os.environ.get("JAX_COMPILATION_CACHE_DIR",
                                               "~/.cache/thecannon-jax"),
                        help="persistent XLA compilation cache directory "
                             "(reused across folds/grid points/jobs); pass an "
                             "empty string to disable")
    parser.add_argument("--demo", action="store_true",
                        help="run on the bundled golden data instead")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable INFO-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.jax_cache_dir:
        enable_jax_compilation_cache(args.jax_cache_dir)

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

        # Quality cuts before anything is normalized or trained on: drop stars
        # with flagged spectra (spectrum_flags != 0) or any warn_* label set.
        good = quality_mask(label_source)
        if not good.any():
            raise ValueError("quality cuts rejected every star; check the "
                             "spectrum_flags / warn_* columns")
        logger.info("quality cuts: keeping %d/%d stars",
                    int(good.sum()), good.size)
        label_source = label_source[good]
        flux, ivar = flux[good], ivar[good]

        norm_flux, norm_ivar = normalize_spectra(
            dispersion, flux, ivar, args.continuum_list)
        label_sets = args.label_sets or build_label_sets(
            args.base, args.age_cols, args.mass_col, args.abundances,
            args.label_set_mode)
        label_sets = filter_existing(label_sets, label_source)
        if not label_sets:
            raise ValueError("no usable label sets (all referenced columns "
                             "missing); check --base/--age-cols/--abundances")
        n_splits = args.n_splits

    # Union of every label used by any label set; drop non-finite stars once.
    label_union = list(dict.fromkeys(
        name for label_set in label_sets for name in label_set))
    mapping, finite = finite_label_mapping(label_source, label_union)

    logger.info("sweeping %d label sets x %d orders x %d regularizations",
                len(label_sets), len(args.orders), len(args.regularizations))
    for label_set in label_sets:
        logger.info("  label set: %s", "+".join(label_set))

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
