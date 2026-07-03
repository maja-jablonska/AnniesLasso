#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train one CannonModel from a config, then apply it to *every* star in a sample.

Unlike :mod:`scripts.train_cannon` (which only fits the held-out validation
fold and reports one-to-one metrics), this script runs the trained model's test
step over the whole loaded sample and writes a per-star label catalogue: the
inferred labels, their formal uncertainties, the fit chi-squared, and a flag
marking which stars were used for training.

The workflow:

  1. read a JSON config (or take the built-in defaults / CLI overrides),
  2. load the parquet table of APOGEE spectra + ASPCAP labels,
  3. pseudo-continuum-normalize the spectra,
  4. train a polynomial CannonModel on the training stars (those with finite
     labels that pass the quality cut; optionally thinned to ``train_frac``),
  5. apply the trained model to ALL stars in the sample, and
  6. write the trained ``.model`` file and a per-star predictions catalogue.

JAX device selection is via the ``JAX_PLATFORMS`` environment variable (this
script does not force one), e.g. ``JAX_PLATFORMS=cpu`` where the GPU backend is
unavailable.

Usage
-----
With an explicit config file::

    python -m scripts.apply_cannon \\
        --spectra /path/to/merged_with_ages_raw.parquet \\
        --continuum-list /path/to/continuum.list \\
        --config config.json \\
        --output-dir results/

A minimal ``config.json`` looks like::

    {
      "labels": ["raw_teff", "raw_logg", "raw_fe_h", "mg_fe"],
      "order": 2,
      "regularization": 0.0,
      "train_frac": 1.0,
      "quality_cut": true,
      "seed": 888
    }

Quick end-to-end check on the bundled golden data (train on all, apply to all)::

    JAX_PLATFORMS=cpu python -m scripts.apply_cannon --demo
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import json
import logging
import os

import numpy as np

import thecannon as tc

# Work both as a package module (`python -m scripts.apply_cannon`) and when run
# directly from the scripts/ directory.
try:
    from scripts.train_cannon import (load_spectra, normalize_spectra,
                                       quality_mask, DEFAULT_DATA_DIR,
                                       DEFAULT_SPECTRA, DEFAULT_CONTINUUM_LIST,
                                       DEFAULT_LABELS)
    from scripts.sweep_config import (add_filter_arg, filter_mask,
                                       per_label_masks, label_set_row_mask)
except ImportError:
    from train_cannon import (load_spectra, normalize_spectra, quality_mask,
                              DEFAULT_DATA_DIR, DEFAULT_SPECTRA,
                              DEFAULT_CONTINUUM_LIST, DEFAULT_LABELS)
    from sweep_config import (add_filter_arg, filter_mask, per_label_masks,
                             label_set_row_mask)

logger = logging.getLogger("thecannon.apply")


# Config keys that select the training hyper-parameters and the training
# sample. Anything the CannonModel/vectorizer accepts can be surfaced here.
DEFAULT_CONFIG = {
    "labels": DEFAULT_LABELS,     # label column names to train on
    "order": 2,                   # polynomial order of the vectorizer
    "regularization": 0.0,        # L1 regularization strength (0 = none)
    "train_frac": 1.0,            # fraction of eligible stars used for training
    "quality_cut": True,          # apply the spectrum_flags / warn_* cuts
    "filters": None,              # row filters that gate the training set only
    "seed": 888,                  # RNG seed for the training-set draw
}

# Columns that, if present, are copied into the catalogue to identify each star.
ID_COLUMN_CANDIDATES = ("apogee_id", "APOGEE_ID", "source_id", "gaia_source_id",
                        "star_id", "id", "obj", "objid")


def load_config(path):
    """
    Return the run config: :data:`DEFAULT_CONFIG` updated with the JSON file at
    ``path`` (if given). Unknown keys are rejected so typos fail loudly.
    """
    config = dict(DEFAULT_CONFIG)
    if path:
        with open(path) as fp:
            user = json.load(fp)
        unknown = set(user) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError("unknown config key(s): {0}; valid keys are {1}"
                             .format(", ".join(sorted(unknown)),
                                     ", ".join(sorted(DEFAULT_CONFIG))))
        config.update(user)
    return config


def select_training_set(label_array, eligible, train_frac, seed):
    """
    Return a boolean mask over the sample selecting the training stars: those
    with all-finite labels, flagged ``eligible`` (e.g. by the quality cut), and
    kept by a reproducible uniform draw at rate ``train_frac``.
    """
    finite = np.isfinite(label_array).all(axis=1)
    pool = finite & eligible

    if train_frac >= 1.0:
        train_set = pool
    else:
        rng = np.random.RandomState(seed)
        u = rng.random_sample(len(label_array))
        train_set = pool & (u < train_frac)

    logger.info("training on %d stars (%d finite-label, %d eligible, "
                "%d in sample)", int(train_set.sum()), int(finite.sum()),
                int(pool.sum()), len(label_array))
    return train_set


def find_id_column(label_source):
    """ Return the first recognised identifier column name, or ``None``. """
    columns = getattr(label_source, "columns", None)
    if columns is None:
        return None
    for name in ID_COLUMN_CANDIDATES:
        if name in columns:
            return name
    return None


def train_and_apply(label_source, normalized_flux, normalized_ivar, dispersion,
    config, output_dir=".", save_model=None, test_batch_size=None):
    """
    Train a CannonModel per ``config`` and apply it to every star in the
    sample. Writes the trained model (if ``save_model``) and a per-star
    predictions catalogue, and returns ``(model, predictions_dataframe)``.
    """
    import pandas as pd

    label_names = list(config["labels"])
    label_array = np.vstack(
        [np.asarray(label_source[name], dtype=float) for name in label_names]).T

    if config["quality_cut"]:
        eligible = np.asarray(quality_mask(label_source), dtype=bool)
    else:
        eligible = np.ones(len(label_array), dtype=bool)

    # Row filters (e.g. snr>100) gate the TRAINING set only; the trained model
    # is still applied to every star in the sample below.
    if config.get("filters"):
        eligible = eligible & filter_mask(
            label_source, config["filters"], log=logger)

    # Per-label eligibility for the fitted labels: age/mass reliability
    # (RelAge_*) and abundance flags (X_FE_FLAG==0). Also training-set only.
    row_mask = label_set_row_mask(label_names, per_label_masks(label_source))
    if row_mask is not None:
        eligible = eligible & np.asarray(row_mask, dtype=bool)

    train_set = select_training_set(
        label_array, eligible, config["train_frac"], config["seed"])
    if train_set.sum() == 0:
        raise ValueError("empty training set; relax the quality cut, raise "
                         "train_frac, or check the label columns")

    # --- Train -----------------------------------------------------------
    vectorizer = tc.vectorizer.PolynomialVectorizer(
        label_names=label_names, order=config["order"])
    model = tc.CannonModel(
        label_array[train_set], normalized_flux[train_set],
        normalized_ivar[train_set], vectorizer, dispersion=dispersion,
        regularization=config["regularization"])
    logger.info("%s", model)

    model.train()
    logger.info("trained: %s | theta shape: %s", model.is_trained,
                np.asarray(model.theta).shape)

    # --- Apply to the whole sample --------------------------------------
    logger.info("applying the trained model to all %d stars", len(label_array))
    predicted, cov, meta = model.test(
        normalized_flux, normalized_ivar, batch_size=test_batch_size)
    predicted = np.asarray(predicted)
    cov = np.asarray(cov)
    # Formal per-label uncertainties are the sqrt of the covariance diagonal.
    # `test` returns `cov` in the *scaled* label basis (fitting.py optimizes
    # scaled params) while `predicted` is in physical units, so rescale the
    # errors to physical units too: sigma_phys = scales * sigma_scaled.
    scales = np.asarray(model._scales)                     # (L,)
    sigma = np.sqrt(np.clip(np.einsum("sii->si", cov), 0, None)) * scales

    # --- Assemble the catalogue -----------------------------------------
    columns = {}
    id_column = find_id_column(label_source)
    if id_column is not None:
        columns[id_column] = np.asarray(label_source[id_column])
        logger.info("using '%s' as the star identifier column", id_column)
    else:
        columns["star_index"] = np.arange(len(label_array))
        logger.warning("no identifier column found; using row index instead")

    columns["in_training_set"] = train_set
    for i, name in enumerate(label_names):
        columns["{0}_cannon".format(name)] = predicted[:, i]
        columns["{0}_cannon_err".format(name)] = sigma[:, i]
        columns["{0}_reference".format(name)] = label_array[:, i]
    columns["chi_sq"] = np.array([m["chi_sq"] for m in meta])
    columns["r_chi_sq"] = np.array([m["r_chi_sq"] for m in meta])

    predictions = pd.DataFrame(columns)

    # --- Write outputs ---------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "sample_predictions.csv")
    predictions.to_csv(csv_path, index=False)
    logger.info("wrote %d-star catalogue to %s", len(predictions), csv_path)

    if save_model:
        model.write(save_model, include_training_set_spectra=False,
                    overwrite=True)
        logger.info("wrote model to %s", save_model)

    _report(predictions, label_names)
    return model, predictions


def _report(predictions, label_names):
    """ Print a short summary of the catalogue (training vs. all-sample). """
    train_set = predictions["in_training_set"].to_numpy(dtype=bool)
    n = len(predictions)
    print("\n=== applied to {0} stars ({1} in training set) ===".format(
        n, int(train_set.sum())))
    for name in label_names:
        cannon = predictions["{0}_cannon".format(name)].to_numpy()
        ref = predictions["{0}_reference".format(name)].to_numpy()
        d = (cannon - ref)
        good = np.isfinite(d)
        if good.any():
            print("  {0:<12s} train-set bias={1:+.4f}  scatter={2:.4f}".format(
                name, float(np.mean(d[good & train_set]))
                if (good & train_set).any() else float("nan"),
                float(np.std(d[good & train_set]))
                if (good & train_set).any() else float("nan")))


# --------------------------------------------------------------------------- #
#  Entry points                                                               #
# --------------------------------------------------------------------------- #

def _run_real(args, config):
    label_source, dispersion, flux, ivar = load_spectra(args.spectra)
    normalized_flux, normalized_ivar = normalize_spectra(
        dispersion, flux, ivar, args.continuum_list)
    return train_and_apply(
        label_source, normalized_flux, normalized_ivar, dispersion, config,
        output_dir=args.output_dir, save_model=args.save_model,
        test_batch_size=args.test_batch_size)


def _run_demo(args, config):
    """ Smoke test on the bundled (already-normalized) golden data. """
    import pandas as pd
    try:
        from scripts.sweep_config import load_golden
    except ImportError:
        from sweep_config import load_golden

    names, labels, dispersion, flux, ivar = load_golden()
    label_source = pd.DataFrame(labels)

    # The golden set has no quality/filter columns; train on all of it.
    config = dict(config, labels=names, quality_cut=False, train_frac=1.0,
                  filters=None)
    print("Demo on golden data: {0} stars, {1} pixels, labels {2}".format(
        flux.shape[0], flux.shape[1], names))

    return train_and_apply(
        label_source, flux, ivar, dispersion, config,
        output_dir=args.output_dir, save_model=args.save_model,
        test_batch_size=args.test_batch_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spectra", default=DEFAULT_SPECTRA,
                        help="parquet table of spectra + labels")
    parser.add_argument("--continuum-list", default=DEFAULT_CONTINUUM_LIST,
                        help="text file of continuum pixel indices")
    parser.add_argument("--config", default=None,
                        help="JSON config file (labels, order, regularization, "
                             "train_frac, quality_cut, filters, seed)")
    parser.add_argument("--labels", type=lambda s: s.split(","), default=None,
                        help="comma-separated label columns; overrides --config")
    add_filter_arg(parser)
    parser.add_argument("--order", type=int, default=None,
                        help="polynomial order; overrides --config")
    parser.add_argument("--regularization", type=float, default=None,
                        help="L1 regularization strength; overrides --config")
    parser.add_argument("--train-frac", type=float, default=None,
                        help="fraction of eligible stars to train on; "
                             "overrides --config (1.0 = all)")
    parser.add_argument("--output-dir", default=".",
                        help="directory for the model and predictions catalogue")
    parser.add_argument("--save-model", default=None,
                        help="path to write the trained .model file")
    parser.add_argument("--test-batch-size", type=int, default=None,
                        help="spectra fit per batch in the apply step; lower it "
                             "if the device OOMs (default: memory-aware auto)")
    parser.add_argument("--demo", action="store_true",
                        help="run on the bundled golden data instead")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable INFO-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    config = load_config(args.config)
    # CLI flags override the config file where provided.
    if args.labels is not None:
        config["labels"] = args.labels
    if args.order is not None:
        config["order"] = args.order
    if args.regularization is not None:
        config["regularization"] = args.regularization
    if args.train_frac is not None:
        config["train_frac"] = args.train_frac
    if args.filters is not None:
        config["filters"] = args.filters
    logger.info("run config: %s", config)

    if args.demo:
        _run_demo(args, config)
    else:
        _run_real(args, config)


if __name__ == "__main__":
    main()
