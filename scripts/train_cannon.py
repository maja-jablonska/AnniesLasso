#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train and validate a single CannonModel end-to-end (from notebooks/start.ipynb).

The workflow:

  1. load a parquet table of APOGEE spectra + ASPCAP labels,
  2. assemble the (n_pixels,) dispersion and (n_stars, n_pixels) flux / ivar
     arrays (each table cell holds a per-star array),
  3. pseudo-continuum-normalize the spectra,
  4. split into training / validation sets (dropping non-finite labels),
  5. fit a polynomial CannonModel on the training set,
  6. run the test step on the validation set, and
  7. write a one-to-one (Cannon vs reference) figure, a predictions CSV, and
     optionally the trained model.

JAX device selection is via the ``JAX_PLATFORMS`` environment variable (this
script does not force one), e.g. ``JAX_PLATFORMS=cpu`` on machines where the GPU
backend is unavailable.

Usage
-----
On the real data (defaults point at the bulge-ages-and-orbits data set)::

    python -m scripts.train_cannon \\
        --spectra /path/to/cleaned_ages.parquet \\
        --continuum-list /path/to/continuum.list \\
        --labels raw_teff,raw_logg,raw_fe_h,raw_mg_h,raw_ce_h,age_L,mass_L \\
        --order 2 --output-dir results/

Quick end-to-end check on the bundled golden data::

    JAX_PLATFORMS=cpu python -m scripts.train_cannon --demo
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import logging
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")              # headless: only saves figures
import matplotlib.pyplot as plt

import thecannon as tc
from thecannon import continuum

logger = logging.getLogger("thecannon.train")


# Defaults mirroring notebooks/start.ipynb.
DEFAULT_DATA_DIR = "/home/100/mj8805/scr_mk27/bulge-ages-and-orbits/data"
DEFAULT_LABELS = ["raw_teff", "raw_logg", "raw_fe_h", "raw_mg_h", "raw_ce_h",
                  "age_L", "mass_L"]
APOGEE_REGIONS = ([15090, 15822], [15823, 16451], [16452, 16971])
CONTINUUM_L = 1400
CONTINUUM_ORDER = 3


# --------------------------------------------------------------------------- #
#  Data loading                                                                #
# --------------------------------------------------------------------------- #

def _to_array(x):
    """ Coerce a table cell (array, list, or stringified list) to a 1-D array. """
    if isinstance(x, str):
        return np.fromstring(x.strip("[] \n"), sep=",")
    return np.asarray(x, dtype=float)


def add_x_fe_columns(table):
    """
    Derive [X/Fe] abundance columns from the raw [X/H] ones: every
    ``raw_<x>_h`` column except iron itself gains a ``<x>_fe`` counterpart
    equal to ``raw_<x>_h - raw_fe_h``. Existing columns are never overwritten,
    and the table is returned (modified in place) for convenience.
    """
    import re

    if "raw_fe_h" not in table.columns:
        logger.warning("no raw_fe_h column; cannot derive any [X/Fe] columns")
        return table

    fe_h = np.asarray(table["raw_fe_h"], dtype=float)
    derived = []
    for column in list(table.columns):
        match = re.fullmatch(r"raw_([a-z0-9]+)_h", str(column))
        if match is None or match.group(1) == "fe":
            continue
        name = "{0}_fe".format(match.group(1))
        if name in table.columns:
            continue
        table[name] = np.asarray(table[column], dtype=float) - fe_h
        derived.append(name)

    if derived:
        logger.info("derived [X/Fe] columns: %s", ", ".join(derived))
    return table


def quality_mask(table):
    """
    Boolean mask over the rows of ``table`` selecting stars that pass the
    quality cuts: ``spectrum_flags == 0`` (no flagged reduction/calibration
    issues) and no ``warn_*`` column set to True. Missing columns skip the
    corresponding cut with a warning rather than rejecting everything.
    """
    n = len(table)
    mask = np.ones(n, dtype=bool)

    if "spectrum_flags" in table.columns:
        clean = np.asarray(table["spectrum_flags"].fillna(-1) == 0)
        logger.info("quality cut: %d/%d stars rejected by spectrum_flags != 0",
                    int((~clean).sum()), n)
        mask &= clean
    else:
        logger.warning("no spectrum_flags column; skipping that quality cut")

    warn_columns = [c for c in table.columns if str(c).startswith("warn_")]
    if warn_columns:
        for column in warn_columns:
            mask &= ~np.asarray(table[column].fillna(False), dtype=bool)
        logger.info("quality cut: %d/%d stars left after rejecting any of "
                    "%s set", int(mask.sum()), n, ", ".join(warn_columns))
    else:
        logger.warning("no warn_* columns; skipping that quality cut")

    return mask


def load_spectra(spectra_path):
    """
    Read the parquet table and assemble ``(spectra, dispersion, flux, ivar)``.
    Each of the ``wavelength``/``flux``/``ivar`` columns holds one array per
    row. Derived ``<x>_fe`` abundance columns are added alongside the raw
    ``raw_<x>_h`` ones (see :func:`add_x_fe_columns`).
    """
    import pandas as pd

    spectra = pd.read_parquet(spectra_path)
    add_x_fe_columns(spectra)

    dispersion = _to_array(spectra["wavelength"].iloc[0])         # (n_pixels,)
    flux = np.vstack([_to_array(x) for x in spectra["flux"]])     # (n_stars, P)
    ivar = np.vstack([_to_array(x) for x in spectra["ivar"]])     # (n_stars, P)

    assert dispersion.ndim == 1, dispersion.shape
    assert flux.shape == ivar.shape == (len(spectra), dispersion.size)
    assert np.all(np.diff(dispersion) > 0), "dispersion must be sorted ascending"
    logger.info("loaded %d stars x %d pixels", *flux.shape)
    return spectra, dispersion, flux, ivar


def normalize_spectra(dispersion, flux, ivar, continuum_list_path):
    """ Pseudo-continuum-normalize the spectra over the APOGEE chip regions. """
    continuum_pixels = np.loadtxt(
        continuum_list_path, dtype=int, comments="#")
    continuum_pixels = continuum_pixels[continuum_pixels < dispersion.size]

    normalized_flux, normalized_ivar, _, _ = continuum.normalize(
        dispersion, flux, ivar, continuum_pixels,
        L=CONTINUUM_L, order=CONTINUUM_ORDER, regions=APOGEE_REGIONS)

    good = ivar > 0
    logger.info("median normalized flux on good pixels: %.4f (target ~1)",
                np.median(normalized_flux[good]))
    return normalized_flux, normalized_ivar


# --------------------------------------------------------------------------- #
#  Train / validate split                                                      #
# --------------------------------------------------------------------------- #

def make_split(label_array, train_frac, validate_frac, seed):
    """
    Return boolean ``(train_set, validate_set)`` masks. Stars with any
    non-finite label are excluded from both. A single shuffled uniform draw
    assigns each finite star to the training fold, the validation fold, or
    neither (reproducible via ``seed``).
    """
    finite = np.isfinite(label_array).all(axis=1)
    rng = np.random.RandomState(seed)
    u = rng.random_sample(len(label_array))

    train_set = finite & (u < train_frac)
    validate_set = finite & (u >= train_frac) & (u < train_frac + validate_frac)
    logger.info("%d training, %d validation stars (%d dropped, non-finite)",
                train_set.sum(), validate_set.sum(), (~finite).sum())
    return train_set, validate_set


# --------------------------------------------------------------------------- #
#  One-to-one figure                                                           #
# --------------------------------------------------------------------------- #

def one_to_one_figure(truth, predicted, label_names, title=None):
    """ Cannon-vs-reference scatter per label, annotated with bias and scatter. """
    K = len(label_names)
    ncols = 2
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)
    axes = axes.ravel()

    for i, name in enumerate(label_names):
        ax = axes[i]
        x, y = truth[:, i], predicted[:, i]
        good = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[good], y[good], facecolor="k", s=10, alpha=0.5)
        if good.any():
            lims = [float(min(x[good].min(), y[good].min())),
                    float(max(x[good].max(), y[good].max()))]
        else:
            lims = [0.0, 1.0]
        ax.plot(lims, lims, "-", color="r", lw=1)        # 1:1 line
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        d = (y - x)[good]
        bias = float(np.nanmean(d)) if d.size else float("nan")
        scatter = float(np.nanstd(d)) if d.size else float("nan")
        ax.set_xlabel("{0} (reference)".format(name))
        ax.set_ylabel("{0} (Cannon)".format(name))
        ax.set_title("{0}: bias={1:+.3f}, scatter={2:.3f}".format(
            name, bias, scatter))

    for j in range(K, len(axes)):
        axes[j].set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Orchestration                                                               #
# --------------------------------------------------------------------------- #

def run(labels, normalized_flux, normalized_ivar, dispersion, label_names,
    order=2, regularization=0, train_frac=0.1, validate_frac=0.1, seed=888,
    output_dir=".", save_model=None, test_batch_size=None):
    """
    Train on the training split, test on the validation split, and write the
    one-to-one figure + predictions CSV. ``labels`` is anything CannonModel
    accepts (a DataFrame, structured array, or (N, K) array aligned with
    ``label_names``).
    """
    import pandas as pd

    label_array = np.vstack(
        [np.asarray(labels[name], dtype=float) for name in label_names]).T

    train_set, validate_set = make_split(
        label_array, train_frac, validate_frac, seed)
    if train_set.sum() == 0 or validate_set.sum() == 0:
        raise ValueError("empty training or validation set; adjust the "
                         "train/validate fractions or seed")

    vectorizer = tc.vectorizer.PolynomialVectorizer(
        label_names=label_names, order=order)
    model = tc.CannonModel(
        label_array[train_set], normalized_flux[train_set],
        normalized_ivar[train_set], vectorizer, dispersion=dispersion,
        regularization=regularization)
    logger.info("%s", model)

    model.train()
    logger.info("trained: %s | theta shape: %s", model.is_trained,
                np.asarray(model.theta).shape)

    predicted, _, _ = model.test(
        normalized_flux[validate_set], normalized_ivar[validate_set],
        batch_size=test_batch_size)
    predicted = np.asarray(predicted)
    truth = label_array[validate_set]

    # Per-label metrics.
    residual = predicted - truth
    print("\n=== validation metrics ({0} stars) ===".format(len(truth)))
    for i, name in enumerate(label_names):
        d = residual[:, i][np.isfinite(residual[:, i])]
        print("  {0:<12s} bias={1:+.4f}  scatter={2:.4f}".format(
            name, float(np.mean(d)), float(np.std(d))))

    os.makedirs(output_dir, exist_ok=True)

    fig = one_to_one_figure(
        truth, predicted, label_names,
        title="order={0}, reg={1:g}".format(order, regularization))
    fig_path = os.path.join(output_dir, "one_to_one.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", fig_path)

    pred_cols = {"{0}_reference".format(n): truth[:, i]
                 for i, n in enumerate(label_names)}
    pred_cols.update({"{0}_cannon".format(n): predicted[:, i]
                      for i, n in enumerate(label_names)})
    csv_path = os.path.join(output_dir, "validation_predictions.csv")
    pd.DataFrame(pred_cols).to_csv(csv_path, index=False)
    logger.info("wrote %s", csv_path)

    if save_model:
        model.write(save_model, include_training_set_spectra=False,
                    overwrite=True)
        logger.info("wrote model to %s", save_model)

    return model, truth, predicted


def _run_real(args):
    spectra, dispersion, flux, ivar = load_spectra(args.spectra)
    normalized_flux, normalized_ivar = normalize_spectra(
        dispersion, flux, ivar, args.continuum_list)
    return run(
        spectra, normalized_flux, normalized_ivar, dispersion, args.labels,
        order=args.order, regularization=args.regularization,
        train_frac=args.train_frac, validate_frac=args.validate_frac,
        seed=args.seed, output_dir=args.output_dir, save_model=args.save_model,
        test_batch_size=args.test_batch_size)


def _run_demo(args):
    """ Smoke test on the bundled (already-normalized) golden data. """
    try:
        from scripts.sweep_config import load_golden
    except ImportError:
        from sweep_config import load_golden

    names, labels, dispersion, flux, ivar = load_golden()
    print("Demo on golden data: {0} stars, {1} pixels, labels {2}".format(
        flux.shape[0], flux.shape[1], names))

    # Golden flux is already normalized; use a 50/50 split given the small N.
    return run(
        labels, flux, ivar, dispersion, names,
        order=args.order, regularization=args.regularization,
        train_frac=0.5, validate_frac=0.5, seed=args.seed,
        output_dir=args.output_dir, save_model=args.save_model,
        test_batch_size=args.test_batch_size)


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
                        help="comma-separated label column names")
    parser.add_argument("--order", type=int, default=2,
                        help="polynomial order of the vectorizer")
    parser.add_argument("--regularization", type=float, default=0.0,
                        help="L1 regularization strength (0 = none)")
    parser.add_argument("--train-frac", type=float, default=0.1,
                        help="fraction of finite-label stars used for training")
    parser.add_argument("--validate-frac", type=float, default=0.1,
                        help="fraction used for validation")
    parser.add_argument("--seed", type=int, default=888,
                        help="RNG seed for the split")
    parser.add_argument("--output-dir", default=".",
                        help="directory for the figure and predictions CSV")
    parser.add_argument("--save-model", default=None,
                        help="optional path to write the trained .model file")
    parser.add_argument("--test-batch-size", type=int, default=None,
                        help="spectra fit per batch in the test step; lower it "
                             "if the device OOMs (default: memory-aware auto)")
    parser.add_argument("--demo", action="store_true",
                        help="run on the bundled golden data instead")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable INFO-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.demo:
        _run_demo(args)
    else:
        _run_real(args)


if __name__ == "__main__":
    main()
