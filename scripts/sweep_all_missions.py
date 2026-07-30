#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-validated (order, regularization) sweep for the all-missions RGB
one-step model -- the batch counterpart of the RUN_SWEEP cell in
``notebooks/train_rgb_wilett_all_missions.ipynb``.

The training sample is rebuilt exactly as in that notebook: finite Willett
age -> derived [X/Fe] and [C/N] columns -> sentinel / RelAge_L /
young-alpha-rich-impostor cuts -> seismic logg from the nu_max scaling
relation (the Cannon logg label) -> continuum normalization -> RGB selection
(seismic EvoState where present, gradient-boosting classifier at
p > --min-proba elsewhere) -> finite-label cut -> data-driven mg_fe censor
windows. Every (order, regularization, censoring) grid point is then scored
with :func:`scripts.sweep_cannon.cross_validate` on the same folds and seed
(888) as the notebook, so rows are directly comparable with the in-notebook
k-fold numbers.

Grid points run cheapest first (low order, closed-form reg=0 before the
iterative L1 path) and every finished row is appended to ``--output``
immediately, so a walltime kill keeps all completed rows.

JAX device selection is via the ``JAX_PLATFORMS`` environment variable; the
PBS wrapper (``scripts/sweep_all_missions.pbs``) aborts if no GPU backend is
available.

Usage
-----
::

    python -m scripts.sweep_all_missions \\
        --spectra /path/merged_with_ages_raw.parquet \\
        --continuum-list /path/continuum.list \\
        --orders 2,3,4 --regularizations 0,10,100,1000,10000 \\
        --output results/sweep_all_missions.csv -v
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import itertools
import logging
import os
import time

import numpy as np

# Work both as a package module (`python -m scripts.sweep_all_missions`) and
# when run directly from the scripts/ directory.
try:
    from scripts.train_cannon import (add_x_fe_columns, normalize_spectra,
                                       _to_array, DEFAULT_SPECTRA,
                                       DEFAULT_CONTINUUM_LIST)
    from scripts.apply_rgb_ages import (RGB, fit_state_classifier,
                                         select_state)
    from scripts.sweep_cannon import cross_validate, _summarize
except ImportError:
    from train_cannon import (add_x_fe_columns, normalize_spectra, _to_array,
                              DEFAULT_SPECTRA, DEFAULT_CONTINUUM_LIST)
    from apply_rgb_ages import RGB, fit_state_classifier, select_state
    from sweep_cannon import cross_validate, _summarize

logger = logging.getLogger("thecannon.sweep_all_missions")

# Label vector of the notebook's current configuration (INCLUDE_C_N=True,
# INCLUDE_AL=True; c_fe/n_fe/o_fe/ce_fe/nd_fe off).
LABEL_COLS = ["raw_teff", "logg_seismic", "raw_fe_h", "mg_fe", "c_n",
              "al_fe", "log_age_L"]

# Training-set quality cuts (see the notebook for the full rationale).
SENTINEL_COLS = ["raw_teff", "raw_logg", "raw_fe_h", "raw_mg_h",
                 "raw_c_h", "raw_n_h", "raw_al_h"]

# nu_max scaling relation constants (seismic logg label).
NUMAX_SUN, TEFF_SUN, LOGG_SUN = 3090.0, 5777.0, 4.438

# Data-driven censor windows for narrow-line abundance labels (C/N labels
# deliberately uncensored -- see the notebook's censor cell).
CENSORED_LABELS = ("mg_fe",)
KEEP_FRAC = 0.05
DILATE = 2

CENSOR_MODES = ("censored", "none")


def build_sample(spectra_path, continuum_list_path, min_proba):
    """
    Rebuild the notebook's RGB training sample and return
    ``(labels_f, flux_f, ivar_f, dispersion)``: the finite-label RGB stars'
    label DataFrame (columns = ``LABEL_COLS``) and their normalized spectra.
    """
    import pandas as pd

    spectra = pd.read_parquet(spectra_path)
    spectra = spectra[np.isfinite(
        spectra["age_L"].to_numpy(dtype=float))].reset_index(drop=True)
    logger.info("finite Willett age: %d stars\n%s", len(spectra),
                spectra["source"].value_counts().to_string())

    add_x_fe_columns(spectra)
    spectra["c_n"] = spectra["raw_c_h"] - spectra["raw_n_h"]
    spectra["log_age_L"] = np.log10(spectra["age_L"].to_numpy(dtype=float))

    # 1) ASPCAP -999.999 sentinel rows (isfinite does not catch them);
    # 2) Willett age-reliability flag; 3) young alpha-rich impostors
    # (mass-transfer/merger products with wrong reference ages).
    sentinel = (spectra[SENTINEL_COLS] < -100).any(axis=1)
    unreliable_age = ~spectra["RelAge_L"].astype(bool)
    impostor = ((spectra["age_L"] < 4) & (spectra["mg_fe"] > 0.15)
                & (spectra["c_n"] > -0.2))
    drop = sentinel | unreliable_age | impostor
    logger.info("cuts: %d sentinel, %d RelAge_L=False, %d impostors "
                "(overlapping); dropping %d of %d",
                int(sentinel.sum()), int(unreliable_age.sum()),
                int(impostor.sum()), int(drop.sum()), len(spectra))
    spectra = spectra[~drop].reset_index(drop=True)

    # Seismic logg from the nu_max scaling relation -- the Cannon logg label.
    numax = spectra["numax"].to_numpy(dtype=float)
    teff = spectra["raw_teff"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        spectra["logg_seismic"] = (LOGG_SUN + np.log10(numax / NUMAX_SUN)
                                   + 0.5 * np.log10(teff / TEFF_SUN))

    # Stack the per-row spectrum arrays, dropping rows on a different grid.
    dispersion = _to_array(spectra["wavelength"].iloc[0])
    flux_rows = [_to_array(x) for x in spectra["flux"]]
    ivar_rows = [_to_array(x) for x in spectra["ivar"]]
    ok = np.array([f.size == dispersion.size and v.size == dispersion.size
                   for f, v in zip(flux_rows, ivar_rows)])
    if not ok.all():
        logger.warning("dropping %d spectra whose flux/ivar size does not "
                       "match the dispersion grid", int((~ok).sum()))
        spectra = spectra.loc[ok].reset_index(drop=True)
        flux_rows = list(itertools.compress(flux_rows, ok))
        ivar_rows = list(itertools.compress(ivar_rows, ok))
    flux = np.vstack(flux_rows)
    ivar = np.vstack(ivar_rows)
    assert flux.shape == ivar.shape == (len(spectra), dispersion.size)
    assert np.all(np.diff(dispersion) > 0)

    normalized_flux, normalized_ivar = normalize_spectra(
        dispersion, flux, ivar, continuum_list_path)
    nf = np.asarray(normalized_flux)        # sklearn wants host numpy
    ni = np.asarray(normalized_ivar)

    # RGB selection: seismic EvoState (Kepler), classifier for the rest.
    # The classifier reads EVO_FEATURES['raw_logg']; swap in the seismic
    # logg on a copy so `spectra` keeps both columns.
    clf_table = spectra.copy()
    clf_table["raw_logg"] = clf_table["logg_seismic"]
    classifier = fit_state_classifier(clf_table, nf, ni, features="both",
                                      min_proba=min_proba, target=RGB)
    is_rgb, source, _ = select_state(clf_table, classifier, min_proba,
                                     nf, ni, target=RGB)
    keep_idx = np.where(is_rgb)[0]
    spectra = spectra.iloc[keep_idx].reset_index(drop=True)
    nf, ni = nf[keep_idx], ni[keep_idx]
    logger.info("%d RGB stars (%d seismic, %d classified)", len(spectra),
                int((source[keep_idx] == "seismic").sum()),
                int((source[keep_idx] == "classified").sum()))

    labels = spectra[LABEL_COLS]
    finite = np.isfinite(labels.values).all(axis=1)
    labels_f = labels.loc[finite].reset_index(drop=True)
    logger.info("%d stars with finite labels enter the sweep (%d dropped)",
                int(finite.sum()), int((~finite).sum()))
    return labels_f, nf[finite], ni[finite], dispersion


def build_censors(labels_f, flux_f, ivar_f, dispersion):
    """
    The notebook's data-driven censor windows: a first-order uncensored probe
    model picks, for each censored label, the ``KEEP_FRAC`` of well-constrained
    pixels where its |linear coefficient| is largest, dilated by ``DILATE``
    pixels for line wings.
    """
    import thecannon as tc
    from thecannon.censoring import Censors

    label_names = list(labels_f.columns)
    good_pix = (np.asarray(ivar_f) > 0).mean(axis=0) > 0.8
    probe = tc.CannonModel(
        labels_f, flux_f, ivar_f,
        tc.vectorizer.PolynomialVectorizer(label_names=label_names, order=1),
        dispersion=dispersion)
    probe.train(progressbar=False)
    probe_theta = np.asarray(probe.theta)

    def censor_mask(label):
        coef = np.abs(probe_theta[:, 1 + label_names.index(label)])
        coef = np.where(good_pix, coef, 0.0)
        keep = coef >= np.quantile(coef, 1.0 - KEEP_FRAC)
        for _ in range(DILATE):
            keep = keep | np.roll(keep, 1) | np.roll(keep, -1)
        return ~keep

    to_censor = [l for l in CENSORED_LABELS if l in label_names]
    censors = Censors(label_names, int(dispersion.size),
                      {l: censor_mask(l) for l in to_censor})
    for l in to_censor:
        logger.info("censor %s: %d/%d pixels retained", l,
                    int((~censors[l]).sum()), dispersion.size)
    return censors


def parse_floats(text):
    return [float(x) for x in text.split(",") if x.strip() != ""]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spectra", default=DEFAULT_SPECTRA,
                        help="parquet with the all-missions seismic sample")
    parser.add_argument("--continuum-list", default=DEFAULT_CONTINUUM_LIST,
                        help="text file of continuum pixel indices")
    parser.add_argument("--orders", default="2,3,4",
                        help="comma list of polynomial orders (default: 2,3,4)")
    parser.add_argument("--regularizations", default="0,10,100,1000,10000",
                        help="comma list of L1 strengths; 0 uses the "
                             "closed-form fast path "
                             "(default: 0,10,100,1000,10000)")
    parser.add_argument("--censor-modes", default="censored",
                        help="comma list from {censored,none}: run each grid "
                             "point with the mg_fe censor windows, without "
                             "them, or both (default: censored)")
    parser.add_argument("--min-proba", type=float, default=0.9,
                        help="classifier acceptance threshold for the RGB "
                             "selection (default: 0.9)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="cross-validation folds (default: 5)")
    parser.add_argument("--seed", type=int, default=888,
                        help="fold seed; 888 matches the notebook")
    parser.add_argument("--test-batch-size", type=int, default=32,
                        help="test-step batch size (default: 32)")
    parser.add_argument("--output", default="results/sweep_all_missions.csv",
                        help="CSV appended to after every grid point")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable INFO-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    import pandas as pd

    orders = [int(o) for o in parse_floats(args.orders)]
    regularizations = parse_floats(args.regularizations)
    censor_modes = [m.strip() for m in args.censor_modes.split(",")
                    if m.strip()]
    unknown = set(censor_modes) - set(CENSOR_MODES)
    if unknown:
        raise ValueError("unknown censor mode(s): {0}".format(
            ", ".join(sorted(unknown))))

    labels_f, flux_f, ivar_f, dispersion = build_sample(
        args.spectra, args.continuum_list, args.min_proba)
    label_names = list(labels_f.columns)
    truth = labels_f.values

    censors = (build_censors(labels_f, flux_f, ivar_f, dispersion)
               if "censored" in censor_modes else None)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # Cheapest first: within an order, reg=0 (closed-form) precedes the
    # iterative L1 strengths; the order-4 L1 rows land last.
    grid = sorted(itertools.product(orders, regularizations, censor_modes))
    logger.info("sweeping %d grid points x %d folds on %d stars, %d labels",
                len(grid), args.n_splits, len(truth), len(label_names))

    for i, (order, reg, mode) in enumerate(grid):
        tick = time.time()
        cv = cross_validate(
            truth, flux_f, ivar_f, np.asarray(dispersion), label_names,
            order, reg, censors=censors if mode == "censored" else None,
            n_splits=args.n_splits, seed=args.seed,
            test_kwds={"batch_size": args.test_batch_size})
        row = {"order": order, "regularization": reg, "censor": mode,
               "n_stars": len(truth), "n_splits": args.n_splits,
               "seed": args.seed, "elapsed_s": round(time.time() - tick, 1)}
        row.update(_summarize(cv, label_names))
        pd.DataFrame([row]).to_csv(
            args.output, mode="a", index=False,
            header=not os.path.exists(args.output))
        print("[{0}/{1}] order={2} reg={3:g} censor={4}: "
              "sigma_mad_log_age_L={5:.4f} mean_r2={6:.4f} ({7:.0f} s)"
              .format(i + 1, len(grid), order, reg, mode,
                      row.get("sigma_mad_log_age_L", float("nan")),
                      row.get("mean_r2", float("nan")), row["elapsed_s"]),
              flush=True)

    print("sweep complete: {0} rows appended to {1}".format(
        len(grid), args.output))


if __name__ == "__main__":
    main()
