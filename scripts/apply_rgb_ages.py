#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Apply a saved CannonModel to the RGB stars of a sample and estimate their ages.

Unlike :mod:`scripts.apply_cannon` (which trains a model first), this script
loads an already-trained pickled model, selects the first-ascent RGB stars,
runs the test step on them only, and writes a per-star age catalogue.

RGB selection uses the seismic evolutionary state where available
(``EvoState`` == 1) and falls back to a gradient-boosting classifier in
Teff / log g / [Fe/H] / [C/Fe] / [N/Fe] / [Mg/Fe] space for stars without one,
accepting only predictions with probability above ``--min-proba``. The
classifier is trained on the seismically labeled rows of ``--classifier-train``
(default: the input sample itself, so a mixed seismic+unknown table needs no
extra file; a pure target sample, e.g. bulge stars, should point this at the
seismic training parquet).

The spectra are pseudo-continuum-normalized with the same continuum pixel list
and chip regions used in training -- the model is only valid on spectra
normalized identically.

JAX device selection is via the ``JAX_PLATFORMS`` environment variable (this
script does not force one), e.g. ``JAX_PLATFORMS=cpu`` where the GPU backend is
unavailable.

Usage
-----
::

    python -m scripts.apply_rgb_ages \\
        --model results/trained_cannon_model.pkl \\
        --spectra /path/to/target_sample.parquet \\
        --continuum-list /path/to/continuum.list \\
        --classifier-train /path/to/merged_with_ages_raw.parquet \\
        --output results/rgb_ages.csv
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import logging
import os

import numpy as np

import thecannon as tc

# Work both as a package module (`python -m scripts.apply_rgb_ages`) and when
# run directly from the scripts/ directory.
try:
    from scripts.train_cannon import (load_spectra, normalize_spectra,
                                       quality_mask, add_x_fe_columns,
                                       DEFAULT_SPECTRA, DEFAULT_CONTINUUM_LIST)
    from scripts.apply_cannon import find_id_column
except ImportError:
    from train_cannon import (load_spectra, normalize_spectra, quality_mask,
                              add_x_fe_columns, DEFAULT_SPECTRA,
                              DEFAULT_CONTINUUM_LIST)
    from apply_cannon import find_id_column

logger = logging.getLogger("thecannon.apply_rgb")

RGB, HEB = 1, 2
EVO_FEATURES = ["raw_teff", "raw_logg", "raw_fe_h", "c_fe", "n_fe", "mg_fe"]


def seismic_state(table):
    """ Numeric ``EvoState`` per row (NaN where absent or not RGB/HeB). """
    import pandas as pd
    if "EvoState" not in table.columns:
        return pd.Series(np.nan, index=table.index)
    state = pd.to_numeric(table["EvoState"], errors="coerce")
    return state.where(state.isin([RGB, HEB]))


def fit_state_classifier(train_table, min_labeled=200):
    """
    Fit the RGB/HeB classifier on the seismically labeled rows of
    ``train_table`` and return it (or ``None`` if too few labeled stars).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    state = seismic_state(train_table)
    labeled = state.notna()
    if labeled.sum() < min_labeled:
        logger.warning("only %d seismically labeled stars in the classifier "
                       "training table (< %d); skipping classification",
                       int(labeled.sum()), min_labeled)
        return None

    X = train_table.loc[labeled, EVO_FEATURES].to_numpy(dtype=float)
    y = state[labeled].to_numpy()
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    clf.fit(X, y)
    logger.info("state classifier trained on %d labeled stars "
                "(%d RGB, %d HeB)", len(y), int((y == RGB).sum()),
                int((y == HEB).sum()))
    return clf


def select_rgb(table, classifier, min_proba):
    """
    Return ``(is_rgb, source, rgb_proba)`` arrays over the rows of ``table``:
    a boolean RGB mask, how each star was identified (``"seismic"``,
    ``"classified"`` or ``""``), and the classifier RGB probability (NaN for
    seismically identified stars).
    """
    state = seismic_state(table)
    is_rgb = (state == RGB).to_numpy()
    source = np.where(state.notna(), "seismic", "")
    rgb_proba = np.full(len(table), np.nan)

    unknown = state.isna().to_numpy()
    if unknown.any() and classifier is not None:
        X = table.loc[unknown, EVO_FEATURES].to_numpy(dtype=float)
        proba = classifier.predict_proba(X)
        p_rgb = proba[:, list(classifier.classes_).index(RGB)]
        rgb_proba[unknown] = p_rgb
        accepted = p_rgb > min_proba
        is_rgb[unknown] = accepted
        source[unknown] = np.where(accepted, "classified", "")
        logger.info("classifier: %d/%d unlabeled stars accepted as RGB "
                    "(p > %.2f)", int(accepted.sum()), int(unknown.sum()),
                    min_proba)
    elif unknown.any():
        logger.warning("%d stars have no evolutionary state and no classifier "
                       "is available; they are dropped", int(unknown.sum()))

    logger.info("RGB selection: %d seismic + %d classified = %d of %d stars",
                int((state == RGB).sum()),
                int((source == "classified").sum()),
                int(is_rgb.sum()), len(table))
    return is_rgb, source, rgb_proba


def apply_model(model, table, normalized_flux, normalized_ivar, source,
    rgb_proba, test_batch_size=None):
    """
    Run the model's test step on the (already RGB-selected, normalized)
    spectra and return the per-star age catalogue as a DataFrame.
    """
    import pandas as pd

    label_names = list(model.vectorizer.label_names)
    missing = [n for n in ("log_age_Dnu",) if n not in label_names]
    if missing:
        raise ValueError("the model has no {0} label (labels: {1}); cannot "
                         "estimate ages".format(missing[0],
                                                ", ".join(label_names)))
    age_index = label_names.index("log_age_Dnu")

    predicted, cov, meta = model.test(
        normalized_flux, normalized_ivar, batch_size=test_batch_size)
    predicted = np.asarray(predicted)
    cov = np.asarray(cov)

    # `test` returns `cov` in the *scaled* label basis while `predicted` is in
    # physical units (see scripts.apply_cannon), so rescale the errors too.
    scales = np.asarray(model._scales)
    sigma = np.sqrt(np.clip(np.einsum("sii->si", cov), 0, None)) * scales

    columns = {}
    id_column = find_id_column(table)
    if id_column is not None:
        columns[id_column] = np.asarray(table[id_column])
        logger.info("using '%s' as the star identifier column", id_column)
    else:
        columns["star_index"] = np.asarray(table.index)
        logger.warning("no identifier column found; using row index instead")

    columns["evo_state_source"] = source
    columns["rgb_proba"] = rgb_proba
    for i, name in enumerate(label_names):
        columns["{0}_cannon".format(name)] = predicted[:, i]
        columns["{0}_cannon_err".format(name)] = sigma[:, i]

    # Linear age with the log-normal error propagated: for x = log10(age),
    # sigma_age = age * ln(10) * sigma_x.
    log_age, log_age_err = predicted[:, age_index], sigma[:, age_index]
    age_gyr = 10.0 ** log_age
    columns["age_gyr"] = age_gyr
    columns["age_gyr_err"] = age_gyr * np.log(10.0) * log_age_err
    columns["flag_unphysical_age"] = ~((age_gyr > 0) & (age_gyr < 20))

    columns["chi_sq"] = np.array([m["chi_sq"] for m in meta])
    columns["r_chi_sq"] = np.array([m["r_chi_sq"] for m in meta])
    return pd.DataFrame(columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="path to the saved (pickled) CannonModel")
    parser.add_argument("--spectra", default=DEFAULT_SPECTRA,
                        help="parquet table of spectra to estimate ages for")
    parser.add_argument("--continuum-list", default=DEFAULT_CONTINUUM_LIST,
                        help="text file of continuum pixel indices (must be "
                             "the one used to train the model)")
    parser.add_argument("--classifier-train", default=None,
                        help="parquet with seismic EvoState rows to train the "
                             "RGB classifier on (default: the input sample)")
    parser.add_argument("--min-proba", type=float, default=0.9,
                        help="classifier probability above which an unlabeled "
                             "star is accepted as RGB (default: 0.9)")
    parser.add_argument("--no-quality-cut", action="store_true",
                        help="skip the spectrum_flags / warn_* quality cuts")
    parser.add_argument("--output", default="rgb_ages.csv",
                        help="output catalogue path (.csv or .parquet)")
    parser.add_argument("--test-batch-size", type=int, default=None,
                        help="spectra fit per batch in the test step; lower it "
                             "if the device OOMs (default: memory-aware auto)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable INFO-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    model = tc.CannonModel.read(args.model)
    if not model.is_trained:
        raise ValueError("model at {0} is not trained".format(args.model))
    logger.info("loaded %s", model)

    table, dispersion, flux, ivar = load_spectra(args.spectra)
    if model.dispersion is not None \
    and np.asarray(model.dispersion).size != dispersion.size:
        raise ValueError(
            "sample has {0} pixels but the model was trained on {1}; the "
            "spectra are on a different wavelength grid".format(
                dispersion.size, np.asarray(model.dispersion).size))

    keep = np.ones(len(table), dtype=bool)
    if not args.no_quality_cut:
        keep &= np.asarray(quality_mask(table), dtype=bool)
        logger.info("quality cuts keep %d/%d stars", int(keep.sum()),
                    len(table))

    # RGB selection: seismic state first, classifier fallback for the rest.
    if args.classifier_train:
        import pandas as pd
        classifier_table = add_x_fe_columns(
            pd.read_parquet(args.classifier_train))
    else:
        classifier_table = table
    classifier = fit_state_classifier(classifier_table)
    is_rgb, source, rgb_proba = select_rgb(table, classifier, args.min_proba)
    keep &= is_rgb
    if keep.sum() == 0:
        raise ValueError("no RGB stars survive the selection; check the "
                         "EvoState column, the classifier training table, or "
                         "--min-proba")

    # Normalize only the selected stars, identically to training.
    normalized_flux, normalized_ivar = normalize_spectra(
        dispersion, flux[keep], ivar[keep], args.continuum_list)

    catalogue = apply_model(
        model, table.loc[keep], normalized_flux, normalized_ivar,
        source[keep], rgb_proba[keep], test_batch_size=args.test_batch_size)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    if args.output.endswith(".parquet"):
        catalogue.to_parquet(args.output, index=False)
    else:
        catalogue.to_csv(args.output, index=False)
    logger.info("wrote %d-star age catalogue to %s", len(catalogue),
                args.output)

    finite = np.isfinite(catalogue["age_gyr"])
    physical = finite & ~catalogue["flag_unphysical_age"]
    print("\n=== ages for {0} RGB stars ({1} seismic, {2} classified) ==="
          .format(len(catalogue),
                  int((catalogue["evo_state_source"] == "seismic").sum()),
                  int((catalogue["evo_state_source"] == "classified").sum())))
    print("  median age: {0:.2f} Gyr | 16th-84th: {1:.2f}-{2:.2f} Gyr | "
          "{3} unphysical (outside 0-20 Gyr)".format(
              float(np.median(catalogue.loc[physical, "age_gyr"])),
              float(np.percentile(catalogue.loc[physical, "age_gyr"], 16)),
              float(np.percentile(catalogue.loc[physical, "age_gyr"], 84)),
              int((~physical & finite).sum())))


if __name__ == "__main__":
    main()
