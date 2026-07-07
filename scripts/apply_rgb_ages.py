#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Apply a saved CannonModel to the RGB stars of a sample and estimate their ages.

Unlike :mod:`scripts.apply_cannon` (which trains a model first), this script
loads an already-trained pickled model, selects the first-ascent RGB stars,
runs the test step on them only, and writes a per-star age catalogue.

RGB selection uses the seismic evolutionary state where available
(``EvoState`` == 1) and falls back to a gradient-boosting classifier for stars
without one, accepting only predictions with probability above ``--min-proba``.
The classifier features are set by ``--classifier-features``: the label space
(Teff / log g / [Fe/H] / [C/Fe] / [N/Fe] / [Mg/Fe]), a PCA compression of the
continuum-normalized spectra (the CN/CH molecular features that make RC and
RGB stars spectroscopically separable; Hawkins+2018, Ting+2018), or both (the
default). The classifier is trained on the seismically labeled rows of
``--classifier-train`` (default: the input sample itself, so a mixed
seismic+unknown table needs no extra file; a pure target sample, e.g. bulge
stars, should point this at the seismic training parquet) and its stratified
cross-validated accuracy, ROC AUC, and RGB purity/completeness at the
acceptance threshold are logged before it is trusted.

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
        --output results/rgb_ages.parquet
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
CLASSIFIER_FEATURE_MODES = ("labels", "spectra", "both")
DEFAULT_SPECTRAL_COMPONENTS = 50


def seismic_state(table):
    """ Numeric ``EvoState`` per row (NaN where absent or not RGB/HeB). """
    import pandas as pd
    if "EvoState" not in table.columns:
        return pd.Series(np.nan, index=table.index)
    state = pd.to_numeric(table["EvoState"], errors="coerce")
    return state.where(state.isin([RGB, HEB]))


def clean_flux(normalized_flux, normalized_ivar=None):
    """
    Continuum-fill (flux = 1) the unusable pixels -- non-finite flux or, when
    ``normalized_ivar`` is given, zero inverse variance -- and clip artifacts,
    so PCA sees finite values and bad pixels do not dominate the variance.
    """
    flux = np.array(normalized_flux, dtype=float)
    bad = ~np.isfinite(flux)
    if normalized_ivar is not None:
        bad |= ~(np.asarray(normalized_ivar) > 0)
    flux[bad] = 1.0
    return np.clip(flux, 0.0, 3.0)


def classifier_matrix(table, normalized_flux=None, normalized_ivar=None,
                      features="both"):
    """
    Raw feature matrix for the evolutionary-state classifier: the label
    columns (``EVO_FEATURES``), the cleaned normalized flux, or both side by
    side. The spectral block stays uncompressed here -- the PCA lives inside
    the classifier pipeline so cross-validation refits it per fold.
    """
    if features not in CLASSIFIER_FEATURE_MODES:
        raise ValueError("unknown classifier features {0!r}".format(features))
    parts = []
    if features in ("labels", "both"):
        parts.append(table[EVO_FEATURES].to_numpy(dtype=float))
    if features in ("spectra", "both"):
        if normalized_flux is None:
            raise ValueError("features={0!r} needs normalized spectra"
                             .format(features))
        parts.append(clean_flux(normalized_flux, normalized_ivar))
    return np.hstack(parts)


def _make_estimator(features, n_components):
    """ The classifier pipeline: gradient boosting on the label columns
    (passed through) and a PCA compression of the spectral block. """
    from sklearn.ensemble import HistGradientBoostingClassifier

    gbm = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    if features == "labels":
        return gbm

    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    pca = PCA(n_components=n_components, random_state=0)
    if features == "spectra":
        return Pipeline([("pca", pca), ("gbm", gbm)])
    reduce = ColumnTransformer(
        [("labels", "passthrough", slice(0, len(EVO_FEATURES))),
         ("spectra", pca, slice(len(EVO_FEATURES), None))])
    return Pipeline([("features", reduce), ("gbm", gbm)])


def _report_cv(estimator, X, y, labeled_table, min_proba, features):
    """
    Log honest (stratified cross-validated) accuracy estimates before the
    classifier is trusted: overall accuracy and ROC AUC, RGB purity and
    completeness at the ``min_proba`` acceptance threshold, and the accuracy
    inside the ambiguous clump box (the only region where the answer is not
    obvious from log g alone, so the global number is inflated by easy stars).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    classes, counts = np.unique(y, return_counts=True)
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        logger.warning("too few stars in the rarest class (%d) to "
                       "cross-validate the classifier", int(counts.min()))
        return
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    proba = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
    p_rgb = proba[:, list(classes).index(RGB)]
    predicted = classes[proba.argmax(axis=1)]
    correct = predicted == y

    accepted = p_rgb > min_proba
    purity = (float((y[accepted] == RGB).mean()) if accepted.any()
              else float("nan"))
    completeness = float(accepted[y == RGB].mean())
    logger.info("classifier CV (%d-fold, features=%s): accuracy %.3f, "
                "ROC AUC %.3f; at p > %.2f: RGB purity %.3f, "
                "completeness %.3f (%d/%d accepted)",
                n_splits, features, float(correct.mean()),
                float(roc_auc_score(y == RGB, p_rgb)), min_proba, purity,
                completeness, int(accepted.sum()), len(y))

    logg = np.asarray(labeled_table["raw_logg"], dtype=float)
    teff = np.asarray(labeled_table["raw_teff"], dtype=float)
    box = (logg > 2.2) & (logg < 2.7) & (teff > 4500) & (teff < 5100)
    if box.any():
        logger.info("classifier CV clump-box accuracy (2.2 < logg < 2.7, "
                    "4500 < Teff < 5100 K): %.3f (n=%d)",
                    float(correct[box].mean()), int(box.sum()))


def fit_state_classifier(train_table, normalized_flux=None,
                         normalized_ivar=None, features="both",
                         n_components=DEFAULT_SPECTRAL_COMPONENTS,
                         min_proba=0.9, min_labeled=200):
    """
    Fit the RGB/HeB classifier on the seismically labeled rows of
    ``train_table`` and return it (or ``None`` if too few labeled stars).
    Spectral feature modes need the training spectra as ``normalized_flux`` /
    ``normalized_ivar`` (aligned with the table rows); if they are absent the
    classifier falls back to label features with a warning. The fitted
    estimator records the mode actually used as ``features_used_``.
    """
    state = seismic_state(train_table)
    labeled = state.notna().to_numpy()
    if labeled.sum() < min_labeled:
        logger.warning("only %d seismically labeled stars in the classifier "
                       "training table (< %d); skipping classification",
                       int(labeled.sum()), min_labeled)
        return None

    if features != "labels" and normalized_flux is None:
        logger.warning("no spectra available for the classifier training "
                       "table; falling back to label features only")
        features = "labels"

    y = state[labeled].to_numpy()
    flux = ivar = None
    if features != "labels":
        flux = np.asarray(normalized_flux)[labeled]
        if normalized_ivar is not None:
            ivar = np.asarray(normalized_ivar)[labeled]
        n_components = int(min(n_components, flux.shape[1], len(y) - 1))
    X = classifier_matrix(train_table.loc[labeled], flux, ivar, features)

    clf = _make_estimator(features, n_components)
    _report_cv(clf, X, y, train_table.loc[labeled], min_proba, features)
    clf.fit(X, y)
    clf.features_used_ = features
    logger.info("state classifier trained on %d labeled stars "
                "(%d RGB, %d HeB; features=%s)", len(y),
                int((y == RGB).sum()), int((y == HEB).sum()), features)
    return clf


def select_rgb(table, classifier, min_proba, normalized_flux=None,
               normalized_ivar=None):
    """
    Return ``(is_rgb, source, rgb_proba)`` arrays over the rows of ``table``:
    a boolean RGB mask, how each star was identified (``"seismic"``,
    ``"classified"`` or ``""``), and the classifier RGB probability (NaN for
    seismically identified stars). ``normalized_flux`` / ``normalized_ivar``
    (aligned with the table rows) are required when the classifier was
    trained with spectral features.
    """
    state = seismic_state(table)
    # copy=True: under pandas copy-on-write, to_numpy() may return a read-only
    # view, and this mask is assigned into below.
    is_rgb = (state == RGB).to_numpy(copy=True)
    # Widen the dtype past "seismic" or assigning "classified" truncates it.
    source = np.where(state.notna(), "seismic", "").astype("<U10")
    rgb_proba = np.full(len(table), np.nan)

    unknown = state.isna().to_numpy()
    if unknown.any() and classifier is not None:
        features = getattr(classifier, "features_used_", "labels")
        flux = ivar = None
        if features != "labels":
            if normalized_flux is None:
                raise ValueError("the classifier uses spectral features but "
                                 "no normalized spectra were passed")
            flux = np.asarray(normalized_flux)[unknown]
            if normalized_ivar is not None:
                ivar = np.asarray(normalized_ivar)[unknown]
        X = classifier_matrix(table.loc[unknown], flux, ivar, features)
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


def load_classifier_table(path, want_spectra, dispersion, continuum_list_path):
    """
    Load the ``--classifier-train`` parquet, returning ``(table,
    normalized_flux, normalized_ivar)``. When ``want_spectra`` the spectra are
    continuum-normalized identically to the sample; a table without usable
    spectra columns degrades to ``(table, None, None)`` with a warning (the
    classifier then uses label features only).
    """
    import pandas as pd

    if want_spectra:
        try:
            table, train_dispersion, flux, ivar = load_spectra(path)
        except (KeyError, ValueError, AssertionError) as e:
            logger.warning("classifier training table %s has no usable "
                           "spectra (%s); using label features only", path, e)
            return add_x_fe_columns(pd.read_parquet(path)), None, None
        if train_dispersion.size != dispersion.size:
            raise ValueError(
                "classifier training spectra have {0} pixels but the sample "
                "has {1}; they are on different wavelength grids".format(
                    train_dispersion.size, dispersion.size))
        normalized_flux, normalized_ivar = normalize_spectra(
            train_dispersion, flux, ivar, continuum_list_path)
        return table, normalized_flux, normalized_ivar
    return add_x_fe_columns(pd.read_parquet(path)), None, None


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
    parser.add_argument("--classifier-features", default="both",
                        choices=CLASSIFIER_FEATURE_MODES,
                        help="feature space of the RGB classifier: stellar "
                             "labels, PCA-compressed normalized spectra, or "
                             "both (default: both)")
    parser.add_argument("--classifier-components", type=int,
                        default=DEFAULT_SPECTRAL_COMPONENTS,
                        help="number of PCA components kept from the spectra "
                             "(default: {0})".format(
                                 DEFAULT_SPECTRAL_COMPONENTS))
    parser.add_argument("--min-proba", type=float, default=0.9,
                        help="classifier probability above which an unlabeled "
                             "star is accepted as RGB (default: 0.9)")
    parser.add_argument("--no-quality-cut", action="store_true",
                        help="skip the spectrum_flags / warn_* quality cuts")
    parser.add_argument("--output", default="rgb_ages.parquet",
                        help="output catalogue path (parquet; a .csv suffix "
                             "writes CSV instead)")
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

    if not args.no_quality_cut:
        keep = np.asarray(quality_mask(table), dtype=bool)
        logger.info("quality cuts keep %d/%d stars", int(keep.sum()),
                    len(table))
        table, flux, ivar = table.loc[keep], flux[keep], ivar[keep]
        if len(table) == 0:
            raise ValueError("no stars survive the quality cuts")

    # With spectral classifier features the whole (quality-cut) sample is
    # normalized once, up front: the unknown-state stars need it for
    # classification, and the test step reuses the RGB rows.
    use_spectra = args.classifier_features != "labels"
    normalized_flux = normalized_ivar = None
    if use_spectra:
        normalized_flux, normalized_ivar = normalize_spectra(
            dispersion, flux, ivar, args.continuum_list)

    # RGB selection: seismic state first, classifier fallback for the rest.
    if args.classifier_train:
        classifier_table, classifier_flux, classifier_ivar = \
            load_classifier_table(args.classifier_train, use_spectra,
                                  dispersion, args.continuum_list)
    else:
        classifier_table = table
        classifier_flux, classifier_ivar = normalized_flux, normalized_ivar
    classifier = fit_state_classifier(
        classifier_table, classifier_flux, classifier_ivar,
        features=args.classifier_features,
        n_components=args.classifier_components, min_proba=args.min_proba)
    is_rgb, source, rgb_proba = select_rgb(
        table, classifier, args.min_proba, normalized_flux, normalized_ivar)
    if is_rgb.sum() == 0:
        raise ValueError("no RGB stars survive the selection; check the "
                         "EvoState column, the classifier training table, or "
                         "--min-proba")

    # Normalize only the selected stars, identically to training (already
    # done above when the classifier consumed spectra).
    if normalized_flux is None:
        normalized_flux, normalized_ivar = normalize_spectra(
            dispersion, flux[is_rgb], ivar[is_rgb], args.continuum_list)
    else:
        normalized_flux = normalized_flux[is_rgb]
        normalized_ivar = normalized_ivar[is_rgb]

    catalogue = apply_model(
        model, table.loc[is_rgb], normalized_flux, normalized_ivar,
        source[is_rgb], rgb_proba[is_rgb],
        test_batch_size=args.test_batch_size)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    if args.output.endswith(".csv"):
        catalogue.to_csv(args.output, index=False)
    else:
        catalogue.to_parquet(args.output, index=False)
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
