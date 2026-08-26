#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classify a bulge sample into RGB / RC from its SPECTRA, not from labels alone.

``bingo-modern/classify_bulge.py`` classifies on ASPCAP labels because it was
written to feed a label-only BNN, before the bulge spectra were assembled.
Nothing about the bulge forces that: ``crossmatched_5.parquet`` carries
per-row ``wavelength``/``flux``/``ivar`` for the whole cleaned catalogue, on
the same APOGEE grid as the calibration sample and normalized with the same
continuum list. The Cannon and Lux already transfer spectral models across
this gap to produce the ages, so a spectral classifier transfers on exactly
the same terms -- and the CN/CH molecular features it can then see are the
ones that actually separate RC from RGB (Hawkins+2018, Ting+2018).

The classifier is ``scripts.apply_rgb_ages``'s: gradient boosting on the label
columns plus a PCA of the normalized flux, fitted here on the seismically
labelled rows of the calibration sample. Cross-validated accuracy, ROC AUC and
purity/completeness at the acceptance threshold are logged before it is
trusted -- read them (and the clump-box number) before believing the output.

ONE feature differs from the calibration-side selection, and it is the one
difference the bulge genuinely forces: log g is the SPECTROSCOPIC value, not
the seismic one, because no bulge star has a numax. Train and apply with the
same column or the classifier is being handed a feature it will not have.

Outputs one parquet with the id column, ``rgb_proba`` (= p(RGB)),
``evo_class`` (RGB / RC / ambiguous), ``is_rgb``, ``evo_state_source`` and
``in_domain`` -- the schema of ``rgb_selection_all_missions.parquet``, so
anything already reading that file reads this too.

``--compare <label-only parquet>`` additionally reports the flip test: how
many stars change class relative to an existing label-only classification,
and in which direction.

Usage
-----
::

    python -m scripts.classify_bulge_spectral \\
        --classifier-train $DATA/merged_with_ages_raw.parquet \\
        --spectra $DATA/crossmatched_5.parquet \\
        --continuum-list $DATA/continuum.list \\
        --output $DATA/rgb_selection_bulge_spectral.parquet \\
        --compare $DATA/../bingo-modern/RGB_RC_classifier/bulge_rgb_bnn.parquet
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import logging
import os

import numpy as np
import pandas as pd

try:
    from scripts.train_cannon import (load_spectra, normalize_spectra,
                                      add_x_fe_columns, _to_array)
    from scripts.apply_rgb_ages import (RGB, HEB, EVO_FEATURES, seismic_state,
                                        classifier_matrix, fit_state_classifier,
                                        DEFAULT_SPECTRAL_COMPONENTS)
    from scripts.apply_cannon import find_id_column
except ImportError:
    from train_cannon import (load_spectra, normalize_spectra,
                              add_x_fe_columns, _to_array)
    from apply_rgb_ages import (RGB, HEB, EVO_FEATURES, seismic_state,
                                classifier_matrix, fit_state_classifier,
                                DEFAULT_SPECTRAL_COMPONENTS)
    from apply_cannon import find_id_column

logger = logging.getLogger("thecannon.classify_bulge_spectral")

SENTINEL_COLS = ["raw_teff", "raw_logg", "raw_fe_h", "raw_mg_h", "raw_c_h",
                 "raw_n_h"]
ARRAY_COLS = ("wavelength", "flux", "ivar")


def prepare(table):
    """ Derive the [X/Fe] columns the classifier features need, in place. """
    add_x_fe_columns(table)
    if "c_n" not in table.columns and {"raw_c_h", "raw_n_h"} <= set(table.columns):
        table["c_n"] = table["raw_c_h"] - table["raw_n_h"]
    return table


def fit(args):
    """ Fit the state classifier on the calibration sample's labelled rows.

    Returns ``(classifier, dispersion, domain)``: ``domain`` is the 1st-99th
    percentile range of each label feature over the training stars, used for
    the ``in_domain`` flag exactly as the label-only classifier does.
    """
    table, dispersion, flux, ivar = load_spectra(args.classifier_train)
    prepare(table)

    sentinel = (table[[c for c in SENTINEL_COLS if c in table.columns]]
                < -100).any(axis=1).to_numpy()
    if sentinel.any():
        logger.info("dropping %d ASPCAP sentinel rows from the classifier "
                    "training set", int(sentinel.sum()))
        table = table[~sentinel].reset_index(drop=True)
        flux, ivar = flux[~sentinel], ivar[~sentinel]

    # The bulge has no numax, so the classifier must be trained on the log g
    # the bulge actually carries. Passing --logg-column logg_seismic here
    # would train on a feature the target sample cannot supply.
    if args.logg_column != "raw_logg":
        if args.logg_column not in table.columns:
            raise SystemExit("no '%s' column in the training table"
                             % args.logg_column)
        logger.warning("training on %s: the target sample must carry that "
                       "same column", args.logg_column)
        table["raw_logg"] = table[args.logg_column]

    nf, ni = normalize_spectra(dispersion, flux, ivar, args.continuum_list)
    clf = fit_state_classifier(table, np.asarray(nf), np.asarray(ni),
                               features=args.features,
                               n_components=args.n_components,
                               min_proba=args.min_proba, target=RGB)
    if clf is None:
        raise SystemExit("too few seismically labelled stars to fit")

    labelled = seismic_state(table).notna().to_numpy()
    feats = table.loc[labelled, EVO_FEATURES].apply(pd.to_numeric,
                                                    errors="coerce")
    domain = {c: (float(np.nanpercentile(feats[c], 1)),
                  float(np.nanpercentile(feats[c], 99))) for c in EVO_FEATURES}
    return clf, dispersion, domain


def classify(args, clf, dispersion, domain):
    """ Stream the target sample and return a per-row classification table. """
    import pyarrow.dataset as ds

    dset = ds.dataset(args.spectra)
    names = set(dset.schema.names)
    missing = [c for c in ARRAY_COLS if c not in names]
    if missing:
        raise SystemExit("%s has no %s column -- it carries no spectra; the "
                         "label-only classifier is the only option on that "
                         "table" % (args.spectra, "/".join(missing)))
    scalar = [c for c in dset.schema.names if c not in ARRAY_COLS]

    out, n_seen = [], 0
    for batch in dset.to_batches(batch_size=args.batch_rows):
        if batch.num_rows == 0:
            continue
        chunk = batch.to_pandas()
        prepare(chunk)
        disp = _to_array(chunk["wavelength"].iloc[0])
        if disp.size != dispersion.size:
            raise SystemExit(
                "target spectra have %d pixels but the classifier was trained "
                "on %d; they are on different wavelength grids"
                % (disp.size, dispersion.size))
        flux = np.vstack([_to_array(x) for x in chunk["flux"]])
        ivar = np.vstack([_to_array(x) for x in chunk["ivar"]])
        nf, ni = normalize_spectra(disp, flux, ivar, args.continuum_list)

        X = classifier_matrix(chunk, np.asarray(nf), np.asarray(ni),
                              getattr(clf, "features_used_", args.features))
        p_rgb = clf.predict_proba(X)[:, list(clf.classes_).index(RGB)]

        keep = chunk[[c for c in scalar if c in chunk.columns]].copy()
        keep["rgb_proba"] = p_rgb
        out.append(keep)
        n_seen += len(chunk)
        logger.info("classified %d rows", n_seen)

    res = pd.concat(out, ignore_index=True)
    p = res["rgb_proba"].to_numpy()
    res["evo_class"] = np.select([p > args.min_proba, p < 1.0 - args.min_proba],
                                 ["RGB", "RC"], default="ambiguous")
    res["is_rgb"] = res["evo_class"].eq("RGB")
    res["evo_state_source"] = np.where(res["is_rgb"], "classified", "")
    in_domain = np.ones(len(res), bool)
    for c, (lo, hi) in domain.items():
        if c in res.columns:
            v = pd.to_numeric(res[c], errors="coerce").to_numpy(float)
            in_domain &= np.isfinite(v) & (v >= lo) & (v <= hi)
    res["in_domain"] = in_domain
    return res


def flip_test(res, compare_path, id_col):
    """ How the spectral classification differs from a label-only one. """
    other = pd.read_parquet(compare_path)
    other_id = find_id_column(other) if id_col not in other.columns else id_col
    if other_id is None:
        logger.warning("no id column in %s; skipping the flip test",
                       compare_path)
        return None
    # The label-only products are one file per class, so membership IS the
    # label: a star present in the RGB file was called RGB.
    if "evo_class" in other.columns:
        other_rgb = other["evo_class"].eq("RGB")
    elif "p_rc" in other.columns:
        other_rgb = other["p_rc"] < 0.3
    else:
        other_rgb = pd.Series(True, index=other.index)
    lut = pd.Series(other_rgb.to_numpy(),
                    index=other[other_id].astype(str)).groupby(level=0).first()

    ids = res[id_col].astype(str)
    theirs = ids.map(lut)
    both = theirs.notna().to_numpy()
    mine = res["is_rgb"].to_numpy()
    t = theirs.fillna(False).to_numpy().astype(bool)
    summary = dict(
        n_matched=int(both.sum()),
        agree_rgb=int((mine & t & both).sum()),
        agree_not_rgb=int((~mine & ~t & both).sum()),
        spectral_only=int((mine & ~t & both).sum()),
        labels_only=int((~mine & t & both).sum()),
        not_in_comparison=int((~both).sum()))
    summary["agreement"] = (float((mine[both] == t[both]).mean())
                            if both.any() else float("nan"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Classify a bulge sample into RGB/RC from its spectra.")
    parser.add_argument("--spectra", required=True,
                        help="target sample WITH spectra (crossmatched_5.parquet)")
    parser.add_argument("--classifier-train", required=True,
                        help="calibration sample with seismic EvoState + spectra")
    parser.add_argument("--continuum-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", default="both",
                        choices=("labels", "spectra", "both"))
    parser.add_argument("--logg-column", default="raw_logg",
                        help="log g the classifier trains on; keep the "
                             "SPECTROSCOPIC raw_logg for a bulge sample, "
                             "which has no numax (default: raw_logg)")
    parser.add_argument("--n-components", type=int,
                        default=DEFAULT_SPECTRAL_COMPONENTS)
    parser.add_argument("--min-proba", type=float, default=0.9,
                        help="accept as RGB above this p(RGB), as RC below "
                             "1 - it, ambiguous in between (default 0.9)")
    parser.add_argument("--batch-rows", type=int, default=4000)
    parser.add_argument("--compare", default=None,
                        help="label-only classification to flip-test against")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("thecannon").propagate = False

    clf, dispersion, domain = fit(args)
    res = classify(args, clf, dispersion, domain)

    id_col = find_id_column(res)
    counts = res["evo_class"].value_counts()
    logger.info("classified %d rows: %s", len(res), counts.to_dict())
    print("\n=== spectral RGB/RC classification (p(RGB) > %.2f) ==="
          % args.min_proba)
    print(counts.to_string())
    print("in_domain: %.1f%%" % (100 * res["in_domain"].mean()))

    if args.compare:
        summary = flip_test(res, args.compare, id_col)
        if summary:
            print("\n=== flip test vs %s ===" % os.path.basename(args.compare))
            for k, v in summary.items():
                print("  %-20s %s" % (k, v))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    res.to_parquet(args.output, index=False)
    print("\nwrote %s (%d rows, %d RGB)"
          % (args.output, len(res), int(res["is_rgb"].sum())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
