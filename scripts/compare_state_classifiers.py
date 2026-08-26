#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ablate the RGB/HeB classifier's INFORMATION CONTENT, to justify (or refute)
running two different classifiers on the calibration and application samples.

The training sample is selected with seismic ``EvoState`` where it exists and,
for the rest, a classifier on (Teff, SEISMIC log g, [Fe/H], [C/Fe], [N/Fe],
[Mg/Fe]) + a PCA of the normalized spectra (``train_rgb_wilett_all_missions``
cell 17). The bulge cannot have either of those: no numax, and spectral PCA
components fitted on Kepler/K2/TESS field spectra are not obviously valid at
bulge metallicity, extinction and S/N -- so ``bingo-modern/classify_bulge.py``
uses ASPCAP labels with the SPECTROSCOPIC log g instead.

That difference is only defensible if it is quantified. This script fits the
SAME estimator family at three information levels on the SAME stars and scores
every one of them out-of-fold against seismic truth:

    bulge      labels, spectroscopic log g   <- the bulge classifier's inputs
    bulge+seis labels, seismic log g         <- isolates the log g term
    full       labels (seismic log g) + PCA  <- the production training-side one

It follows the benchmark's conventions so its numbers sit next to the
Cannon/Lux/BNN ones: the cuts and derivations come from ``stardata`` itself,
the folds are STAR-level (a star observed by two missions never straddles a
fold, as ``stardata`` insists) and seeded with the benchmark's split seed 42,
and rows repeating an (APOGEE_ID, source) pair -- the same spectrum handed
out twice -- are dropped rather than allowed to vote twice.

``--dataset-dir <benchmark>/dataset`` inherits that benchmark's merged
catalogue, split seed and fold count from its ``manifest.json``, so the
ablation is pinned to the run it will be read beside. It cannot read
``stars.parquet`` itself: that table is ALREADY RGB-selected, so the rejected
stars are gone and the HeB negative class -- the thing this classifier has to
separate -- does not exist in it. The ablation therefore starts from the same
merged catalogue and re-derives the pre-selection cuts.

It answers the two questions a referee will ask:

1. **What does the bulge feature space actually cost?**  Purity and
   completeness against seismic truth at a grid of acceptance thresholds,
   including the matched-purity threshold -- the operating point at which the
   bulge-style classifier is as pure as the production one, so the cost shows
   up as completeness rather than as hidden contamination.

2. **Is the lost completeness flat, or does it correlate with age?**  An
   incompleteness that tracks [C/N] tracks mass, hence age, and would bias the
   bulge age distribution directly. Reported as completeness per quintile of
   [C/N] / log(age) / [Fe/H] / Teff, the induced mean-age shift in dex, and a
   KS test of accepted vs rejected true RGB stars.

It also applies each fitted classifier to the WHOLE sample and counts how many
stars change class relative to the production selection -- the selection
function difference between calibration and application.

Outputs (in ``--outdir``): ``cv_metrics.csv``, ``completeness_profile.csv``,
``selection_flips.csv``, ``summary.json``, and the per-star out-of-fold
probabilities in ``oof_probabilities.parquet``.

Usage
-----
::

    python -m scripts.compare_state_classifiers \\
        --dataset-dir ~/scr_mk27/benchmark_v8/dataset \\
        --continuum-list ~/scr_mk27/bulge-ages-and-orbits/data/continuum.list \\
        --outdir ~/scr_mk27/benchmark_v8/state_ablation -v

The continuum normalization is the expensive step; ``--cache <file.npz>``
stores it and later runs reuse it (the cache records which rows it was built
from and is rejected if the selection changed).
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

for _c in (REPO.parent / "stardiag", Path.home() / "code" / "stardiag",
           Path.home() / "scr_mk27" / "stardiag"):
    if (_c / "stardata.py").exists():
        sys.path.insert(0, str(_c))
        break
else:
    sys.exit("stardiag checkout (stardata.py) not found next to this repo")
import stardata  # noqa: E402

try:
    from scripts.train_cannon import load_spectra, normalize_spectra
    from scripts.apply_rgb_ages import (RGB, HEB, EVO_FEATURES, seismic_state,
                                        clean_flux,
                                        DEFAULT_SPECTRAL_COMPONENTS)
except ImportError:
    from train_cannon import load_spectra, normalize_spectra
    from apply_rgb_ages import (RGB, HEB, EVO_FEATURES, seismic_state,
                                clean_flux, DEFAULT_SPECTRAL_COMPONENTS)

logger = logging.getLogger("thecannon.state_ablation")

# The bulge classifier's feature list (bingo-modern/classify_bulge.py FEATS).
# It differs from EVO_FEATURES by carrying c_n explicitly; kept verbatim so
# this measures the bulge classifier as it is, not a tidied-up version.
BULGE_FEATURES = ["raw_teff", "raw_logg", "raw_fe_h", "mg_fe", "c_fe", "n_fe",
                  "c_n"]
# bingo-modern's hyperparameters, for the control config that shows the
# feature space -- not the hyperparameters -- drives the difference.
BULGE_HGB = dict(max_leaf_nodes=15, l2_regularization=1.0, learning_rate=0.06,
                 random_state=42)

# Production acceptance threshold on p(RGB) (train_rgb_wilett_all_missions
# MIN_PROBA) and the bulge one (classify_bulge THR_LO on p(RC) = 0.3).
PROD_MIN_PROBA, BULGE_MIN_PROBA = 0.9, 0.7
THRESHOLD_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

# The cuts, the [X/Fe] and seismic-log g derivations and the deterministic
# best-SNR row pick all come from stardata, so this ablation and the
# benchmark can never drift apart on a formula or a sentinel threshold.
SENTINEL_COLS = stardata.SENTINEL_COLS
# The benchmark's split seed (stardata.build_dataset / run_benchmark
# --split-seed) and fold count, so the folds here are drawn the same way.
DEFAULT_SPLIT_SEED, DEFAULT_FOLDS = 42, 5

PROFILE_COLS = ["c_n", "log_age_L", "raw_fe_h", "raw_teff"]

# The bulge is more metal-rich and lower-S/N than the calibration field, so a
# classifier's global numbers are optimistic for it. These subsets score the
# SAME out-of-fold probabilities on the calibration stars that most resemble
# the bulge: if a feature space holds up in the hardest calibration regime it
# will likely hold in the bulge, and if it collapses there the transfer was
# never safe. Quartiles, so the counts stay honest and visible.
STRESS_SUBSETS = (
    ("all", None, None),
    ("metal_rich", "raw_fe_h", "high"),
    ("low_snr", "snr", "low"),
    ("bulge_like", None, "both"),        # metal-rich AND low-S/N
)


def stress_masks(table, min_n=100):
    """ ``{name: boolean mask}`` over the labelled rows (subsets with fewer
    than ``min_n`` stars, or whose column is missing, are dropped). """
    out, parts = {}, {}
    for name, col, side in STRESS_SUBSETS:
        if name == "all":
            out["all"] = np.ones(len(table), bool)
            continue
        if side == "both":
            continue
        if col not in table.columns:
            logger.warning("no '%s' column; skipping the %s subset", col, name)
            continue
        v = pd.to_numeric(table[col], errors="coerce").to_numpy(float)
        if not np.isfinite(v).any():
            logger.warning("'%s' is all non-finite; skipping the %s subset",
                           col, name)
            continue
        q = np.nanquantile(v, 0.75 if side == "high" else 0.25)
        m = (v >= q) if side == "high" else (v <= q)
        m &= np.isfinite(v)
        out[name] = parts[name] = m
    if len(parts) == 2:
        out["bulge_like"] = np.logical_and(*parts.values())
    kept = {k: v for k, v in out.items() if k == "all" or v.sum() >= min_n}
    for k in set(out) - set(kept):
        logger.warning("stress subset %s has only %d stars (< %d); dropped",
                       k, int(out[k].sum()), min_n)
    return kept


# --------------------------------------------------------------------- data

def derive_columns(table):
    """ [X/Fe], [C/N], seismic log g and log(age) -- delegated to
    ``stardata.derive_columns``, the same code the benchmark dataset is built
    with, so the ablation selects on identical numbers. """
    return stardata.derive_columns(table)


def apply_training_cuts(table):
    """ The all-missions notebook's pre-selection: finite Willett age, no
    ASPCAP sentinels, reliable RelAge_L, no young alpha-rich impostors. The
    RGB selection itself is what we are ablating, so it is NOT applied. """
    counts = {"input_rows": int(len(table))}
    table = table[np.isfinite(table["age_L"])]
    counts["nonfinite_age_L"] = counts["input_rows"] - len(table)
    table = derive_columns(table.copy())

    sentinel = (table[SENTINEL_COLS] < -100).any(axis=1)
    unreliable = ~table["RelAge_L"].astype(bool)
    impostor = ((table["age_L"] < 4) & (table["mg_fe"] > 0.15)
                & (table["c_n"] > -0.2))
    counts.update(aspcap_sentinel=int(sentinel.sum()),
                  unreliable_RelAge_L=int(unreliable.sum()),
                  young_alpha_rich_impostor=int(impostor.sum()))
    keep = ~(sentinel | unreliable | impostor)
    table = table[keep.to_numpy()].reset_index(drop=True)
    counts["output_rows"] = int(len(table))
    logger.info("cuts: %s", json.dumps(counts))
    return table, counts


def restrict_rows(table, counts, drop_duplicate_spectra, primary_only):
    """
    Apply the benchmark's row conventions and return the surviving positions.

    ``stardata`` flags rows that repeat an (APOGEE_ID, source) pair: they are
    NOT independent observations, because ``load_spectra`` dedupes on that
    pair and hands every one of them the SAME spectrum. Left in, a star with
    three such rows votes three times in cross-validation and its spectrum
    appears in both a training fold and the fold scoring it.

    ``--primary-only`` goes further and keeps each star's deterministic
    best-SNR row (``stardata.primary_index``) -- the benchmark's "one
    deterministic row per star everywhere".
    """
    pos = np.arange(len(table))
    if drop_duplicate_spectra and "source" in table.columns:
        dup = table.duplicated(["APOGEE_ID", "source"], keep="first").to_numpy()
        counts["duplicate_spectrum_rows"] = int(dup.sum())
        table, pos = table[~dup].reset_index(drop=True), pos[~dup]
    if primary_only:
        keep = table.index.isin(stardata.primary_index(table))
        counts["non_primary_rows"] = int((~keep).sum())
        table, pos = table[keep].reset_index(drop=True), pos[keep]
    counts["rows_used"] = int(len(table))
    counts["stars_used"] = int(table["APOGEE_ID"].nunique())
    logger.info("rows: %d used, %d stars (dropped %d duplicate-spectrum, "
                "%d non-primary)", counts["rows_used"], counts["stars_used"],
                counts.get("duplicate_spectrum_rows", 0),
                counts.get("non_primary_rows", 0))
    return table, pos, counts


def load_normalized(args):
    """ ``(table, normalized_flux, normalized_ivar, counts)`` for the post-cut
    sample, reusing ``--cache`` when it was built for the same rows. """
    table, dispersion, flux, ivar = load_spectra(args.spectra)
    # Positions carried explicitly: (APOGEE_ID, source) is NOT unique in the
    # merged catalogue, so the surviving rows cannot be recovered by key.
    table = table.reset_index(drop=True)
    table["_row_pos"] = np.arange(len(table))

    table, counts = apply_training_cuts(table)
    table, _, counts = restrict_rows(table, counts, args.drop_duplicate_spectra,
                                     args.primary_only)
    pos = table.pop("_row_pos").to_numpy()
    flux, ivar = flux[pos], ivar[pos]

    if args.cache and os.path.exists(args.cache):
        cached = np.load(args.cache, allow_pickle=False)
        if (cached["pos"].shape == pos.shape and np.array_equal(cached["pos"],
                                                                pos)):
            logger.info("reusing normalized spectra from %s", args.cache)
            return table, cached["flux"], cached["ivar"], counts
        logger.warning("%s was built for a different row set; renormalizing",
                       args.cache)

    nf, ni = normalize_spectra(dispersion, flux, ivar, args.continuum_list)
    nf, ni = np.asarray(nf), np.asarray(ni)
    if args.cache:
        np.savez_compressed(args.cache, flux=nf, ivar=ni, pos=pos)
        logger.info("cached normalized spectra -> %s", args.cache)
    return table, nf, ni, counts


# -------------------------------------------------------------- classifiers

CONFIGS = (
    # name, features, logg column, feature list, estimator params
    ("bulge", "labels", "raw_logg", BULGE_FEATURES, None),
    ("bulge+seis", "labels", "logg_seismic", BULGE_FEATURES, None),
    ("full", "both", "logg_seismic", EVO_FEATURES, None),
    ("bulge-hgbparams", "labels", "raw_logg", BULGE_FEATURES, BULGE_HGB),
)


def _config_table(table, logg_col, feature_list):
    """ A copy whose ``raw_logg`` is the config's log g, restricted to the
    config's feature columns (classifier_matrix reads EVO_FEATURES, so the
    feature list is injected by renaming rather than by argument). """
    out = table.copy()
    out["raw_logg"] = out[logg_col]
    return out


def _matrix(table, nf, ni, features, feature_list):
    """ The config's raw feature matrix: its label columns, the cleaned
    normalized flux, or both side by side. The PCA lives inside the estimator
    so cross-validation refits it per fold (as apply_rgb_ages does). """
    parts = []
    if features in ("labels", "both"):
        parts.append(table[feature_list].to_numpy(dtype=float))
    if features in ("spectra", "both"):
        parts.append(clean_flux(nf, ni))
    return np.hstack(parts)


def _estimator(features, n_components, params, n_label_features):
    """ ``_make_estimator`` with the label block sized for THIS config (it
    hard-codes ``len(EVO_FEATURES)``) and optional hyperparameter override. """
    from sklearn.ensemble import HistGradientBoostingClassifier
    gbm = HistGradientBoostingClassifier(**(params or dict(max_iter=300,
                                                           random_state=0)))
    if features == "labels":
        return gbm
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    pca = PCA(n_components=n_components, random_state=0)
    reduce = ColumnTransformer(
        [("labels", "passthrough", slice(0, n_label_features)),
         ("spectra", pca, slice(n_label_features, None))])
    return Pipeline([("features", reduce), ("gbm", gbm)])


def _folds(n_splits, seed):
    """
    Stratified K-fold that keeps every row of a STAR in one fold.

    The benchmark's unit of splitting is the star, not the (APOGEE_ID, source)
    row: the same star observed by Kepler and K2 must never sit in train for
    one fold and test for another (stardata's module docstring). A row-level
    KFold here would leak the same star across the split and quote a purity
    the bulge could never reproduce, since no bulge star was ever seen in
    training.
    """
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed), True
    except ImportError:      # sklearn < 1.0
        from sklearn.model_selection import GroupKFold
        logger.warning("sklearn has no StratifiedGroupKFold; falling back to "
                       "GroupKFold (still star-level, not stratified)")
        return GroupKFold(n_splits=n_splits), True


def out_of_fold(config, table, nf, ni, labeled, n_splits, seed, n_components):
    """ Out-of-fold p(RGB) for the seismically labeled rows, plus the fitted
    full-sample classifier and its p(RGB) over EVERY row. """
    from sklearn.model_selection import cross_val_predict

    name, features, logg_col, feature_list, params = config
    tab = _config_table(table, logg_col, feature_list)
    y = seismic_state(tab)[labeled].to_numpy()

    n_comp = int(min(n_components, nf.shape[1], int(labeled.sum()) - 1))
    X_lab = _matrix(tab.loc[labeled], nf[labeled], ni[labeled], features,
                    feature_list)
    clf = _estimator(features, n_comp, params, len(feature_list))
    cv, grouped = _folds(n_splits, seed)
    groups = tab.loc[labeled, "APOGEE_ID"].to_numpy() if grouped else None
    proba = cross_val_predict(clf, X_lab, y, cv=cv, groups=groups,
                              method="predict_proba")
    classes = np.unique(y)
    p_oof = proba[:, list(classes).index(RGB)]

    clf.fit(X_lab, y)
    X_all = _matrix(tab, nf, ni, features, feature_list)
    p_all = clf.predict_proba(X_all)[:, list(clf.classes_).index(RGB)]
    logger.info("[%s] fitted on %d labeled stars (%d RGB, %d HeB)", name,
                len(y), int((y == RGB).sum()), int((y == HEB).sum()))
    return p_oof, p_all, y


# ------------------------------------------------------------------ metrics

def threshold_metrics(p_rgb, y, table_labeled, thresholds, subset=None):
    """ Purity/completeness of the accepted RGB set against seismic truth,
    optionally restricted to a stress subset (the probabilities are still the
    out-of-fold ones from the full fit -- only the scoring is restricted). """
    from sklearn.metrics import roc_auc_score
    if subset is not None:
        p_rgb, y = p_rgb[subset], y[subset]
        table_labeled = table_labeled.loc[subset].reset_index(drop=True)
    is_rgb = (y == RGB)
    logg = table_labeled["raw_logg"].to_numpy(float)
    teff = table_labeled["raw_teff"].to_numpy(float)
    box = (logg > 2.2) & (logg < 2.7) & (teff > 4500) & (teff < 5100)
    rows = []
    for t in thresholds:
        acc = p_rgb > t
        rows.append(dict(
            threshold=float(t), n_scored=int(len(y)),
            n_accepted=int(acc.sum()),
            purity=float(is_rgb[acc].mean()) if acc.any() else np.nan,
            completeness=float(acc[is_rgb].mean()),
            accuracy=float(((p_rgb > 0.5) == is_rgb).mean()),
            roc_auc=(float(roc_auc_score(is_rgb, p_rgb))
                     if 0 < is_rgb.sum() < len(is_rgb) else np.nan),
            clumpbox_accuracy=(float(((p_rgb > 0.5) == is_rgb)[box].mean())
                               if box.any() else np.nan),
            n_clumpbox=int(box.sum())))
    return rows


def matched_purity_threshold(p_rgb, y, target_purity, min_accepted=50):
    """
    Lowest threshold reaching ``target_purity`` -- the operating point that
    makes two classifiers comparable in contamination, so the cost of a poorer
    feature space appears as completeness instead of hidden interlopers.

    A feature space that never reaches the target returns its BEST achievable
    purity instead, flagged ``matched=False``: that is itself the answer (the
    bulge inputs cannot buy that purity at any threshold), and the downstream
    completeness profile still needs a threshold to work at.
    """
    is_rgb = (y == RGB)
    best = None
    for t in np.unique(np.round(p_rgb, 3)):
        acc = p_rgb > t
        if acc.sum() < min_accepted:
            break
        purity = float(is_rgb[acc].mean())
        if best is None or purity > best[1]:
            best = (float(t), purity, float(acc[is_rgb].mean()))
        if purity >= target_purity:
            return dict(threshold=float(t), purity=purity,
                        completeness=float(acc[is_rgb].mean()), matched=True)
    if best is None:
        return dict(threshold=float("nan"), purity=float("nan"),
                    completeness=float("nan"), matched=False)
    return dict(threshold=best[0], purity=best[1], completeness=best[2],
                matched=False)


def completeness_profile(p_rgb, y, table_labeled, threshold, cols, n_bins=5):
    """ Completeness of the accepted set per quintile of each column, over
    TRUE RGB stars only: is the incompleteness flat, or age-correlated? """
    is_rgb = (y == RGB)
    accepted = p_rgb > threshold
    rows = []
    for col in cols:
        if col not in table_labeled.columns:
            continue
        v = pd.to_numeric(table_labeled[col], errors="coerce").to_numpy(float)
        ok = is_rgb & np.isfinite(v)
        if ok.sum() < n_bins * 10:
            continue
        edges = np.quantile(v[ok], np.linspace(0, 1, n_bins + 1))
        edges[-1] = np.nextafter(edges[-1], np.inf)
        idx = np.digitize(v[ok], edges[1:-1])
        a = accepted[ok]
        for b in range(n_bins):
            m = idx == b
            if not m.any():
                continue
            rows.append(dict(column=col, bin=b, lo=float(edges[b]),
                             hi=float(edges[b + 1]), n=int(m.sum()),
                             completeness=float(a[m].mean())))
        rows.append(dict(column=col, bin=-1, lo=np.nan, hi=np.nan,
                         n=int(ok.sum()), completeness=float(a.mean())))
    return rows


def selection_bias(p_rgb, y, table_labeled, threshold):
    """ Mean shift the selection induces in each profile column, and a KS test
    of accepted vs rejected TRUE RGB stars. NaN p-values if scipy is absent. """
    is_rgb = (y == RGB)
    accepted = p_rgb > threshold
    out = {}
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        ks_2samp = None
    for col in PROFILE_COLS:
        if col not in table_labeled.columns:
            continue
        v = pd.to_numeric(table_labeled[col], errors="coerce").to_numpy(float)
        ok = is_rgb & np.isfinite(v)
        a, r = v[ok & accepted], v[ok & ~accepted]
        if not len(a) or not len(r):
            continue
        out[col] = dict(
            mean_all=float(v[ok].mean()), mean_accepted=float(a.mean()),
            shift=float(a.mean() - v[ok].mean()),
            ks_p=(float(ks_2samp(a, r).pvalue) if ks_2samp is not None
                  else float("nan")),
            n_accepted=int(len(a)), n_rejected=int(len(r)))
    return out


def selection_flips(selections, reference):
    """ Cross-tab of every config's whole-sample selection against the
    production one: the calibration/application selection-function delta. """
    ref = selections[reference]
    rows = []
    for name, sel in selections.items():
        rows.append(dict(
            config=name, n_selected=int(sel.sum()),
            both=int((sel & ref).sum()),
            only_this=int((sel & ~ref).sum()),
            only_reference=int((~sel & ref).sum()),
            agreement=float((sel == ref).mean())))
    return rows


# --------------------------------------------------------------------- main

def inherit_from_dataset(args, parser):
    """
    Fill unset options from a benchmark dataset's ``manifest.json`` and return
    what was inherited (empty when ``--dataset-dir`` is absent).

    Only the provenance and the fold conventions are taken: the dataset's own
    ``stars.parquet`` is already RGB-selected, so it holds neither the stars
    this classifier rejects nor the HeB class it has to separate.
    """
    if not args.dataset_dir:
        if not args.spectra:
            parser.error("--spectra is required unless --dataset-dir is given")
        return {}
    path = os.path.join(args.dataset_dir, "manifest.json")
    with open(path) as fp:
        manifest = json.load(fp)
    inherited = {}
    if args.spectra is None:
        args.spectra = manifest["merged_path"]
        inherited["merged_path"] = args.spectra
    for opt, key in (("seed", "seed"), ("folds", "n_folds")):
        if getattr(args, opt) == parser.get_default(opt) and key in manifest:
            setattr(args, opt, manifest[key])
            inherited[key] = manifest[key]
    inherited["dataset_dir"] = os.path.abspath(args.dataset_dir)
    return inherited


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ablate the RGB/HeB classifier's information content.")
    parser.add_argument("--dataset-dir", default=None,
                        help="a benchmark dataset dir (built by "
                             "stardata.build_dataset); its manifest.json "
                             "supplies the merged catalogue, split seed and "
                             "fold count unless overridden")
    parser.add_argument("--spectra", default=None,
                        help="merged_with_ages_raw.parquet (labels + spectra); "
                             "required unless --dataset-dir is given")
    parser.add_argument("--continuum-list", required=True,
                        help="continuum.list used by the training notebook")
    parser.add_argument("--outdir", default="state_classifier_ablation")
    parser.add_argument("--cache", default=None,
                        help="npz cache of the normalized spectra")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS,
                        help="star-level CV folds (default: the benchmark's "
                             "%d)" % DEFAULT_FOLDS)
    parser.add_argument("--split-seed", dest="seed", type=int,
                        default=DEFAULT_SPLIT_SEED,
                        help="fold seed; the benchmark's split seed (%d) by "
                             "default" % DEFAULT_SPLIT_SEED)
    parser.add_argument("--keep-duplicate-spectra", dest="drop_duplicate_spectra",
                        action="store_false",
                        help="keep rows repeating an (APOGEE_ID, source) pair; "
                             "they carry the SAME spectrum, so by default they "
                             "are dropped as stardata flags them")
    parser.add_argument("--min-stress-n", type=int, default=100,
                        help="drop a bulge-like stress subset with fewer than "
                             "this many labelled stars (default 100)")
    parser.add_argument("--primary-only", action="store_true",
                        help="one deterministic best-SNR row per star "
                             "(stardata.primary_index), as the BNN uses")
    parser.add_argument("--n-components", type=int,
                        default=DEFAULT_SPECTRAL_COMPONENTS)
    parser.add_argument("--target-purity", type=float, default=None,
                        help="matched-purity target; default = the 'full' "
                             "config's purity at its production threshold")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    inherited = inherit_from_dataset(args, parser)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")
    # thecannon installs its own handler; without this every line is logged
    # twice in the job output.
    logging.getLogger("thecannon").propagate = False

    os.makedirs(args.outdir, exist_ok=True)
    table, nf, ni, cut_counts = load_normalized(args)

    labeled = seismic_state(table).notna().to_numpy()
    logger.info("%d rows, %d with a seismic evolutionary state",
                len(table), int(labeled.sum()))
    if labeled.sum() < 200:
        raise SystemExit("too few seismically labeled stars to ablate")
    tab_lab = table.loc[labeled].reset_index(drop=True)

    masks = stress_masks(tab_lab, args.min_stress_n)
    logger.info("stress subsets: %s",
                ", ".join("%s=%d" % (k, v.sum()) for k, v in masks.items()))

    cv_rows, profile_rows, bias, selections, oof = [], [], {}, {}, {}
    truth = None
    for config in CONFIGS:
        name = config[0]
        p_oof, p_all, y = out_of_fold(config, table, nf, ni, labeled,
                                      args.folds, args.seed, args.n_components)
        truth = y
        oof[name] = p_oof
        thr = PROD_MIN_PROBA if name == "full" else BULGE_MIN_PROBA
        for subset, mask in masks.items():
            for row in threshold_metrics(p_oof, y, tab_lab, THRESHOLD_GRID,
                                         None if subset == "all" else mask):
                row.update(config=name, subset=subset,
                           is_operating_point=bool(row["threshold"] == thr))
                cv_rows.append(row)
        selections[name] = p_all > thr

    cv = pd.DataFrame(cv_rows)[
        ["config", "subset", "threshold", "is_operating_point", "n_scored",
         "n_accepted", "purity", "completeness", "accuracy", "roc_auc",
         "clumpbox_accuracy", "n_clumpbox"]]
    cv.to_csv(os.path.join(args.outdir, "cv_metrics.csv"), index=False)

    # Matched purity: what the bulge feature space costs once contamination is
    # held fixed at the production classifier's level.
    full_op = cv[(cv.config == "full") & (cv.subset == "all")
                 & (cv.threshold == PROD_MIN_PROBA)]
    target = args.target_purity or float(full_op["purity"].iloc[0])
    matched = {}
    for name, p_oof in oof.items():
        m = matched_purity_threshold(p_oof, truth, target)
        matched[name] = m
        own = PROD_MIN_PROBA if name == "full" else BULGE_MIN_PROBA
        points = [("operating", own)]
        if np.isfinite(m["threshold"]) and m["threshold"] != own:
            points.append(("matched_purity" if m["matched"]
                           else "best_purity", m["threshold"]))
        for kind, t in points:
            profile_rows += [dict(config=name, threshold_kind=kind,
                                  threshold=float(t), **r)
                             for r in completeness_profile(
                                 p_oof, truth, tab_lab, t, PROFILE_COLS)]
            bias.setdefault(name, {})[kind] = dict(
                threshold=float(t),
                columns=selection_bias(p_oof, truth, tab_lab, t))

    pd.DataFrame(profile_rows).to_csv(
        os.path.join(args.outdir, "completeness_profile.csv"), index=False)
    flips = pd.DataFrame(selection_flips(selections, "full"))
    flips.to_csv(os.path.join(args.outdir, "selection_flips.csv"), index=False)

    out = pd.DataFrame({"APOGEE_ID": tab_lab["APOGEE_ID"].to_numpy(),
                        "source": (tab_lab["source"].to_numpy()
                                   if "source" in tab_lab else ""),
                        "EvoState": truth})
    for name, p in oof.items():
        out["p_rgb_" + name.replace("+", "_").replace("-", "_")] = p
    out.to_parquet(os.path.join(args.outdir, "oof_probabilities.parquet"),
                   index=False)

    manifest = dict(
        merged_path=os.path.abspath(args.spectra),
        continuum_list=os.path.abspath(args.continuum_list),
        split_seed=args.seed, folds=args.folds, inherited=inherited,
        drop_duplicate_spectra=bool(args.drop_duplicate_spectra),
        primary_only=bool(args.primary_only),
        n_spectral_components=args.n_components,
        production_threshold=PROD_MIN_PROBA, bulge_threshold=BULGE_MIN_PROBA,
        configs=[dict(name=c[0], features=c[1], logg=c[2], inputs=c[3])
                 for c in CONFIGS],
        cuts=cut_counts, n_rows=int(len(table)),
        n_stars=int(table["APOGEE_ID"].nunique()),
        n_labeled=int(labeled.sum()),
        stardata=os.path.dirname(os.path.abspath(stardata.__file__)))
    with open(os.path.join(args.outdir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2, default=float)

    summary = dict(cuts=cut_counts, n_rows=int(len(table)),
                   n_labeled=int(labeled.sum()),
                   n_rgb=int((truth == RGB).sum()),
                   n_heb=int((truth == HEB).sum()),
                   n_stars=int(table["APOGEE_ID"].nunique()),
                   folds=args.folds, split_seed=args.seed,
                   production_threshold=PROD_MIN_PROBA,
                   bulge_threshold=BULGE_MIN_PROBA,
                   matched_purity_target=target, matched=matched,
                   selection_bias=bias,
                   operating_points=cv[cv.is_operating_point]
                   .to_dict("records"))
    with open(os.path.join(args.outdir, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2, default=float)

    print("\n=== out-of-fold vs seismic truth (%d rows / %d stars: "
          "%d RGB, %d HeB; %d star-level folds, seed %d) ==="
          % (labeled.sum(), tab_lab["APOGEE_ID"].nunique(),
             (truth == RGB).sum(), (truth == HEB).sum(), args.folds,
             args.seed))
    print(cv[cv.is_operating_point & (cv.subset == "all")]
          .drop(columns=["subset"]).to_string(index=False))
    stress = cv[cv.is_operating_point & (cv.subset != "all")]
    if len(stress):
        print("\n=== the same probabilities scored on bulge-like calibration "
              "stars ===")
        print(stress[["config", "subset", "n_scored", "n_accepted", "purity",
                      "completeness", "roc_auc"]].to_string(index=False))
    print("\n=== matched purity (target %.3f = 'full' at its operating "
          "point) ===" % target)
    for name, m in matched.items():
        print("  %-16s p(RGB) > %.3f  purity %.3f  completeness %.3f%s"
              % (name, m["threshold"], m["purity"], m["completeness"],
                 "" if m["matched"] else "   <- target UNREACHABLE (best)"))
    print("\n=== selection shift over true RGB stars (accepted vs all) ===")
    for name, kinds in bias.items():
        for kind, entry in kinds.items():
            for col, b in entry["columns"].items():
                print("  %-16s %-14s %-10s mean %+.4f -> %+.4f "
                      "(shift %+.4f, KS p=%.2g)"
                      % (name, kind, col, b["mean_all"], b["mean_accepted"],
                         b["shift"], b["ks_p"]))
    print("\n=== whole-sample selection vs production ===")
    print(flips.to_string(index=False))
    print("\nwrote %s/{cv_metrics,completeness_profile,selection_flips}.csv, "
          "manifest.json, summary.json, oof_probabilities.parquet"
          % args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
