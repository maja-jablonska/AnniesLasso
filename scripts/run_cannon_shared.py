#!/usr/bin/env python
"""Train/test The Cannon on the homogenized shared dataset (stardata).

Consumes a dataset directory built by ``stardata.build_dataset`` (stars.parquet
+ splits.csv + manifest.json) so the Cannon sees exactly the same stars and
split as Lux and the bingo BNN. Writes predictions in the shared contract:
``APOGEE_ID, source, split, <label>_truth, <label>_pred, <label>_pred_err,
val_chi2`` — directly consumable by ``stardiag.load_cannon``.

Modes:
  holdout (default): train on split=='train' (+ val with --include-val),
                     predict the held-out test stars.
  oof:               additionally K-fold out-of-fold predictions over the
                     non-test stars using the shared ``fold`` column.

Usage:
    python scripts/run_cannon_shared.py --dataset-dir <dir> \
        [--data-root ~/scr_mk27] [--order 3] [--regularization 100] \
        [--mode holdout|oof] [--out cannon_shared_predictions.parquet]
"""
import argparse
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


def fit_and_predict(labels_df, flux, ivar, dispersion, train_mask, pred_mask,
                    label_names, order, regularization, batch_size):
    import thecannon as tc

    vectorizer = tc.vectorizer.PolynomialVectorizer(label_names, order)
    model = tc.CannonModel(
        labels_df.loc[train_mask, label_names], flux[train_mask],
        ivar[train_mask], vectorizer, dispersion=dispersion,
        regularization=regularization)
    model.train(progressbar=False)
    op_labels, cov, meta = model.test(flux[pred_mask], ivar[pred_mask],
                                      batch_size=batch_size, progressbar=False)
    # cov is in the scaled basis; scale back to physical units
    err = np.sqrt(np.diagonal(cov, axis1=1, axis2=2)) * model._scales
    r_chi_sq = np.array([m["r_chi_sq"] for m in meta])
    return np.asarray(op_labels), np.asarray(err), r_chi_sq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--merged-path", default=None,
                    help="override path to merged_with_ages_raw.parquet")
    ap.add_argument("--continuum-list", default=None)
    ap.add_argument("--labels", nargs="+", default=stardata.CANNON_LABELS)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--regularization", type=float, default=100.0)
    ap.add_argument("--mode", choices=["holdout", "oof"], default="holdout")
    ap.add_argument("--include-val", action="store_true",
                    help="fold the val stars into the training set")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--good-pixel-frac", type=float, default=0.99,
                    help="only used if the dataset has no pixel_mask.npy yet")
    ap.add_argument("--seed", type=int, default=0,
                    help="recorded for provenance; the Cannon's training is "
                         "deterministic given the data, so this changes "
                         "nothing (kept so every runner takes --seed)")
    ap.add_argument("--out", default=None,
                    help="output parquet (default: <dataset-dir>/"
                         "cannon_shared_predictions.parquet)")
    args = ap.parse_args()

    stars, manifest = stardata.load_stars(args.dataset_dir)
    print(f"{len(stars)} rows / {stars['APOGEE_ID'].nunique()} stars "
          f"from {args.dataset_dir}")
    dispersion, flux, ivar = stardata.load_spectra(
        stars, data_root=args.data_root, merged_path=args.merged_path,
        continuum_list=args.continuum_list)

    labels_df = stars[args.labels]
    split = stars["split"].to_numpy()
    fold = stars["fold"].to_numpy()
    train_mask = (split == "train") | (args.include_val & (split == "val"))

    # the SAME wavelength columns Lux uses — otherwise the two spectral
    # methods see different data for the same star
    keep = stardata.shared_pixel_mask(args.dataset_dir, ivar, train_mask,
                                      args.good_pixel_frac)
    print(f"pixel mask: keeping {keep.sum()}/{keep.size} columns")
    dispersion, flux, ivar = dispersion[keep], flux[:, keep], ivar[:, keep]

    pred = np.full((len(stars), len(args.labels)), np.nan)
    perr = np.full_like(pred, np.nan)
    rchi2 = np.full(len(stars), np.nan)
    predicted = np.zeros(len(stars), bool)

    test_mask = split == "test"
    print(f"holdout: training on {train_mask.sum()} rows, "
          f"predicting {test_mask.sum()} test rows")
    p, e, c = fit_and_predict(labels_df, flux, ivar, dispersion, train_mask,
                              test_mask, args.labels, args.order,
                              args.regularization, args.batch_size)
    pred[test_mask], perr[test_mask], rchi2[test_mask] = p, e, c
    predicted |= test_mask

    if args.mode == "oof":
        for k in sorted(set(fold[fold >= 0])):
            tm = train_mask & (fold != k)
            pm = (fold == k) & (split != "test")
            if args.include_val:
                pm &= split != "test"
            print(f"fold {k}: training on {tm.sum()}, predicting {pm.sum()}")
            p, e, c = fit_and_predict(labels_df, flux, ivar, dispersion, tm,
                                      pm, args.labels, args.order,
                                      args.regularization, args.batch_size)
            pred[pm], perr[pm], rchi2[pm] = p, e, c
            predicted |= pm

    out_cols = {"APOGEE_ID": stars["APOGEE_ID"], "split": stars["split"]}
    for c in ("row_id", "source", "is_primary", "is_dup_spectrum",
              "evo_state_source", "rgb_proba", "snr"):
        if c in stars.columns:
            out_cols[c] = stars[c]
    for j, name in enumerate(args.labels):
        out_cols[f"{name}_truth"] = labels_df[name].to_numpy()
        out_cols[f"{name}_pred"] = pred[:, j]
        out_cols[f"{name}_pred_err"] = perr[:, j]
    out_cols["val_chi2"] = rchi2
    out = pd.DataFrame(out_cols)[predicted].reset_index(drop=True)

    out_path = Path(args.out) if args.out else \
        Path(args.dataset_dir) / "cannon_shared_predictions.parquet"
    out.to_parquet(out_path)
    print(f"wrote {len(out)} predictions to {out_path}")


if __name__ == "__main__":
    main()
