#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diagnose the "labels X and Y are highly correlated (rho = 1.0)" warning by
reproducing exactly the label matrix the Cannon sees for a label set -- the same
file, the same quality/row cuts -- and reporting the correlations and any
duplicate / identical columns.

Usage
-----
    python -m scripts.check_labels \\
        --spectra /path/merged_with_ages_raw.parquet \\
        --labels raw_teff,raw_logg,raw_fe_h,age_L,mg_fe --filter snr_x>100
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse

import numpy as np

try:
    from scripts.train_cannon import (load_spectra, quality_mask,
                                       DEFAULT_SPECTRA)
    from scripts.sweep_config import (filter_mask, per_label_masks,
                                       label_set_row_mask)
except ImportError:
    from train_cannon import load_spectra, quality_mask, DEFAULT_SPECTRA
    from sweep_config import filter_mask, per_label_masks, label_set_row_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spectra", default=DEFAULT_SPECTRA)
    parser.add_argument("--labels", type=lambda s: s.split(","),
                        default=["raw_teff", "raw_logg", "raw_fe_h", "age_L"])
    parser.add_argument("--filter", dest="filters", action="append",
                        default=None)
    parser.add_argument("--no-cuts", action="store_true",
                        help="skip quality/row cuts; correlate the raw columns")
    args = parser.parse_args()

    src, _, flux, _ = load_spectra(args.spectra)
    print("file: {0}\nrows: {1}".format(args.spectra, len(src)))

    # 1) Duplicate column names -> pandas returns a 2-D frame for src[name].
    cols = list(src.columns)
    dupes = sorted({c for c in cols if cols.count(c) > 1})
    print("duplicate column names:", dupes or "none")
    for name in args.labels:
        obj = src[name]
        ndim = getattr(obj, "ndim", 1)
        if ndim > 1:
            print("  !! '{0}' is DUPLICATED ({1} columns) -> ambiguous"
                  .format(name, obj.shape[1]))

    # 2) Build the training subset exactly as the sweep does.
    mask = np.ones(len(src), dtype=bool)
    if not args.no_cuts:
        mask &= np.asarray(quality_mask(src), dtype=bool)
        if args.filters:
            mask &= filter_mask(src, args.filters)
        row = label_set_row_mask(args.labels, per_label_masks(src))
        if row is not None:
            mask &= np.asarray(row, dtype=bool)

    matrix = np.vstack(
        [np.asarray(src[n], dtype=float) for n in args.labels]).T
    finite = np.isfinite(matrix).all(axis=1)
    sub = mask & finite
    labels = matrix[sub]
    print("training stars after cuts: {0}".format(int(sub.sum())))

    # 3) Correlation matrix + identical-column pairs on that subset.
    print("\ncorrelation matrix ({0} stars):".format(labels.shape[0]))
    names = args.labels
    corr = np.corrcoef(labels, rowvar=False)
    header = "        " + " ".join("{0:>9.9s}".format(n) for n in names)
    print(header)
    for i, n in enumerate(names):
        print("{0:>8.8s} ".format(n)
              + " ".join("{0:>9.3f}".format(corr[i, j])
                         for j in range(len(names))))

    print("\nsuspicious pairs (|rho| > 0.98):")
    flagged = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr[i, j]) > 0.98:
                identical = bool(np.allclose(labels[:, i], labels[:, j]))
                print("  {0} vs {1}: rho={2:.4f}{3}".format(
                    names[i], names[j], corr[i, j],
                    "  (IDENTICAL values)" if identical else ""))
                flagged = True
    if not flagged:
        print("  none")


if __name__ == "__main__":
    main()
