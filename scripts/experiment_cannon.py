#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run a grid of Cannon "experiments" and report what was recovered.

Where :mod:`scripts.sweep_cannon` cross-validates over the *model* knobs (label
set, polynomial order, regularization), this driver adds the two *data* knobs --
an S/N cutoff and arbitrary row filters on the table columns -- and, for every
grid point, also:

  * reports the recovered spread (per-label bias / scatter / RMSE) on a held-out
    validation set,
  * reconstructs the validation spectra by forward-modelling the recovered
    labels through the trained model,
  * plots a few observed-vs-reconstructed spectra, and
  * reports the reconstruction chi-squared (reduced, at both the recovered and
    the reference labels).

Each grid point is therefore one experiment over

    (filter) x (snr_cutoff) x (label_set) x (order) x (regularization)

and writes a row to a tidy CSV plus a one-to-one spread figure and a
reconstruction figure.

JAX device selection is via the ``JAX_PLATFORMS`` environment variable.

Usage
-----
On the real data::

    python -m scripts.experiment_cannon \\
        --spectra /path/cleaned_ages.parquet \\
        --continuum-list /path/continuum.list \\
        --labels raw_teff,raw_logg,raw_fe_h,mg_fe,ce_fe,log_age_L \\
        --snr-cutoffs 100,150,200 \\
        --filter "EvoState == 2" \\
        --filter "EvoState in [1, 2]" \\
        --orders 1,2 --regularizations 0,1e-6,1e2 \\
        --output-dir results/

Quick self-contained smoke test (no data files, runs in seconds)::

    JAX_PLATFORMS=cpu python -m scripts.experiment_cannon --demo
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import logging
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")              # headless: only saves figures
import matplotlib.pyplot as plt

import thecannon as tc

# Work both as a package module (`python -m scripts.experiment_cannon`) and when
# run directly from the scripts/ directory.
try:
    from scripts.train_cannon import (load_spectra, normalize_spectra,
                                       make_split, DEFAULT_DATA_DIR,
                                       DEFAULT_LABELS)
    from scripts.sweep_cannon import spread_figure
except ImportError:
    from train_cannon import (load_spectra, normalize_spectra, make_split,
                              DEFAULT_DATA_DIR, DEFAULT_LABELS)
    from sweep_cannon import spread_figure

logger = logging.getLogger("thecannon.experiment")


# --------------------------------------------------------------------------- #
#  Row filtering (the data knobs: S/N cutoff + arbitrary column queries)       #
# --------------------------------------------------------------------------- #

def filter_mask(table, snr_column=None, snr_cutoff=None, query=None):
    """
    Boolean mask over the rows of ``table`` (a pandas DataFrame) selecting stars
    that pass an optional ``snr_column > snr_cutoff`` cut *and* an optional
    pandas ``query`` expression (e.g. ``"EvoState == 2"``). Either may be
    ``None`` to skip it.
    """
    n = len(table)
    mask = np.ones(n, dtype=bool)

    if snr_cutoff is not None:
        if snr_column not in getattr(table, "columns", []):
            raise KeyError("snr column {0!r} not in table; pass --snr-column"
                           .format(snr_column))
        mask &= np.asarray(table[snr_column], dtype=float) > float(snr_cutoff)

    if query:
        # table.query returns the surviving rows; map back to a positional mask.
        kept_index = table.query(query).index
        mask &= np.asarray(table.index.isin(kept_index))

    return mask


def _tag(s):
    """ Filesystem-safe short tag from an arbitrary string. """
    return re.sub(r"[^0-9A-Za-z]+", "_", str(s)).strip("_") or "none"


# --------------------------------------------------------------------------- #
#  Reconstruction + chi-squared                                                #
# --------------------------------------------------------------------------- #

def reconstruct(model, label_array):
    """ Forward-model labels -> (N, P) flux through the trained model. """
    return np.atleast_2d(np.asarray(model(label_array)))


def reconstruction_chisq(flux, ivar, model_flux, n_labels):
    """
    Per-star chi-squared between observed and reconstructed flux, using only the
    measured pixels (``ivar > 0``). Returns ``(chi2, reduced_chi2, n_good)``;
    the reduced value uses ``dof = n_good - n_labels - 1`` (matching the test
    step's convention).
    """
    flux = np.atleast_2d(flux)
    ivar = np.atleast_2d(ivar)
    model_flux = np.atleast_2d(model_flux)

    good = ivar > 0
    resid2 = np.where(good, (flux - model_flux) ** 2 * ivar, 0.0)
    chi2 = resid2.sum(axis=1)
    n_good = good.sum(axis=1)
    dof = np.maximum(1, n_good - n_labels - 1)
    return chi2, chi2 / dof, n_good


# --------------------------------------------------------------------------- #
#  Figures                                                                     #
# --------------------------------------------------------------------------- #

def _default_window(dispersion):
    """ A ~120-pixel window near the middle of the grid, for legible plots. """
    P = len(dispersion)
    lo = int(0.45 * P)
    hi = min(P - 1, lo + 120)
    return float(dispersion[lo]), float(dispersion[hi])


def spectra_figure(dispersion, flux, ivar, recovered_flux, reference_flux,
    star_indices, red_chi_sq, wmin, wmax, title=None):
    """
    Observed vs reconstructed spectra for a handful of validation stars, one row
    per star, zoomed to ``[wmin, wmax]``. The observed flux is drawn with its
    1-sigma band; the recovered-label reconstruction is overplotted (and the
    reference-label reconstruction, if given, as a dotted line).
    """
    dispersion = np.asarray(dispersion)
    win = (dispersion >= wmin) & (dispersion <= wmax)
    if not win.any():
        win = np.ones(len(dispersion), dtype=bool)

    k = len(star_indices)
    fig, axes = plt.subplots(k, 1, figsize=(10, 2.1 * k), squeeze=False,
                             sharex=True)
    axes = axes.ravel()

    for ax, s in zip(axes, star_indices):
        f = np.asarray(flux[s])[win]
        iv = np.asarray(ivar[s])[win]
        sigma = np.where(iv > 0, 1.0 / np.sqrt(np.where(iv > 0, iv, 1.0)), np.nan)
        wl = dispersion[win]

        ax.fill_between(wl, f - sigma, f + sigma, color="0.8", step="mid",
                        label="observed 1$\\sigma$")
        ax.step(wl, f, where="mid", color="k", lw=0.8, label="observed")
        ax.plot(wl, np.asarray(recovered_flux[s])[win], color="C3", lw=1.0,
                label="model (recovered labels)")
        if reference_flux is not None:
            ax.plot(wl, np.asarray(reference_flux[s])[win], color="C0", lw=1.0,
                    ls=":", label="model (reference labels)")
        ax.set_ylabel("flux")
        ax.text(0.01, 0.04, "star {0}: reduced $\\chi^2$={1:.2f}".format(
            s, red_chi_sq[s]), transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left")

    axes[0].legend(fontsize=7, ncol=4, loc="upper right")
    axes[-1].set_xlabel("wavelength")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  One experiment (one grid point)                                             #
# --------------------------------------------------------------------------- #

def run_one(table, dispersion, norm_flux, norm_ivar, label_names, order, reg,
    base_mask, train_frac, validate_frac, seed, output_dir, tag,
    n_plot=3, wmin=None, wmax=None, label_err=None):
    """
    Train + validate one (filter, cutoff, label_set, order, reg) combination,
    then reconstruct the validation spectra and write the spread + reconstruction
    figures. Returns a tidy metric row.
    """
    label_array = np.vstack(
        [np.asarray(table[n], dtype=float) for n in label_names]).T

    # Combine the data filter with per-label finiteness, then split.
    finite = np.isfinite(label_array).all(axis=1)
    use = base_mask & finite
    n_use = int(use.sum())
    if n_use < 5:
        raise ValueError("only {0} usable stars after filtering".format(n_use))

    idx = np.where(use)[0]
    sub_labels = label_array[idx]
    sub_flux = np.asarray(norm_flux)[idx]
    sub_ivar = np.asarray(norm_ivar)[idx]
    sub_err = None if label_err is None else np.asarray(label_err)[idx]

    train_set, validate_set = make_split(
        sub_labels, train_frac, validate_frac, seed)
    if train_set.sum() == 0 or validate_set.sum() == 0:
        raise ValueError("empty train/validation fold; adjust fractions/seed")

    vectorizer = tc.vectorizer.PolynomialVectorizer(
        label_names=label_names, order=order)
    model = tc.CannonModel(
        sub_labels[train_set], sub_flux[train_set], sub_ivar[train_set],
        vectorizer, dispersion=dispersion, regularization=reg,
        training_set_label_err=(None if sub_err is None else sub_err[train_set]))
    model.train(progressbar=False)

    # Test step: recover labels for the held-out spectra.
    val_flux = sub_flux[validate_set]
    val_ivar = sub_ivar[validate_set]
    truth = sub_labels[validate_set]
    recovered, _, meta = model.test(val_flux, val_ivar, progressbar=False)
    recovered = np.asarray(recovered)
    test_r_chi_sq = np.array([m["r_chi_sq"] for m in meta], dtype=float)

    # Reconstruct the validation spectra from the recovered (and reference)
    # labels and score the reconstruction.
    L = len(label_names)
    recon_recovered = reconstruct(model, recovered)
    recon_reference = reconstruct(model, truth)
    _, red_chi_recovered, _ = reconstruction_chisq(
        val_flux, val_ivar, recon_recovered, L)
    _, red_chi_reference, _ = reconstruction_chisq(
        val_flux, val_ivar, recon_reference, L)

    # --- metric row ---
    residual = recovered - truth
    row = dict(tag=tag, label_set="+".join(label_names), n_labels=L,
               order=order, regularization=reg,
               n_train=int(train_set.sum()), n_val=int(validate_set.sum()),
               n_used=n_use)
    scatters = []
    for i, name in enumerate(label_names):
        r = residual[:, i]
        r = r[np.isfinite(r)]
        row["bias_{0}".format(name)] = float(np.mean(r)) if r.size else np.nan
        row["scatter_{0}".format(name)] = float(np.std(r)) if r.size else np.nan
        row["rmse_{0}".format(name)] = \
            float(np.sqrt(np.mean(r ** 2))) if r.size else np.nan
        scatters.append(row["scatter_{0}".format(name)])
    row["mean_scatter"] = float(np.nanmean(scatters))
    row["median_test_r_chi_sq"] = float(np.nanmedian(test_r_chi_sq))
    row["median_recon_r_chi_sq_recovered"] = float(np.nanmedian(red_chi_recovered))
    row["median_recon_r_chi_sq_reference"] = float(np.nanmedian(red_chi_reference))
    try:
        row["frac_in_hull"] = float(np.mean(model.in_convex_hull(truth)))
    except Exception:
        row["frac_in_hull"] = np.nan

    # --- figures ---
    os.makedirs(output_dir, exist_ok=True)
    title = "{0} | order={1} | reg={2:g}".format(row["label_set"], order, reg)

    fig = spread_figure(recovered, truth, label_names, title=title)
    if fig is not None:
        fig.savefig(os.path.join(output_dir, "spread__{0}.png".format(tag)),
                    dpi=150)
        plt.close(fig)

    if wmin is None or wmax is None:
        wmin, wmax = _default_window(dispersion)
    # Plot the n_plot validation stars spanning the reconstruction-chi2 range
    # (best, worst, and evenly spaced in between) so the figure is representative.
    finite_chi = np.where(np.isfinite(red_chi_recovered))[0]
    if finite_chi.size:
        ordered = finite_chi[np.argsort(red_chi_recovered[finite_chi])]
        pick = np.unique(np.linspace(
            0, len(ordered) - 1, min(n_plot, len(ordered))).astype(int))
        star_indices = ordered[pick].tolist()
        fig = spectra_figure(
            dispersion, val_flux, val_ivar, recon_recovered, recon_reference,
            star_indices, red_chi_recovered, wmin, wmax,
            title="reconstruction | " + title)
        if fig is not None:
            fig.savefig(
                os.path.join(output_dir, "reconstruction__{0}.png".format(tag)),
                dpi=150)
            plt.close(fig)

    logger.info("%s -> mean_scatter=%.4f recon_chi2(recovered)=%.2f "
                "recon_chi2(reference)=%.2f", tag, row["mean_scatter"],
                row["median_recon_r_chi_sq_recovered"],
                row["median_recon_r_chi_sq_reference"])
    return row


# --------------------------------------------------------------------------- #
#  The experiment grid                                                         #
# --------------------------------------------------------------------------- #

def experiment(table, dispersion, norm_flux, norm_ivar, label_sets, orders,
    regularizations, snr_cutoffs=(None,), filters=(None,), snr_column="snr",
    train_frac=0.1, validate_frac=0.1, seed=888, output_dir=".",
    n_plot=3, wmin=None, wmax=None, label_err=None):
    """
    Loop the full (filter x snr_cutoff x label_set x order x reg) grid, calling
    :func:`run_one` for each, and collect a tidy results table. Failures are
    recorded (status column) without stopping the grid.
    """
    rows = []
    n_combo = (len(filters) * len(snr_cutoffs) * len(label_sets)
               * len(orders) * len(regularizations))
    logger.info("Running %d experiments", n_combo)

    i = 0
    for query in filters:
        for cutoff in snr_cutoffs:
            base_mask = filter_mask(table, snr_column, cutoff, query)
            logger.info("filter=%r snr>%s -> %d stars", query, cutoff,
                        int(base_mask.sum()))
            for label_set in label_sets:
                for order in orders:
                    for reg in regularizations:
                        i += 1
                        tag = "{0}__snr{1}__{2}__o{3}__reg{4:g}".format(
                            _tag(query), "none" if cutoff is None else cutoff,
                            _tag("+".join(label_set)), order, reg)
                        meta = dict(filter=("" if query is None else query),
                                    snr_cutoff=("" if cutoff is None else cutoff))
                        try:
                            row = run_one(
                                table, dispersion, norm_flux, norm_ivar,
                                list(label_set), order, reg, base_mask,
                                train_frac, validate_frac, seed, output_dir, tag,
                                n_plot=n_plot, wmin=wmin, wmax=wmax,
                                label_err=label_err)
                            row.update(meta)
                            row["status"] = "ok"
                        except Exception as exc:           # keep the grid alive
                            logger.warning("[%d/%d] %s FAILED: %s",
                                           i, n_combo, tag, exc)
                            row = dict(tag=tag, label_set="+".join(label_set),
                                       order=order, regularization=reg,
                                       status="error: {0}".format(exc))
                            row.update(meta)
                        rows.append(row)

    csv_path = os.path.join(output_dir, "experiment_results.csv")
    os.makedirs(output_dir, exist_ok=True)
    _write_csv(rows, csv_path)
    logger.info("Wrote %d rows to %s", len(rows), csv_path)

    # Console summary of the recovered spread + reconstruction chi-squared.
    print("\n=== experiment summary ===")
    cols = ("tag", "n_train", "n_val", "mean_scatter",
            "median_recon_r_chi_sq_recovered",
            "median_recon_r_chi_sq_reference", "status")
    print("  ".join("{0:>10s}".format(c[:18]) for c in cols))
    for r in rows:
        print("  ".join("{0:>10}".format(
            ("{0:.4g}".format(r[c]) if isinstance(r.get(c), float)
             else str(r.get(c, ""))[:18])) for c in cols))

    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


def _write_csv(rows, path):
    """ CSV with a stable union-of-keys header. """
    import csv
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------------- #
#  Demo (self-contained synthetic data with snr + a population column)         #
# --------------------------------------------------------------------------- #

def _demo(output_dir):
    """
    A tiny synthetic data set with a known quadratic label dependence, plus
    ``snr`` and ``pop`` columns, so the S/N cutoff and filter knobs are
    exercised end-to-end without any data files. The flux is already normalized.
    """
    import pandas as pd

    rng = np.random.default_rng(0)
    names = ["teff", "logg", "feh"]
    N, P = 300, 200
    x = rng.uniform(-1, 1, size=(N, len(names)))

    dispersion = np.linspace(15100.0, 16900.0, P)
    A = rng.normal(0, 0.05, size=(P, len(names)))
    B = rng.normal(0, 0.01, size=(P, len(names)))
    flux = 1.0 + x @ A.T + (x ** 2) @ B.T

    snr = rng.uniform(50, 400, size=N)
    noise = rng.normal(0, 1.0 / snr[:, None], size=(N, P))
    flux = flux + noise
    ivar = np.broadcast_to((snr[:, None]) ** 2, (N, P)).astype(float)

    table = pd.DataFrame({names[0]: 4800 + 600 * x[:, 0],
                          names[1]: 2.5 + 1.0 * x[:, 1],
                          names[2]: -0.2 + 0.4 * x[:, 2],
                          "snr": snr,
                          "pop": rng.integers(0, 3, size=N)})

    print("Demo: {0} synthetic stars x {1} pixels (snr in [50, 400], "
          "pop in {{0,1,2}})".format(N, P))
    return experiment(
        table, dispersion, flux, ivar,
        label_sets=[tuple(names)],
        orders=[1, 2],
        regularizations=[0.0, 1e-4],
        snr_cutoffs=[None, 150],
        filters=[None, "pop == 2"],
        snr_column="snr",
        train_frac=0.5, validate_frac=0.5, seed=888,
        output_dir=output_dir, n_plot=3)


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def _floats(s):
    return [float(x) for x in s.split(",")]


def _ints(s):
    return [int(x) for x in s.split(",")]


def _cutoffs(s):
    # "none" or empty -> no cutoff; otherwise a list of floats.
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        out.append(None if tok.lower() in ("", "none") else float(tok))
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spectra",
                        default=os.path.join(DEFAULT_DATA_DIR,
                                             "cleaned_ages.parquet"),
                        help="parquet table of spectra + labels")
    parser.add_argument("--continuum-list",
                        default=os.path.join(DEFAULT_DATA_DIR, "continuum.list"),
                        help="text file of continuum pixel indices")
    parser.add_argument("--labels", type=lambda s: s.split(","),
                        default=DEFAULT_LABELS,
                        help="comma-separated label column names (one label set)")
    parser.add_argument("--label-set", dest="label_sets", action="append",
                        type=lambda s: tuple(s.split(",")), default=None,
                        help="explicit label set (repeatable); overrides --labels")
    parser.add_argument("--orders", type=_ints, default=[1, 2],
                        help="comma-separated polynomial orders")
    parser.add_argument("--regularizations", type=_floats,
                        default=[0.0], help="comma-separated L1 strengths")
    parser.add_argument("--snr-cutoffs", type=_cutoffs, default=[None],
                        help="comma-separated S/N cutoffs (keep snr > cutoff); "
                             "use 'none' for no cut, e.g. none,100,150,200")
    parser.add_argument("--snr-column", default="snr",
                        help="name of the S/N column in the table")
    parser.add_argument("--filter", dest="filters", action="append",
                        default=None,
                        help="pandas query string to keep (repeatable), e.g. "
                             "\"EvoState == 2\"; omit for no filter")
    parser.add_argument("--label-err", dest="label_err_cols",
                        type=lambda s: s.split(","), default=None,
                        help="comma-separated per-label 1-sigma error columns "
                             "(aligned to the label set) to train with label "
                             "uncertainties")
    parser.add_argument("--train-frac", type=float, default=0.1)
    parser.add_argument("--validate-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--n-plot", type=int, default=3,
                        help="number of example reconstructed spectra to plot")
    parser.add_argument("--plot-wmin", type=float, default=None,
                        help="min wavelength for the reconstruction plots")
    parser.add_argument("--plot-wmax", type=float, default=None,
                        help="max wavelength for the reconstruction plots")
    parser.add_argument("--output-dir", default="experiment_results")
    parser.add_argument("--demo", action="store_true",
                        help="run a self-contained synthetic experiment")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.demo:
        _demo(args.output_dir)
        return

    table, dispersion, flux, ivar = load_spectra(args.spectra)
    norm_flux, norm_ivar = normalize_spectra(
        dispersion, flux, ivar, args.continuum_list)

    label_sets = args.label_sets or [tuple(args.labels)]
    filters = args.filters if args.filters else [None]

    # Optional per-label uncertainties: pull the named error columns into an
    # (N, K) array aligned with the (single) label set.
    label_err = None
    if args.label_err_cols:
        label_set0 = list(label_sets[0])
        if len(args.label_err_cols) != len(label_set0):
            raise ValueError("--label-err must list one error column per label "
                             "in the (first) label set")
        label_err = np.vstack(
            [np.asarray(table[c], dtype=float) for c in args.label_err_cols]).T

    experiment(
        table, dispersion, norm_flux, norm_ivar,
        label_sets=label_sets, orders=args.orders,
        regularizations=args.regularizations, snr_cutoffs=args.snr_cutoffs,
        filters=filters, snr_column=args.snr_column,
        train_frac=args.train_frac, validate_frac=args.validate_frac,
        seed=args.seed, output_dir=args.output_dir, n_plot=args.n_plot,
        wmin=args.plot_wmin, wmax=args.plot_wmax, label_err=label_err)


if __name__ == "__main__":
    main()
