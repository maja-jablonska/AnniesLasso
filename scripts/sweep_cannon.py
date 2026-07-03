#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-validated hyper-parameter sweep for The Cannon.

This explores the knobs that matter for a Cannon model -- the label set, the
polynomial order of the vectorizer, the L1 regularization strength, and
(optionally) pixel censoring -- and scores each combination by k-fold
cross-validation. For every grid point it trains on the training folds, runs the
test step on the held-out fold, and reports how well the held-out labels are
recovered. For every label it records the classical (mean/std) and robust
(median/MAD) bias and scatter, the catastrophic-outlier fraction, a
dimensionless ``r2`` goodness score, and the uncertainty-calibration "pull"
statistics (residual / formal error). These are summarized per grid point by
``mean_r2`` (the recommended, unit-free ranking key), ``mean_sigma_mad`` and
``mean_pull_std``, alongside the median reduced chi-squared, the fraction of
held-out stars inside the convex hull of the training labels, and the trained
model's complexity (``theta_frac_zero`` L1 sparsity and ``median_s2``).

The same fold assignment (fixed RNG seed) is reused for every grid point so the
comparison between hyper-parameters is not confounded by split noise.

Usage
-----
As a module (the intended entry point -- real data comes in too many formats for
a one-size CLI)::

    from scripts.sweep_cannon import sweep

    # `labels` is a mapping {label_name: (N,) array}; a numpy structured/record
    # array or an astropy Table works directly. `flux`/`ivar` are (N, P) arrays
    # of pseudo-continuum-normalized flux and inverse variance.
    results = sweep(
        labels, flux, ivar, dispersion,
        label_sets=[("TEFF", "LOGG", "FE_H"),
                    ("TEFF", "LOGG", "FE_H", "MG_FE")],
        orders=[1, 2],
        regularizations=[0.0, 1e2, 1e3, 1e4],
        n_splits=5,
        output="sweep_results.csv",
    )

For a quick end-to-end smoke test on the bundled golden data::

    python -m scripts.sweep_cannon --demo
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import itertools
import logging

import numpy as np

from thecannon.model import CannonModel
from thecannon.vectorizer.polynomial import PolynomialVectorizer

try:
    from scripts.sweep_config import label_set_row_mask
except ImportError:
    from sweep_config import label_set_row_mask

try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib
    matplotlib.use("Agg")          # headless: the sweep only saves figures
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

logger = logging.getLogger("thecannon.sweep")


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _label_matrix(labels, names):
    """
    Return an ``(N, len(names))`` float array for the requested label names,
    accepting a mapping (dict / structured array / astropy Table) keyed by name.
    """
    try:
        columns = [np.asarray(labels[name], dtype=float) for name in names]
    except (KeyError, IndexError, ValueError) as exc:
        raise KeyError(
            "could not extract labels {0} from the provided container "
            "({1})".format(list(names), exc))
    return np.vstack(columns).T


def _kfold_indices(n, n_splits, seed):
    """ Yield ``(train_idx, test_idx)`` pairs for shuffled k-fold splits. """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > n:
        raise ValueError(
            "n_splits ({0}) cannot exceed the number of stars ({1})"
            .format(n_splits, n))

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, n_splits)
    for k in range(n_splits):
        test_idx = folds[k]
        train_idx = np.concatenate(
            [folds[j] for j in range(n_splits) if j != k])
        yield np.sort(train_idx), np.sort(test_idx)


# --------------------------------------------------------------------------- #
#  Cross-validation for a single hyper-parameter combination                   #
# --------------------------------------------------------------------------- #

def cross_validate(label_array, flux, ivar, dispersion, label_names, order,
    regularization, censors=None, n_splits=5, seed=0, train_kwds=None,
    test_kwds=None, age_reliability=None):
    """
    Run k-fold cross-validation for one hyper-parameter combination.

    :param label_array:
        An ``(N, K)`` array of reference labels aligned with ``label_names``.

    :param flux, ivar:
        ``(N, P)`` arrays of normalized flux and inverse variance.

    :param dispersion:
        The ``(P,)`` dispersion array (may be ``None``).

    :param label_names:
        The labels to fit (defines the vectorizer and the columns used).

    :param order:
        The polynomial order of the vectorizer.

    :param regularization:
        The L1 regularization strength (``0`` uses the closed-form fast path).

    :param censors: [optional]
        A :class:`thecannon.censoring.Censors` object (or ``None``).

    :param age_reliability: [optional]
        The ``{age_col: bool_array}`` mapping from
        :func:`scripts.sweep_config.age_reliability_masks` (row-aligned to the
        input arrays). When given, this label set is restricted to the stars for
        which every age column it fits is flagged reliable (no restriction when
        it fits no age column).

    :returns:
        A dict of held-out arrays -- ``recovered`` ``(N, K)``, ``reference``
        ``(N, K)``, ``residual`` ``(N, K)``, ``formal_err`` ``(N, K)`` (the
        model's own per-label 1-sigma uncertainties), ``r_chi_sq`` ``(N,)`` and
        ``in_hull`` ``(N,)`` -- plus the fold-averaged model-complexity scalars
        ``theta_frac_zero`` and ``median_s2``. ``N`` is the post-age-cut count.
    """

    # Restrict to the stars for which this set's age column(s) are reliable.
    row_mask = label_set_row_mask(label_names, age_reliability)
    if row_mask is not None:
        row_mask = np.asarray(row_mask, dtype=bool)
        logger.info("age-reliability cut for %s: %d/%d stars",
                    "+".join(label_names), int(row_mask.sum()), row_mask.size)
        label_array = label_array[row_mask]
        flux = flux[row_mask]
        ivar = ivar[row_mask]

    N, K = label_array.shape
    recovered = np.full((N, K), np.nan, dtype=float)
    formal_err = np.full((N, K), np.nan, dtype=float)
    r_chi_sq = np.full(N, np.nan, dtype=float)
    in_hull = np.zeros(N, dtype=bool)

    # Model-complexity diagnostics are per-fold properties of the trained model;
    # collect them and average across folds.
    theta_frac_zero, median_s2 = [], []

    # One vectorizer for every fold (it is stateless), and no per-fold progress
    # bars: tqdm host callbacks get baked into the compiled scan, which would
    # force a recompile per fold instead of reusing one program on the device.
    vectorizer = PolynomialVectorizer(label_names=label_names, order=order)
    train_kwds = {"progressbar": False, **(train_kwds or {})}
    test_kwds = {"progressbar": False, **(test_kwds or {})}

    for train_idx, test_idx in _kfold_indices(N, n_splits, seed):

        model = CannonModel(
            label_array[train_idx], flux[train_idx], ivar[train_idx],
            vectorizer, dispersion=dispersion, regularization=regularization,
            censors=censors)
        model.train(**train_kwds)

        # Record the trained model's complexity: the L1 sparsity (fraction of
        # theta coefficients driven to zero) and the learned intrinsic pixel
        # scatter, so the regularization axis of the sweep is interpretable.
        theta = np.asarray(model.theta)
        theta_frac_zero.append(float(np.mean(np.abs(theta) < 1e-12)))
        median_s2.append(float(np.nanmedian(model.s2)))

        # Keep the covariance: its diagonal is the model's formal per-label
        # variance, which powers the uncertainty-calibration ("pull") metrics.
        # NOTE: `test` optimizes in the *scaled* label basis and returns `cov`
        # there (fitting.py: cov = inv(J^T J) with J wrt scaled params), while
        # `op_labels` is converted back to physical units. Rescale the formal
        # errors to physical units too (sigma_phys = scales * sigma_scaled) so
        # they are commensurate with the residuals; without this the pull is
        # inflated by ~scales (e.g. hundreds for TEFF).
        op_labels, cov, meta = model.test(
            flux[test_idx], ivar[test_idx], **test_kwds)
        scales = np.asarray(model._scales)                 # (L,)

        recovered[test_idx] = op_labels
        with np.errstate(invalid="ignore"):
            sigma_scaled = np.sqrt(
                np.clip(np.einsum("sii->si", np.asarray(cov)), 0.0, None))
            formal_err[test_idx] = sigma_scaled * scales
        r_chi_sq[test_idx] = [m["r_chi_sq"] for m in meta]

        # Convex-hull membership flags held-out stars that require extrapolation
        # (Delaunay can fail for degenerate / high-dimensional label sets).
        try:
            in_hull[test_idx] = model.in_convex_hull(label_array[test_idx])
        except Exception as exc:                       # pragma: no cover
            logger.debug("convex-hull test skipped: %s", exc)
            in_hull[test_idx] = False

    return dict(recovered=recovered, reference=label_array,
                residual=recovered - label_array, formal_err=formal_err,
                r_chi_sq=r_chi_sq, in_hull=in_hull,
                theta_frac_zero=float(np.mean(theta_frac_zero)),
                median_s2=float(np.mean(median_s2)))


def _summarize(cv, label_names, catastrophic_nsigma=5.0):
    """
    Reduce the per-star cross-validation arrays to a tidy metric row.

    For every label this reports both the classical (mean/std) and robust
    (median/MAD) bias and scatter, the catastrophic-outlier fraction (residuals
    beyond ``catastrophic_nsigma`` robust sigma), a dimensionless goodness score
    ``r2`` (explained variance), and -- when formal errors are available -- the
    uncertainty-calibration "pull" statistics (residual / formal error; mean ~0
    and std ~1 for a well-calibrated model). Aggregates and the fold-averaged
    model-complexity scalars round out the row.
    """
    residual = cv["residual"]
    reference = cv.get("reference")
    formal_err = cv.get("formal_err")
    row = {}
    scatters, sigma_mads, r2s, pull_stds = [], [], [], []

    for i, name in enumerate(label_names):
        r = residual[:, i]

        # --- classical (mean / std) ---
        row["bias_{0}".format(name)] = float(np.nanmean(r))
        row["scatter_{0}".format(name)] = float(np.nanstd(r))
        row["rmse_{0}".format(name)] = float(np.sqrt(np.nanmean(r ** 2)))
        scatters.append(row["scatter_{0}".format(name)])

        # --- robust (median / MAD), immune to catastrophic failures ---
        med = float(np.nanmedian(r))
        mad = float(1.4826 * np.nanmedian(np.abs(r - med)))
        row["median_bias_{0}".format(name)] = med
        row["sigma_mad_{0}".format(name)] = mad
        row["frac_catastrophic_{0}".format(name)] = (
            float(np.nanmean(np.abs(r - med) > catastrophic_nsigma * mad))
            if mad > 0 else 0.0)
        sigma_mads.append(mad)

        # --- dimensionless goodness (R^2 = explained variance) ---
        if reference is not None:
            ref = reference[:, i]
            good = np.isfinite(r) & np.isfinite(ref)
            ss_tot = float(np.sum((ref[good] - np.mean(ref[good])) ** 2)) \
                if good.sum() > 1 else 0.0
            r2 = float(1.0 - np.sum(r[good] ** 2) / ss_tot) \
                if ss_tot > 0 else float("nan")
        else:
            r2 = float("nan")
        row["r2_{0}".format(name)] = r2
        r2s.append(r2)

        # --- uncertainty calibration (pull = residual / formal error) ---
        if formal_err is not None:
            fe = formal_err[:, i]
            with np.errstate(invalid="ignore", divide="ignore"):
                pull = np.where(fe > 0, r / fe, np.nan)
            row["pull_mean_{0}".format(name)] = float(np.nanmean(pull))
            pull_std = float(np.nanstd(pull))
            row["pull_std_{0}".format(name)] = pull_std
            med_fe = float(np.nanmedian(fe))
            row["err_ratio_{0}".format(name)] = (
                row["scatter_{0}".format(name)] / med_fe
                if med_fe > 0 else float("nan"))
            pull_stds.append(pull_std)

    # --- aggregates ---
    row["mean_scatter"] = float(np.mean(scatters))       # kept for continuity
    row["mean_sigma_mad"] = float(np.mean(sigma_mads))   # robust analogue
    row["mean_r2"] = float(np.nanmean(r2s))              # recommended ranking key
    if pull_stds:
        row["mean_pull_std"] = float(np.nanmean(pull_stds))
    row["median_r_chi_sq"] = float(np.nanmedian(cv["r_chi_sq"]))
    row["frac_in_hull"] = float(np.mean(cv["in_hull"]))

    # --- model complexity (fold-averaged) ---
    if "theta_frac_zero" in cv:
        row["theta_frac_zero"] = float(cv["theta_frac_zero"])
    if "median_s2" in cv:
        row["median_s2"] = float(cv["median_s2"])

    row["n_test"] = int(np.sum(np.isfinite(cv["recovered"][:, 0])))
    return row


# --------------------------------------------------------------------------- #
#  Weights & Biases logging (optional, offline by default)                     #
# --------------------------------------------------------------------------- #

def _init_wandb_run(project, entity, group, mode, run_dir, name, config):
    """
    Start a single (offline by default) W&B run for one grid point. Returns the
    run object, or ``None`` if W&B is unavailable or initialization fails.
    """
    if wandb is None:
        return None
    try:
        return wandb.init(
            project=project, entity=entity, group=group, name=name,
            mode=mode, dir=run_dir, config=config, reinit=True,
            settings=wandb.Settings(silent=True))
    except Exception as exc:                            # pragma: no cover
        logger.warning("wandb.init failed (%s); continuing without logging",
                       exc)
        return None


def _log_wandb_run(run, row, cv, label_names):
    """ Log scalar metrics and per-label residual histograms to a W&B run. """
    if run is None:
        return
    # Scalar metrics only (the string/config fields already live in run.config).
    metrics = {k: v for k, v in row.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}

    if cv is not None:
        residual = cv["residual"]
        formal_err = cv.get("formal_err")
        for i, name in enumerate(label_names):
            finite = residual[:, i][np.isfinite(residual[:, i])]
            if finite.size:
                try:
                    metrics["residual_hist/{0}".format(name)] = \
                        wandb.Histogram(finite)
                except Exception:                      # pragma: no cover
                    pass
            # The pull distribution should be ~unit-Gaussian if the model's
            # formal errors are well calibrated; a histogram makes that visible.
            if formal_err is not None:
                fe = formal_err[:, i]
                with np.errstate(invalid="ignore", divide="ignore"):
                    pull = np.where(fe > 0, residual[:, i] / fe, np.nan)
                pull = pull[np.isfinite(pull)]
                if pull.size:
                    try:
                        metrics["pull_hist/{0}".format(name)] = \
                            wandb.Histogram(pull)
                    except Exception:                  # pragma: no cover
                        pass

    run.log(metrics)
    run.summary["status"] = row.get("status", "ok")


# --------------------------------------------------------------------------- #
#  Spread figures                                                              #
# --------------------------------------------------------------------------- #

def spread_figure(recovered, reference, label_names, title=None):
    """
    Build a one-to-one (recovered vs reference) figure with one panel per label,
    annotated with the held-out bias and scatter (spread). Returns a matplotlib
    ``Figure`` (or ``None`` if matplotlib is unavailable).
    """
    if plt is None:
        return None

    K = len(label_names)
    ncols = min(K, 3)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(3.4 * ncols, 3.2 * nrows))
    axes = axes.flatten()

    for i, name in enumerate(label_names):
        ax = axes[i]
        x, y = reference[:, i], recovered[:, i]
        good = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[good], y[good], s=6, c="k", alpha=0.5)

        if good.any():
            lo = float(min(x[good].min(), y[good].min()))
            hi = float(max(x[good].max(), y[good].max()))
        else:
            lo, hi = 0.0, 1.0
        ax.plot([lo, hi], [lo, hi], ":", c="#666666", zorder=-1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        resid = (y - x)[good]
        bias = float(np.mean(resid)) if resid.size else float("nan")
        scatter = float(np.std(resid)) if resid.size else float("nan")
        ax.set_title("{0}\nbias={1:.3g}  sigma={2:.3g}".format(
            name, bias, scatter), fontsize=9)
        ax.set_xlabel("reference")
        ax.set_ylabel("recovered")

    for j in range(K, len(axes)):
        axes[j].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def parameter_spread_figure(rows):
    """
    Build a bar chart of the mean held-out scatter for every (successful) grid
    point, so the spread can be compared across the swept parameters. Returns a
    matplotlib ``Figure`` (or ``None``).
    """
    if plt is None:
        return None
    ok = [r for r in rows if r.get("status") == "ok" and "mean_scatter" in r]
    if not ok:
        return None

    names = ["{0}|o{1}|reg{2:g}|{3}".format(
        r["label_set"], r["order"], r["regularization"], r["censor"])
        for r in ok]
    values = [r["mean_scatter"] for r in ok]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(ok)), 4.0))
    ax.bar(range(len(ok)), values, color="#4C72B0")
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("mean held-out scatter")
    ax.set_title("Label spread across swept parameters")
    fig.tight_layout()
    return fig


def _log_wandb_summary(project, entity, group, mode, run_dir, rows):
    """
    Log a sweep-level summary run: a table of every grid point's metrics and a
    bar chart comparing the spread across parameters.
    """
    run = _init_wandb_run(project, entity, group, mode, run_dir,
                          name="summary", config=dict(n_combos=len(rows)))
    if run is None:
        return

    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    data = [[row.get(c, None) for c in columns] for row in rows]
    try:
        run.log({"results": wandb.Table(columns=columns, data=data)})
    except Exception as exc:                            # pragma: no cover
        logger.warning("could not log summary table: %s", exc)

    fig = parameter_spread_figure(rows)
    if fig is not None:
        run.log({"spread/by_parameters": wandb.Image(fig)})
        plt.close(fig)

    run.finish()


# --------------------------------------------------------------------------- #
#  The grid sweep                                                              #
# --------------------------------------------------------------------------- #

def sweep(labels, flux, ivar, dispersion, label_sets, orders, regularizations,
    censor_factories=None, n_splits=5, seed=0, output=None, train_kwds=None,
    test_kwds=None, age_reliability=None, wandb_project=None, wandb_entity=None,
    wandb_group=None, wandb_mode="offline", wandb_dir=None):
    """
    Cross-validate The Cannon over a grid of label sets, polynomial orders,
    regularization strengths and (optionally) censoring schemes.

    :param labels:
        A mapping of ``{label_name: (N,) array}`` (dict, numpy structured array,
        or astropy Table) covering every name used in ``label_sets``.

    :param flux, ivar:
        ``(N, P)`` arrays of normalized flux and inverse variance.

    :param dispersion:
        The ``(P,)`` dispersion array, or ``None``.

    :param label_sets:
        An iterable of label-name tuples to try as the model's label set.

    :param orders:
        An iterable of polynomial orders for the vectorizer.

    :param regularizations:
        An iterable of L1 regularization strengths.

    :param censor_factories: [optional]
        A mapping ``{name: factory}`` where ``factory(label_names, dispersion,
        n_pixels)`` returns a :class:`~thecannon.censoring.Censors` object (or
        ``None`` for no censoring). Defaults to a single ``{"none": None}``
        scheme. A factory is used (rather than a fixed object) because the
        censor mask depends on the label set being tried.

    :param n_splits:
        The number of cross-validation folds.

    :param seed:
        RNG seed for the (shared) fold assignment.

    :param output: [optional]
        If given, write the tidy results to this CSV path.

    :param wandb_project: [optional]
        If given (and :mod:`wandb` is installed), log one W&B run per grid point
        under this project. Each run records the hyper-parameters as its config
        and the cross-validation metrics (plus per-label residual histograms).

    :param wandb_entity, wandb_group: [optional]
        The W&B entity (team/user) and the group used to tie the sweep's runs
        together in the UI.

    :param wandb_mode: [optional]
        The W&B run mode. Defaults to ``"offline"`` -- runs are written locally
        (no network/login needed) and can later be uploaded with ``wandb sync``.

    :param wandb_dir: [optional]
        The directory under which offline run data is written (defaults to
        ``./wandb``).

    :returns:
        A list of result-row dicts (one per grid point). If :mod:`pandas` is
        installed, a ``DataFrame`` is returned instead.
    """

    if wandb_project and wandb is None:
        logger.warning("wandb_project given but wandb is not installed; "
                       "install it with `pip install wandb` to enable logging")

    if censor_factories is None:
        censor_factories = {"none": None}

    n_pixels = np.atleast_2d(flux).shape[1]
    grid = list(itertools.product(
        [tuple(s) for s in label_sets], list(orders), list(regularizations),
        list(censor_factories.items())))

    logger.info("Sweeping %d combinations (%d folds each)", len(grid), n_splits)

    rows = []
    for n, (label_set, order, reg, (censor_name, factory)) in enumerate(grid, 1):

        config = dict(label_set="+".join(label_set), n_labels=len(label_set),
                      order=order, regularization=reg, censor=censor_name,
                      n_splits=n_splits, seed=seed)
        row = dict(label_set=config["label_set"], n_labels=len(label_set),
                   order=order, regularization=reg, censor=censor_name)

        run = None
        if wandb_project:
            run = _init_wandb_run(
                wandb_project, wandb_entity, wandb_group, wandb_mode, wandb_dir,
                name="{0}|o{1}|reg{2:g}|{3}".format(
                    config["label_set"], order, reg, censor_name),
                config=config)

        cv = None
        try:
            label_array = _label_matrix(labels, label_set)
            censors = factory(label_set, dispersion, n_pixels) \
                if callable(factory) else factory

            cv = cross_validate(
                label_array, np.atleast_2d(flux), np.atleast_2d(ivar),
                dispersion, label_set, order, reg, censors=censors,
                n_splits=n_splits, seed=seed, train_kwds=train_kwds,
                test_kwds=test_kwds, age_reliability=age_reliability)

            row.update(_summarize(cv, label_set))
            row["status"] = "ok"
            logger.info(
                "[%d/%d] %s order=%d reg=%g censor=%s -> "
                "mean_r2=%.4f mean_sigma_mad=%.4f pull_std=%.2f r_chi_sq=%.2f",
                n, len(grid), row["label_set"], order, reg, censor_name,
                row["mean_r2"], row["mean_sigma_mad"],
                row.get("mean_pull_std", float("nan")),
                row["median_r_chi_sq"])

            # Per-run figure: the spread (recovered vs reference) for all labels.
            # Use cv["reference"], not label_array, since the age cut may have
            # reduced the rows.
            if run is not None:
                fig = spread_figure(
                    cv["recovered"], cv["reference"], label_set,
                    title="{0} | order={1} | reg={2:g} | {3}".format(
                        config["label_set"], order, reg, censor_name))
                if fig is not None:
                    run.log({"spread/one_to_one": wandb.Image(fig)})
                    plt.close(fig)

        except Exception as exc:                       # keep the sweep alive
            row["status"] = "error: {0}".format(exc)
            logger.warning("[%d/%d] %s order=%d reg=%g censor=%s FAILED: %s",
                           n, len(grid), row["label_set"], order, reg,
                           censor_name, exc)
        finally:
            if run is not None:
                _log_wandb_run(run, row, cv, label_set)
                run.finish()
        rows.append(row)

    # Sweep-level summary run: a results table plus the spread-vs-parameters
    # comparison, so the whole sweep can be browsed from one W&B run.
    if wandb_project and wandb is not None:
        _log_wandb_summary(wandb_project, wandb_entity, wandb_group, wandb_mode,
                           wandb_dir, rows)

    if output:
        _write_csv(rows, output)
        logger.info("Wrote %d rows to %s", len(rows), output)

    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


def _write_csv(rows, path):
    """ Write the result rows to CSV with a stable, union-of-keys header. """
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
#  Demo / smoke test                                                           #
# --------------------------------------------------------------------------- #

def _demo(wandb_project=None, wandb_mode="offline"):
    """ Run a tiny sweep on the bundled golden data to verify the pipeline. """
    try:
        from scripts.sweep_config import load_golden
    except ImportError:
        from sweep_config import load_golden

    try:
        names, labels, dispersion, flux, ivar = load_golden()
        print("Loaded golden data: {0} stars, {1} pixels, labels {2}".format(
            flux.shape[0], flux.shape[1], names))
    except (OSError, IOError, KeyError):
        print("Golden data not found; synthesizing a toy data set.")
        rng = np.random.default_rng(0)
        names = ["A", "B", "C"]
        N, P = 80, 120
        label_array = rng.normal(size=(N, 3))
        labels = {name: label_array[:, i] for i, name in enumerate(names)}
        dispersion = np.linspace(0, 1, P)
        coeff = rng.normal(size=(P, 4))
        clean = (coeff[:, 0] + label_array @ coeff[:, 1:].T).T
        flux = 1.0 + 0.1 * clean + rng.normal(0, 0.01, size=(N, P))
        ivar = np.full((N, P), 1.0 / 0.01 ** 2)

    results = sweep(
        labels, flux, ivar, dispersion,
        label_sets=[tuple(names[:2]), tuple(names)],
        orders=[1, 2],
        regularizations=[0.0, 1e3],
        n_splits=3,
        seed=0,
        output="sweep_demo_results.csv",
        wandb_project=wandb_project,
        wandb_group="cannon-sweep-demo",
        wandb_mode=wandb_mode)

    print("\n=== sweep results ===")
    try:
        import pandas as pd  # noqa: F401
        cols = ["label_set", "order", "regularization", "censor",
                "mean_r2", "mean_sigma_mad", "mean_pull_std",
                "median_r_chi_sq", "theta_frac_zero", "frac_in_hull", "status"]
        cols = [c for c in cols if c in results.columns]
        print(results[cols].to_string(index=False))
    except ImportError:
        for row in results:
            print({k: row[k] for k in (
                "label_set", "order", "regularization", "mean_scatter",
                "median_r_chi_sq", "status")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true",
                        help="run a small sweep on the bundled golden data")
    parser.add_argument("--wandb-project", default=None,
                        help="log each grid point to this W&B project")
    parser.add_argument("--wandb-mode", default="offline",
                        choices=["offline", "online", "disabled"],
                        help="W&B run mode (default: offline)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable INFO-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.demo:
        _demo(wandb_project=args.wandb_project, wandb_mode=args.wandb_mode)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
