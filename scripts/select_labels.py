#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Search over combinations of label columns for The Cannon.

Testing "which columns to include" is a subset-selection problem: a fixed core
of labels plus some pool of candidate columns gives up to 2**K subsets, so
brute force is rarely the right move. This driver offers three strategies, all
scored by the same k-fold cross-validation as the sweep
(:func:`scripts.sweep_cannon.cross_validate` / ``_summarize``) with a shared
fold assignment so sets are compared on the *same* stars:

  - ``forward`` (default) -- greedy forward selection. Start from the core, and
    at each step add the single candidate that most improves the objective, until
    no candidate helps by at least ``--min-gain`` (or ``--max-size`` is hit).
    ~K**2 evaluations instead of 2**K; the right choice for many candidates.

  - ``screen`` -- ablation / one-at-a-time. Evaluate the core alone and core +
    each single candidate, reporting each column's *marginal* effect. Cheap
    (linear in K); tells you which columns carry spectral information and which
    ones hurt the core labels when added.

  - ``subsets`` -- exhaustive: evaluate the core + every subset of the candidate
    pool up to ``--max-extra`` added columns. Only feasible for a small pool
    (guarded by ``--max-evals``).

Choosing the objective
----------------------
Rank on a *unit-free* metric so the search is not dominated by the label with
the largest units. The default is ``mean_r2`` (mean per-label explained
variance). If you care about one target label -- e.g. age -- pass
``--target age_L``: the objective becomes that label's own ``r2_age_L`` and the
target is pinned into the core so it is always fit. (Any ``_summarize`` column
works via ``--metric`` / ``--goal``, e.g. ``--metric sigma_mad_age_L --goal
minimize``.)

Note: adding a label the spectra do not constrain can degrade *all* labels (the
model fits a free direction to noise), so ``screen`` is a good first pass and
``forward`` naturally stops before piling on uninformative columns.

Order and regularization are held fixed during selection (``--order`` /
``--regularization``) to keep it cheap; tune those afterwards on the chosen set
with :mod:`scripts.run_sweep` or :mod:`scripts.wandb_sweep`.

JAX device selection is via the ``JAX_PLATFORMS`` environment variable.

Usage
-----
Greedy forward selection toward age, from a pool of abundances::

    python -m scripts.select_labels \\
        --spectra /path/merged_with_ages_raw.parquet \\
        --continuum-list /path/continuum.list \\
        --core raw_teff,raw_logg,raw_fe_h \\
        --candidates mg_fe,ce_fe,ca_fe,si_fe,ni_fe,mn_fe,al_fe,c_fe,n_fe \\
        --target age_L \\
        --order 2 --n-splits 5 --output label_selection.csv

Ablation screen of each abundance over the core::

    python -m scripts.select_labels --mode screen \\
        --core raw_teff,raw_logg,raw_fe_h --candidates mg_fe,ce_fe,ca_fe

Quick offline check on the bundled golden data::

    JAX_PLATFORMS=cpu python -m scripts.select_labels --demo
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import itertools
import logging
import os

import numpy as np

# Work both as a package module and when run directly from scripts/.
try:
    from scripts.train_cannon import (load_spectra, normalize_spectra,
                                       quality_mask, DEFAULT_SPECTRA,
                                       DEFAULT_CONTINUUM_LIST)
    from scripts.run_sweep import finite_label_mapping
    from scripts.sweep_cannon import _label_matrix, cross_validate, _summarize
    from scripts.sweep_config import (DEFAULT_ABUNDANCES, add_filter_arg,
                                       apply_filters, per_label_masks,
                                       load_golden)
except ImportError:
    from train_cannon import (load_spectra, normalize_spectra, quality_mask,
                              DEFAULT_SPECTRA, DEFAULT_CONTINUUM_LIST)
    from run_sweep import finite_label_mapping
    from sweep_cannon import _label_matrix, cross_validate, _summarize
    from sweep_config import (DEFAULT_ABUNDANCES, add_filter_arg,
                             apply_filters, per_label_masks, load_golden)

logger = logging.getLogger("thecannon.select_labels")


# --------------------------------------------------------------------------- #
#  Objective                                                                   #
# --------------------------------------------------------------------------- #

def resolve_metric(args):
    """ Return ``(metric_key, goal)`` from --metric/--goal/--target. """
    if args.metric:
        return args.metric, args.goal
    if args.target:
        return "r2_{0}".format(args.target), "maximize"
    return "mean_r2", "maximize"


def _better(a, b, goal):
    """ True if score ``a`` is better than ``b`` under ``goal``. """
    if b is None:
        return True
    return a > b if goal == "maximize" else a < b


# --------------------------------------------------------------------------- #
#  Evaluation (shared fold assignment; fixed order / regularization)           #
# --------------------------------------------------------------------------- #

def make_evaluator(mapping, flux, ivar, dispersion, order, regularization,
    n_splits, seed, age_masks=None):
    """
    Return ``evaluate(label_set) -> row`` where ``row`` is the ``_summarize``
    metric dict for that label set (cross-validated at the fixed order /
    regularization, with the per-set age-reliability cut). Results are memoized
    so ``forward``/``subsets`` never re-evaluate the same set.
    """
    cache = {}

    def evaluate(label_set):
        key = tuple(label_set)
        if key not in cache:
            label_array = _label_matrix(mapping, key)
            cv = cross_validate(
                label_array, flux, ivar, dispersion, key, order=order,
                regularization=regularization, n_splits=n_splits, seed=seed,
                age_reliability=age_masks)
            cache[key] = _summarize(cv, key)
        return cache[key]

    return evaluate


# --------------------------------------------------------------------------- #
#  Strategies                                                                  #
# --------------------------------------------------------------------------- #

def forward_select(evaluate, core, candidates, metric, goal, min_gain,
    max_size):
    """
    Greedy forward selection. Returns ``(rows, chosen)`` where ``rows`` is the
    tidy per-evaluation log (with ``step``/``added``/``gain``) and ``chosen`` is
    the final label set.
    """
    current = list(core)
    remaining = [c for c in candidates if c not in current]
    rows = []

    base_row = evaluate(current)
    best = base_row[metric]
    rows.append(_row(base_row, metric, step=0, added="(core)", gain=0.0,
                     selected=True))
    logger.info("step 0: core %s -> %s=%.5g", "+".join(current), metric, best)

    step = 0
    while remaining and (max_size is None or len(current) < max_size):
        step += 1
        best_cand, best_cand_row, best_cand_score = None, None, None
        for cand in remaining:
            trial = current + [cand]
            row = evaluate(trial)
            score = row[metric]
            rows.append(_row(row, metric, step=step, added=cand,
                             gain=_gain(score, best, goal), selected=False))
            if _better(score, best_cand_score, goal):
                best_cand, best_cand_row, best_cand_score = cand, row, score

        gain = _gain(best_cand_score, best, goal)
        logger.info("step %d: best add '%s' -> %s=%.5g (gain=%+.5g)",
                    step, best_cand, metric, best_cand_score, gain)
        if gain < min_gain:
            logger.info("gain %+.5g < --min-gain %g; stopping", gain, min_gain)
            break

        # Mark the accepted candidate's row as selected.
        for r in rows:
            if r["step"] == step and r["added"] == best_cand:
                r["selected"] = True
        current.append(best_cand)
        remaining.remove(best_cand)
        best = best_cand_score

    return rows, tuple(current)


def screen(evaluate, core, candidates, metric, goal):
    """
    Ablation: core alone, then core + each single candidate. Returns the tidy
    rows (each candidate row's ``gain`` is its marginal effect over the core).
    """
    rows = []
    base_row = evaluate(list(core))
    base = base_row[metric]
    rows.append(_row(base_row, metric, step=0, added="(core)", gain=0.0,
                     selected=True))
    logger.info("core %s -> %s=%.5g", "+".join(core), metric, base)

    for cand in candidates:
        if cand in core:
            continue
        row = evaluate(list(core) + [cand])
        gain = _gain(row[metric], base, goal)
        rows.append(_row(row, metric, step=1, added=cand, gain=gain,
                         selected=gain > 0))
        logger.info("  + %-8s -> %s=%.5g (marginal %+.5g)", cand, metric,
                    row[metric], gain)
    return rows


def subsets(evaluate, core, candidates, metric, goal, max_extra, max_evals):
    """
    Exhaustive: core + every subset of ``candidates`` with up to ``max_extra``
    added columns. Guarded by ``max_evals``.
    """
    pool = [c for c in candidates if c not in core]
    combos = []
    upper = len(pool) if max_extra is None else min(max_extra, len(pool))
    for k in range(0, upper + 1):
        combos.extend(itertools.combinations(pool, k))
    if len(combos) > max_evals:
        raise ValueError(
            "{0} subsets exceed --max-evals={1}; lower --max-extra, shrink the "
            "candidate pool, or use --mode forward".format(len(combos),
                                                           max_evals))
    logger.info("evaluating %d subsets (core + up to %d extras)",
                len(combos), upper)

    rows = []
    for combo in combos:
        label_set = list(core) + list(combo)
        row = evaluate(label_set)
        rows.append(_row(row, metric, step=len(combo),
                         added="+".join(combo) or "(core)", gain=float("nan"),
                         selected=False))
    return rows


def _gain(score, baseline, goal):
    """ Improvement of ``score`` over ``baseline`` in the direction of ``goal``. """
    if score is None or baseline is None:
        return float("nan")
    return (score - baseline) if goal == "maximize" else (baseline - score)


def _row(summary_row, metric, step, added, gain, selected):
    """ Flatten a _summarize row into a tidy record with selection bookkeeping. """
    row = {"step": step, "added": added,
           "label_set": "+".join(summary_row_labels(summary_row)),
           "n_labels": _n_labels(summary_row),
           "objective_metric": metric, "objective": summary_row.get(metric),
           "gain": gain, "selected": selected}
    row.update(summary_row)
    return row


def summary_row_labels(summary_row):
    """ Recover the label names present in a _summarize row (from bias_* keys). """
    return [k[len("bias_"):] for k in summary_row if k.startswith("bias_")]


def _n_labels(summary_row):
    return sum(1 for k in summary_row if k.startswith("bias_"))


# --------------------------------------------------------------------------- #
#  Data                                                                        #
# --------------------------------------------------------------------------- #

def load(args):
    """ Return ``(mapping, flux, ivar, dispersion, n_stars)`` on the shared,
    finite-across-all-referenced-columns star set. """
    core = list(args.core)
    if args.target and args.target not in core:
        core.append(args.target)                       # pin the target in
    if len(core) < 2:
        raise ValueError("the core must have at least 2 labels (The Cannon "
                         "needs >= 2 to fit); got {0}".format(core))
    candidates = [c for c in args.candidates if c not in core]
    union = list(dict.fromkeys(core + candidates))

    if args.demo:
        _, label_source, dispersion, flux, ivar = load_golden()
        print("Demo on golden data: {0} stars, {1} pixels".format(
            flux.shape[0], flux.shape[1]))
    else:
        label_source, dispersion, flux, ivar = load_spectra(args.spectra)
        good = quality_mask(label_source)
        if not good.any():
            raise ValueError("quality cuts rejected every star")
        label_source = label_source[good]
        flux, ivar = flux[good], ivar[good]
        label_source, flux, ivar = apply_filters(
            label_source, flux, ivar, args.filters, log=logger)
        flux, ivar = normalize_spectra(dispersion, flux, ivar,
                                       args.continuum_list)

    def _has(name):
        cols = getattr(label_source, "columns", None)
        return (name in cols) if cols is not None else (name in label_source)

    # Core columns must exist; missing candidates are dropped (with a warning) so
    # a big speculative pool can be screened against whatever the parquet has.
    missing_core = [n for n in core if not _has(n)]
    if missing_core:
        raise ValueError("core columns not in the data: {0}".format(missing_core))
    dropped = [c for c in candidates if not _has(c)]
    if dropped:
        logger.warning("dropping %d candidate(s) not in the data: %s",
                       len(dropped), ", ".join(dropped))
        candidates = [c for c in candidates if _has(c)]
    if not candidates:
        raise ValueError("no candidate columns are present in the data")
    union = list(dict.fromkeys(core + candidates))

    # One finite mask over EVERY referenced column, so all label sets are scored
    # on identical stars (a fair comparison, at the cost of some dropped stars).
    mapping, finite = finite_label_mapping(label_source, union)
    age_masks = {age: mask[finite]
                 for age, mask in per_label_masks(label_source).items()}
    return (mapping, flux[finite], ivar[finite], dispersion, int(finite.sum()),
            tuple(core), tuple(candidates), age_masks)


# --------------------------------------------------------------------------- #
#  Reporting                                                                   #
# --------------------------------------------------------------------------- #

def log_wandb(args, rows, chosen, metric, goal):
    """
    Log the whole selection as a single (offline by default) W&B run: the
    per-evaluation scalar metrics, the tidy table, and the chosen set in the
    run summary. No-op unless ``--wandb-project`` is given.
    """
    if not args.wandb_project:
        return
    try:
        from scripts.sweep_cannon import wandb, _init_wandb_run
    except ImportError:
        from sweep_cannon import wandb, _init_wandb_run
    if wandb is None:
        logger.warning("wandb is not installed; skipping W&B logging")
        return

    name = args.wandb_name
    if not name:
        name = "select-{0}".format(args.mode)
        if args.target:
            name += "-{0}".format(args.target)
    config = dict(
        mode=args.mode, core=list(args.core),
        candidates=list(args.candidates), target=args.target,
        objective_metric=metric, goal=goal, order=args.order,
        regularization=args.regularization, n_splits=args.n_splits,
        seed=args.seed, filters=args.filters, min_gain=args.min_gain,
        max_size=args.max_size, spectra=args.spectra)
    run = _init_wandb_run(args.wandb_project, None, None, args.wandb_mode,
                          None, name=name, config=config)
    if run is None:
        return

    for row in rows:
        run.log({k: v for k, v in row.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)})

    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    data = [[row.get(c) for c in columns] for row in rows]
    try:
        run.log({"selection": wandb.Table(columns=columns, data=data)})
    except Exception as exc:                            # pragma: no cover
        logger.warning("could not log the selection table: %s", exc)

    run.summary["objective_metric"] = metric
    run.summary["goal"] = goal
    if chosen is not None:
        run.summary["chosen_label_set"] = "+".join(chosen)
    run.finish()


def report(rows, chosen, metric, goal, output):
    import pandas as pd
    df = pd.DataFrame(rows)
    if output:
        df.to_csv(output, index=False)
        logger.info("wrote %d evaluations to %s", len(df), output)

    print("\n=== label selection ({0} evaluations, objective={1} [{2}]) ==="
          .format(len(df), metric, goal))
    ranked = df.sort_values("objective", ascending=(goal == "minimize"))
    cols = ["label_set", "n_labels", "objective", "gain", "selected"]
    extra = [c for c in ("mean_r2", "mean_sigma_mad", "mean_pull_std",
                         "median_r_chi_sq") if c in df.columns and c != metric]
    top = ranked.head(12)[cols + extra]
    print(top.to_string(index=False))

    if chosen is not None:
        print("\nchosen label set: {0}".format("+".join(chosen)))
    return df


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spectra", default=DEFAULT_SPECTRA)
    parser.add_argument("--continuum-list", default=DEFAULT_CONTINUUM_LIST)
    parser.add_argument("--mode", default="forward",
                        choices=["forward", "screen", "subsets"])
    parser.add_argument("--core", type=lambda s: s.split(","),
                        default=["raw_teff", "raw_logg", "raw_fe_h"],
                        help="labels always included")
    parser.add_argument("--candidates", type=lambda s: s.split(","),
                        default=["mg_fe"] + list(DEFAULT_ABUNDANCES),
                        help="candidate columns to select from")
    add_filter_arg(parser)
    parser.add_argument("--target", default=None,
                        help="optimize this label's r2 (pinned into the core)")
    parser.add_argument("--metric", default=None,
                        help="explicit objective column (e.g. mean_r2, "
                             "r2_age_L, sigma_mad_age_L); overrides --target")
    parser.add_argument("--goal", default="maximize",
                        choices=["maximize", "minimize"])
    parser.add_argument("--order", type=int, default=2,
                        help="polynomial order held fixed during selection")
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-gain", type=float, default=1e-4,
                        help="forward: stop when the best add improves the "
                             "objective by less than this")
    parser.add_argument("--max-size", type=int, default=None,
                        help="forward: cap on total number of labels")
    parser.add_argument("--max-extra", type=int, default=None,
                        help="subsets: max added columns per subset")
    parser.add_argument("--max-evals", type=int, default=512,
                        help="subsets: refuse to evaluate more than this many")
    parser.add_argument("--output", default="label_selection.csv")
    parser.add_argument("--wandb-project", default=None,
                        help="log the selection to this W&B project")
    parser.add_argument("--wandb-mode", default="offline",
                        choices=["offline", "online", "disabled"],
                        help="W&B run mode (default: offline)")
    parser.add_argument("--wandb-name", default=None,
                        help="W&B run name (default: select-<mode>[-<target>])")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    metric, goal = resolve_metric(args)

    (mapping, flux, ivar, dispersion, n_stars, core, candidates,
     age_masks) = load(args)
    logger.info("selecting over core=%s candidates=%s on %d stars "
                "(order=%d, reg=%g, %d folds)", "+".join(core),
                ",".join(candidates), n_stars, args.order, args.regularization,
                args.n_splits)

    evaluate = make_evaluator(
        mapping, flux, ivar, dispersion, args.order, args.regularization,
        args.n_splits, args.seed, age_masks=age_masks)

    chosen = None
    if args.mode == "forward":
        rows, chosen = forward_select(
            evaluate, core, candidates, metric, goal, args.min_gain,
            args.max_size)
    elif args.mode == "screen":
        rows = screen(evaluate, core, candidates, metric, goal)
    else:
        rows = subsets(evaluate, core, candidates, metric, goal,
                       args.max_extra, args.max_evals)

    report(rows, chosen, metric, goal, args.output)
    log_wandb(args, rows, chosen, metric, goal)


if __name__ == "__main__":
    main()
