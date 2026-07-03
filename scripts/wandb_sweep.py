#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bayesian (or random / grid) hyper-parameter search for The Cannon via W&B Sweeps.

Unlike :mod:`scripts.run_sweep` -- which exhaustively cross-validates a *fixed*
grid and uses W&B only for logging -- this driver hands the search itself to
W&B's sweep controller: the optimizer proposes a point, an agent evaluates it by
cross-validating one Cannon model (reusing
:func:`scripts.sweep_cannon.cross_validate` / ``_summarize``) and logs the
objective back, and the optimizer proposes the next point.

Three roles (``--role``)
------------------------
``agent`` (default)
    The all-in-one mode: one process talks to the W&B controller *and* does the
    computation. Needs outbound internet on the machine that runs it. Good for a
    laptop / an interactive node / any single machine with network access.

``controller`` + ``worker`` (the split for NCI Gadi and similar clusters)
    Bayesian search is a closed loop -- the optimizer must see a trial's metric
    before it proposes the next point -- but on Gadi only the ``copyq`` queue has
    outbound internet, while the heavy compute must run on ``normal`` /
    ``gpuvolta`` (no internet). Since there is no network path between the
    queues, they talk through the one thing both see: the shared filesystem.

        controller (copyq, online, tiny)      worker (normal/gpuvolta, offline, heavy)
        --------------------------------      ----------------------------------------
        wandb.agent(sweep_id):                loop:
          params <- wandb.config                task  <- claim  $BROKER/tasks/<id>.json
          write  $BROKER/tasks/<id>.json  -->  cv = cross_validate(...) ; _summarize
          wait   $BROKER/results/<id>.json <-  write $BROKER/results/<id>.json
          wandb.log(metrics)                   (never touches the network)

    The Bayesian optimization stays 100% native W&B; only the *transport* from
    optimizer to compute is filesystem-based. The controller is network- and
    CPU-light (it just brokers files) and does not import jax/thecannon; the
    worker needs the science stack but no W&B / internet.

Search space
------------
  - ``regularization`` -- the continuous knob Bayesian search is good at; swept
    log-uniformly over ``--reg-min .. --reg-max`` (or a discrete ``--reg-values``
    list, which also lets you include 0 = no regularization).
  - ``order`` -- polynomial order, a small discrete set (``--orders``).
  - ``label_set`` -- an (unordered) categorical over the candidate label sets.

The objective defaults to **maximizing ``mean_r2``** (the unit-free CV quality
score); switch it with ``--metric`` / ``--goal`` (e.g. minimize
``mean_sigma_mad``). Rank on ``mean_r2`` rather than ``mean_scatter`` so the
search is not dominated by the label with the largest units (e.g. TEFF).

JAX device selection is via the ``JAX_PLATFORMS`` environment variable.

Usage
-----
Single machine with internet (40-trial Bayesian sweep)::

    wandb login
    python -m scripts.wandb_sweep --role agent \\
        --spectra /path/cleaned_ages.parquet \\
        --continuum-list /path/continuum.list \\
        --label-set raw_teff,raw_logg,raw_fe_h \\
        --label-set raw_teff,raw_logg,raw_fe_h,mg_fe,ce_fe \\
        --orders 1,2 --reg-min 1e0 --reg-max 1e5 \\
        --wandb-project cannon-bayes --count 40

Gadi split (see scripts/sweep_bayes_worker.pbs + scripts/sweep_bayes_controller.pbs).
On the compute queue::

    python -m scripts.wandb_sweep --role worker --broker $BROKER \\
        --spectra ... --continuum-list ... --n-splits 5

On copyq (after `wandb login`)::

    python -m scripts.wandb_sweep --role controller --broker $BROKER \\
        --label-set raw_teff,raw_logg,raw_fe_h \\
        --orders 1,2 --reg-min 1e0 --reg-max 1e5 \\
        --wandb-project cannon-bayes --count 40

Offline check of the wiring on the bundled golden data (no login/network)::

    JAX_PLATFORMS=cpu python -m scripts.wandb_sweep --role agent --demo --dry-run
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import glob
import json
import logging
import os
import time

import numpy as np

# Lightweight shared config (stdlib + numpy only, no science stack) -- safe to
# import at module top even for the network-light controller role.
try:
    from scripts.sweep_config import (add_label_builder_args, add_filter_arg,
                                       apply_filters, age_reliability_masks,
                                       load_golden)
except ImportError:
    from sweep_config import (add_label_builder_args, add_filter_arg,
                             apply_filters, age_reliability_masks, load_golden)

logger = logging.getLogger("thecannon.wandb_sweep")

# Task/result files are keyed by the trial id; tmp siblings are hidden dotfiles
# so an in-progress write never matches the "*.json" scan.
_TASKS, _CLAIMED, _RESULTS = "tasks", "claimed", "results"
_STOP_FILE = "STOP"
_SWEEP_ID_FILE = "sweep_id.txt"


# --------------------------------------------------------------------------- #
#  Sweep config                                                               #
# --------------------------------------------------------------------------- #

def build_sweep_config(label_sets, orders, method, metric, goal,
    reg_values=None, reg_min=None, reg_max=None):
    """
    Build the W&B sweep configuration dict.

    ``label_set`` is encoded as ``"+"``-joined strings (a categorical), ``order``
    as a discrete set, and ``regularization`` either as a discrete ``values``
    list (``reg_values``, which may include 0) or -- the natural choice for
    Bayesian search -- log-uniformly over ``[reg_min, reg_max]``.
    """
    if reg_values is not None:
        reg_param = {"values": [float(x) for x in reg_values]}
    else:
        if not (reg_min and reg_max) or reg_min <= 0 or reg_max <= 0:
            raise ValueError("log-uniform regularization needs positive "
                             "--reg-min/--reg-max; pass --reg-values to include 0")
        reg_param = {"distribution": "log_uniform_values",
                     "min": float(reg_min), "max": float(reg_max)}

    return {
        "method": method,
        "metric": {"name": metric, "goal": goal},
        "parameters": {
            "label_set": {"values": ["+".join(s) for s in label_sets]},
            "order": {"values": [int(o) for o in orders]},
            "regularization": reg_param,
        },
    }


# --------------------------------------------------------------------------- #
#  Filesystem broker (controller <-> worker transport)                         #
# --------------------------------------------------------------------------- #

def _broker_paths(broker):
    """ Ensure the broker sub-directories exist and return their paths. """
    paths = {name: os.path.join(broker, name)
             for name in (_TASKS, _CLAIMED, _RESULTS)}
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def _atomic_write_json(path, obj):
    """
    Write ``obj`` as JSON so a reader never sees a partial file: write to a
    hidden tmp sibling in the same directory, then ``os.replace`` (atomic within
    one filesystem -- true for a Lustre scratch shared by both queues).
    """
    directory, base = os.path.split(path)
    tmp = os.path.join(directory, ".{0}.tmp".format(base))
    with open(tmp, "w") as fp:
        json.dump(obj, fp)
    os.replace(tmp, path)


def _read_json(path):
    with open(path) as fp:
        return json.load(fp)


def _claim_one(tasks_dir, claimed_dir):
    """
    Atomically claim one pending task by renaming it out of ``tasks/`` into
    ``claimed/`` (rename is atomic and single-winner, so many workers can poll
    the same directory safely). Returns ``(trial_id, task_dict)`` or ``None``.
    """
    for src in sorted(glob.glob(os.path.join(tasks_dir, "*.json"))):
        trial_id = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(claimed_dir, trial_id + ".json")
        try:
            os.rename(src, dst)                 # atomic claim; loser gets OSError
        except OSError:
            continue
        return trial_id, _read_json(dst)
    return None


# --------------------------------------------------------------------------- #
#  Objectives                                                                  #
# --------------------------------------------------------------------------- #

def make_inline_objective(mapping, flux, ivar, dispersion, n_splits, seed,
    metric, wandb_project=None, age_masks=None):
    """
    Objective for ``--role agent``: read the point from ``wandb.config``,
    cross-validate one model inline, and log the full metric row.
    """
    import wandb
    from importlib import import_module
    sc = import_module("scripts.sweep_cannon" if __package__ else "sweep_cannon")

    def objective():
        run = wandb.init(project=wandb_project)
        try:
            cfg = wandb.config
            label_set = tuple(cfg.label_set.split("+"))
            label_array = sc._label_matrix(mapping, label_set)
            cv = sc.cross_validate(
                label_array, flux, ivar, dispersion, label_set,
                order=int(cfg.order), regularization=float(cfg.regularization),
                n_splits=n_splits, seed=seed, age_reliability=age_masks)
            row = sc._summarize(cv, label_set)
            wandb.log(_scalars(row))
        finally:
            run.finish()

    return objective


def make_broker_objective(broker, metric, poll_interval, trial_timeout,
    wandb_project=None):
    """
    Objective for ``--role controller``: hand the point to a worker via the
    broker and wait for its result -- no computation here. The controller stays
    network- and CPU-light and needs no science stack.
    """
    import wandb
    tasks_dir = os.path.join(broker, _TASKS)
    results_dir = os.path.join(broker, _RESULTS)

    def objective():
        run = wandb.init(project=wandb_project)
        try:
            cfg = wandb.config
            trial_id = run.id
            task = {"trial_id": trial_id, "label_set": cfg.label_set,
                    "order": int(cfg.order),
                    "regularization": float(cfg.regularization)}
            _atomic_write_json(
                os.path.join(tasks_dir, trial_id + ".json"), task)
            logger.info("dispatched trial %s: %s", trial_id, task)

            result_path = os.path.join(results_dir, trial_id + ".json")
            waited = 0.0
            while not os.path.exists(result_path):
                if trial_timeout and waited >= trial_timeout:
                    raise TimeoutError(
                        "no worker result for trial {0} after {1:.0f}s -- is a "
                        "worker running against this broker?".format(
                            trial_id, trial_timeout))
                time.sleep(poll_interval)
                waited += poll_interval

            result = _read_json(result_path)
            if result.get("status") != "ok":
                # Raise so the agent marks the run failed; the optimizer then
                # avoids this region rather than being fed a sentinel metric.
                raise RuntimeError("worker failed trial {0}: {1}".format(
                    trial_id, result.get("error")))
            wandb.log(_scalars(result["metrics"]))
        finally:
            run.finish()

    return objective


def _scalars(row):
    """ The numeric, non-boolean entries of a metric row (what W&B optimizes). """
    return {k: v for k, v in row.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


# --------------------------------------------------------------------------- #
#  Worker loop                                                                 #
# --------------------------------------------------------------------------- #

def run_worker(args):
    """
    Poll the broker, cross-validate each claimed task, and write its result.
    Runs offline with no W&B/network. Exits on a ``STOP`` file, after
    ``--worker-idle-timeout`` seconds without work, or when the job walltime
    ends. Multiple workers can share one broker.
    """
    if __package__:
        from scripts.sweep_cannon import (_label_matrix, cross_validate,
                                          _summarize)
    else:
        from sweep_cannon import _label_matrix, cross_validate, _summarize

    (mapping, flux, ivar, dispersion, label_sets, n_splits,
     age_masks) = _load_data(args)
    paths = _broker_paths(args.broker)

    # Publish the data-filtered label sets so a controller can (optionally) read
    # a consistent, valid list rather than guessing column names.
    _atomic_write_json(
        os.path.join(args.broker, "label_sets.json"),
        {"label_sets": ["+".join(s) for s in label_sets],
         "n_splits": n_splits, "seed": args.seed})

    stop_file = os.path.join(args.broker, _STOP_FILE)
    logger.info("worker up on %s; polling %s (n_splits=%d)",
                os.uname()[1] if hasattr(os, "uname") else "?",
                paths[_TASKS], n_splits)

    idle = 0.0
    n_done = 0
    while True:
        if os.path.exists(stop_file):
            logger.info("STOP file present; worker exiting after %d trials",
                        n_done)
            break

        claimed = _claim_one(paths[_TASKS], paths[_CLAIMED])
        if claimed is None:
            time.sleep(args.poll_interval)
            idle += args.poll_interval
            if args.worker_idle_timeout and idle >= args.worker_idle_timeout:
                logger.info("idle %.0fs (>= --worker-idle-timeout); exiting "
                            "after %d trials", idle, n_done)
                break
            continue

        idle = 0.0
        trial_id, task = claimed
        try:
            label_set = tuple(task["label_set"].split("+"))
            label_array = _label_matrix(mapping, label_set)
            cv = cross_validate(
                label_array, flux, ivar, dispersion, label_set,
                order=int(task["order"]),
                regularization=float(task["regularization"]),
                n_splits=n_splits, seed=args.seed, age_reliability=age_masks)
            row = _summarize(cv, label_set)
            result = {"status": "ok", "trial_id": trial_id, "metrics": row}
            logger.info("trial %s done: %s=%.5g", trial_id, args.metric,
                        row.get(args.metric, float("nan")))
        except Exception as exc:                       # keep the worker alive
            logger.warning("trial %s FAILED: %s", trial_id, exc)
            result = {"status": "error", "trial_id": trial_id,
                      "error": str(exc)}

        _atomic_write_json(
            os.path.join(paths[_RESULTS], trial_id + ".json"), result)
        n_done += 1


# --------------------------------------------------------------------------- #
#  Controller / agent drivers                                                  #
# --------------------------------------------------------------------------- #

def _resolve_label_sets(args):
    """
    Label sets for the controller's sweep config. Prefer explicit --label-set;
    else read the worker-published, data-validated list from the broker (waiting
    briefly for the worker to come up); else fall back to the pure-string
    builder (no data / no science-stack import).
    """
    if args.label_sets:
        return list(args.label_sets)

    published = os.path.join(args.broker, "label_sets.json") if args.broker \
        else None
    if published:
        waited = 0.0
        while not os.path.exists(published) and waited < args.label_sets_wait:
            logger.info("waiting for the worker to publish label_sets.json ...")
            time.sleep(min(10.0, args.poll_interval))
            waited += min(10.0, args.poll_interval)
        if os.path.exists(published):
            sets = _read_json(published)["label_sets"]
            return [tuple(s.split("+")) for s in sets]

    # Pure-string fallback (no data, no thecannon import).
    if __package__:
        from scripts.run_sweep import build_label_sets
    else:
        from run_sweep import build_label_sets
    return build_label_sets(args.base, args.age_cols, args.mass_col,
                            args.abundances, args.label_set_mode)


def run_controller(args):
    """ Create (or attach to) a W&B sweep and run a brokering agent. """
    import wandb
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    _broker_paths(args.broker)

    if args.sweep_id:
        sweep_id = args.sweep_id
        logger.info("attaching controller agent to existing sweep %s", sweep_id)
    else:
        label_sets = _resolve_label_sets(args)
        sweep_config = build_sweep_config(
            label_sets, args.orders, args.method, args.metric, args.goal,
            reg_values=args.reg_values, reg_min=args.reg_min,
            reg_max=args.reg_max)
        print("\n=== W&B sweep config ({0}) ===".format(args.method))
        print(json.dumps(sweep_config, indent=2))
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project,
                               entity=args.wandb_entity)
        print("\nCreated sweep: {0}".format(sweep_id))
        with open(os.path.join(args.broker, _SWEEP_ID_FILE), "w") as fp:
            fp.write(sweep_id + "\n")
        print("Wrote sweep id to {0}; attach more controllers with "
              "--sweep-id {1}".format(
                  os.path.join(args.broker, _SWEEP_ID_FILE), sweep_id))

    objective = make_broker_objective(
        args.broker, args.metric, args.poll_interval, args.trial_timeout,
        wandb_project=args.wandb_project)
    wandb.agent(sweep_id, function=objective, count=args.count)


def run_agent(args):
    """ All-in-one mode: build/attach a sweep and evaluate inline. """
    import wandb
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    (mapping, flux, ivar, dispersion, label_sets, n_splits,
     age_masks) = _load_data(args)

    objective = make_inline_objective(
        mapping, flux, ivar, dispersion, n_splits, args.seed, args.metric,
        wandb_project=args.wandb_project, age_masks=age_masks)

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_config = build_sweep_config(
            label_sets, args.orders, args.method, args.metric, args.goal,
            reg_values=args.reg_values, reg_min=args.reg_min,
            reg_max=args.reg_max)
        print("\n=== W&B sweep config ({0}) ===".format(args.method))
        print(json.dumps(sweep_config, indent=2))
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project,
                               entity=args.wandb_entity)
        print("\nCreated sweep: {0}".format(sweep_id))
    wandb.agent(sweep_id, function=objective, count=args.count)


# --------------------------------------------------------------------------- #
#  Data loading + dry-run                                                      #
# --------------------------------------------------------------------------- #

def _load_data(args):
    """ Return ``(mapping, flux, ivar, dispersion, label_sets, n_splits)``. """
    if __package__:
        from scripts.train_cannon import (load_spectra, normalize_spectra,
                                          quality_mask)
        from scripts.run_sweep import (build_label_sets, filter_existing,
                                       finite_label_mapping)
    else:
        from train_cannon import (load_spectra, normalize_spectra, quality_mask)
        from run_sweep import (build_label_sets, filter_existing,
                              finite_label_mapping)

    if args.demo:
        names, label_source, dispersion, flux, ivar = load_golden()
        label_sets = [tuple(names[:2]), tuple(names)]
        n_splits = min(args.n_splits, 3)
        print("Demo on golden data: {0} stars, {1} pixels, labels {2}".format(
            flux.shape[0], flux.shape[1], names))
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
        label_sets = args.label_sets or build_label_sets(
            args.base, args.age_cols, args.mass_col, args.abundances,
            args.label_set_mode)
        label_sets = filter_existing(label_sets, label_source)
        if not label_sets:
            raise ValueError("no usable label sets; check --base/--age-cols")
        n_splits = args.n_splits

    label_union = list(dict.fromkeys(
        name for label_set in label_sets for name in label_set))
    mapping, finite = finite_label_mapping(label_source, label_union)
    age_masks = {age: mask[finite]
                 for age, mask in age_reliability_masks(label_source).items()}
    return (mapping, flux[finite], ivar[finite], dispersion, label_sets,
            n_splits, age_masks)


def _dry_run(args):
    """ Build/print the sweep config; for data-backed roles, probe once. """
    if args.role == "controller" and not args.demo and not args.spectra_present:
        label_sets = args.label_sets or [("raw_teff", "raw_logg", "raw_fe_h")]
        print("\n=== W&B sweep config ({0}) ===".format(args.method))
        print(json.dumps(build_sweep_config(
            label_sets, args.orders, args.method, args.metric, args.goal,
            reg_values=args.reg_values, reg_min=args.reg_min,
            reg_max=args.reg_max), indent=2))
        print("\n[dry-run] controller config OK (no data loaded).")
        return

    if __package__:
        from scripts.sweep_cannon import (_label_matrix, cross_validate,
                                          _summarize)
    else:
        from sweep_cannon import _label_matrix, cross_validate, _summarize

    (mapping, flux, ivar, dispersion, label_sets, n_splits,
     age_masks) = _load_data(args)
    print("\n=== W&B sweep config ({0}) ===".format(args.method))
    print(json.dumps(build_sweep_config(
        label_sets, args.orders, args.method, args.metric, args.goal,
        reg_values=args.reg_values, reg_min=args.reg_min,
        reg_max=args.reg_max), indent=2))

    reg_probe = (args.reg_values[0] if args.reg_values
                 else (args.reg_min * args.reg_max) ** 0.5)     # geo-mean
    label_set = tuple(label_sets[0])
    cv = cross_validate(_label_matrix(mapping, label_set), flux, ivar,
                        dispersion, label_set, order=int(args.orders[0]),
                        regularization=float(reg_probe), n_splits=n_splits,
                        seed=args.seed, age_reliability=age_masks)
    row = _summarize(cv, label_set)
    print("\n=== dry-run objective probe (label_set={0}, order={1}, reg={2:g}) ==="
          .format("+".join(label_set), args.orders[0], reg_probe))
    for key in ("mean_r2", "mean_sigma_mad", "mean_pull_std",
                "median_r_chi_sq", "theta_frac_zero"):
        if key in row:
            print("  {0:<18s} {1:.5g}".format(key, row[key]))
    print("  -> optimizer target '{0}' = {1:.5g}".format(
        args.metric, row.get(args.metric)))
    print("\n[dry-run] wiring OK; drop --dry-run (and `wandb login`) to run the "
          "real sweep.")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", default="agent",
                        choices=["agent", "controller", "worker"],
                        help="agent: all-in-one (needs internet). controller: "
                             "copyq brokering agent. worker: compute-queue loop.")
    # Data / label-set construction (mirrors scripts.run_sweep).
    parser.add_argument("--spectra", default=None,
                        help="parquet table of spectra + labels")
    parser.add_argument("--continuum-list", default=None,
                        help="text file of continuum pixel indices")
    add_label_builder_args(parser)
    add_filter_arg(parser)
    # Search space.
    parser.add_argument("--orders", type=lambda s: [int(x) for x in s.split(",")],
                        default=[1, 2])
    parser.add_argument("--reg-values",
                        type=lambda s: [float(x) for x in s.split(",")],
                        default=None,
                        help="discrete regularization set (may include 0); "
                             "overrides the log-uniform --reg-min/--reg-max")
    parser.add_argument("--reg-min", type=float, default=1e0,
                        help="log-uniform regularization lower bound (>0)")
    parser.add_argument("--reg-max", type=float, default=1e5,
                        help="log-uniform regularization upper bound (>0)")
    # Objective / search.
    parser.add_argument("--method", default="bayes",
                        choices=["bayes", "random", "grid"])
    parser.add_argument("--metric", default="mean_r2",
                        help="metric key to optimize (a _summarize column)")
    parser.add_argument("--goal", default="maximize",
                        choices=["maximize", "minimize"])
    parser.add_argument("--count", type=int, default=30,
                        help="number of trials this controller/agent runs")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    # Broker (controller + worker).
    parser.add_argument("--broker", default=None,
                        help="shared-filesystem broker directory (controller and "
                             "worker must point at the same path)")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="seconds between broker polls")
    parser.add_argument("--trial-timeout", type=float, default=3600.0,
                        help="controller: max seconds to wait for one result "
                             "(0 = wait forever)")
    parser.add_argument("--worker-idle-timeout", type=float, default=0.0,
                        help="worker: exit after this many idle seconds "
                             "(0 = run until STOP / walltime)")
    parser.add_argument("--label-sets-wait", type=float, default=300.0,
                        help="controller: max seconds to wait for the worker to "
                             "publish label_sets.json before falling back")
    # W&B plumbing.
    parser.add_argument("--wandb-project", default="cannon-bayes-sweep")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"],
                        help="bayes needs 'online' for the controller")
    parser.add_argument("--sweep-id", default=None,
                        help="attach to an existing sweep (ENTITY/PROJECT/ID)")
    parser.add_argument("--jax-cache-dir",
                        default=os.environ.get("JAX_COMPILATION_CACHE_DIR",
                                               "~/.cache/thecannon-jax"),
                        help="persistent XLA compile cache ('' to disable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build/print config (+probe for data roles); no W&B")
    parser.add_argument("--demo", action="store_true",
                        help="use the bundled golden data")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    # Default the data paths to the run_sweep defaults only when needed (keeps
    # the controller, which never loads data, free of that dependency).
    args.spectra_present = args.spectra is not None
    if args.role in ("agent", "worker") and not args.demo:
        if __package__:
            from scripts.train_cannon import DEFAULT_DATA_DIR
        else:
            from train_cannon import DEFAULT_DATA_DIR
        if args.spectra is None:
            args.spectra = os.path.join(DEFAULT_DATA_DIR, "cleaned_ages.parquet")
        if args.continuum_list is None:
            args.continuum_list = os.path.join(DEFAULT_DATA_DIR, "continuum.list")

    if args.role in ("controller", "worker") and not args.broker:
        parser.error("--broker is required for --role controller/worker")

    # Persistent XLA cache only helps the compute roles (and never in dry-run).
    if args.jax_cache_dir and not args.dry_run and args.role in ("agent",
                                                                 "worker"):
        if __package__:
            from scripts.run_sweep import enable_jax_compilation_cache
        else:
            from run_sweep import enable_jax_compilation_cache
        enable_jax_compilation_cache(args.jax_cache_dir)

    if args.dry_run:
        _dry_run(args)
    elif args.role == "worker":
        run_worker(args)
    elif args.role == "controller":
        run_controller(args)
    else:
        run_agent(args)


if __name__ == "__main__":
    main()
