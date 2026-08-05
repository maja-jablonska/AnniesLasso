#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Predict ages for a PRE-SELECTED sample with a saved CannonModel -- the
script version of ``notebooks/predict_ages.ipynb``.

Unlike :mod:`scripts.apply_rgb_ages` this does no evolutionary-state
selection at all: the input sample must already be pure RGB (or RC, e.g.
the ``bulge_{rgb,rc}_spectra.parquet`` tables), and ``--state`` must match
the population the model was trained on -- it only stamps the
``target_state`` column of the catalogue. The spectra are pseudo-continuum-
normalized identically to training and the test step writes a per-star age
catalogue (``age_gyr``, ``age_gyr_err``, all Cannon labels + errors,
``chi_sq``).

JAX device selection is via the ``JAX_PLATFORMS`` environment variable
(this script does not force one).

Usage
-----
::

    python -m scripts.predict_ages \\
        --model /path/to/trained_cannon_model_rgb.pkl \\
        --spectra /path/to/bulge_rgb_spectra.parquet \\
        --state rgb \\
        --output /path/to/cannon_ages_rgb.parquet
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import argparse
import logging
import os

import numpy as np

import thecannon as tc

# Work both as a package module (`python -m scripts.predict_ages`) and when
# run directly from the scripts/ directory.
try:
    from scripts.train_cannon import (load_spectra, normalize_spectra,
                                      quality_mask, DEFAULT_CONTINUUM_LIST)
    from scripts.apply_rgb_ages import RGB, HEB, STATE_LABELS, apply_model
except ImportError:
    from train_cannon import (load_spectra, normalize_spectra, quality_mask,
                              DEFAULT_CONTINUUM_LIST)
    from apply_rgb_ages import RGB, HEB, STATE_LABELS, apply_model

logger = logging.getLogger("thecannon.predict_ages")

# --state values follow the model file naming (trained_cannon_model_{rgb,rc});
# rc is the red clump, i.e. core-helium burning.
STATES = {"rgb": RGB, "rc": HEB}


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="path to the saved (pickled) CannonModel")
    parser.add_argument("--spectra", required=True,
                        help="parquet table of spectra to estimate ages for; "
                             "must already be pure --state stars")
    parser.add_argument("--state", default="rgb", choices=sorted(STATES),
                        help="evolutionary state of the sample (and of the "
                             "model): rgb or rc (default: rgb)")
    parser.add_argument("--continuum-list", default=DEFAULT_CONTINUUM_LIST,
                        help="text file of continuum pixel indices (must be "
                             "the one used to train the model)")
    parser.add_argument("--quality-cut", action="store_true",
                        help="apply the spectrum_flags / warn_* quality cuts "
                             "(off by default, matching the notebook)")
    parser.add_argument("--output", default=None,
                        help="output catalogue path (parquet; a .csv suffix "
                             "writes CSV instead; default: "
                             "cannon_ages_<state>.parquet)")
    parser.add_argument("--test-batch-size", type=int, default=None,
                        help="spectra fit per batch in the test step; lower it "
                             "if the device OOMs (default: memory-aware auto)")
    parser.add_argument("--chunk-size", type=int, default=4000,
                        help="stars per test-step call. `model.test` uploads "
                             "the whole flux/ivar array it is handed (and "
                             "copies it again to batch it), so its device "
                             "footprint scales with the number of stars, not "
                             "with --test-batch-size; chunking bounds it. "
                             "0 = one call for the whole sample (default: 4000)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable INFO-level logging")
    args = parser.parse_args()
    output = args.output or "cannon_ages_{0}.parquet".format(args.state)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    model = tc.CannonModel.read(args.model)
    if not model.is_trained:
        raise ValueError("model at {0} is not trained".format(args.model))
    logger.info("loaded %s (labels: %s)", model,
                ", ".join(model.vectorizer.label_names))

    table, dispersion, flux, ivar = load_spectra(args.spectra)
    if model.dispersion is not None \
    and np.asarray(model.dispersion).size != dispersion.size:
        raise ValueError(
            "sample has {0} pixels but the model was trained on {1}; the "
            "spectra are on a different wavelength grid".format(
                dispersion.size, np.asarray(model.dispersion).size))

    if args.quality_cut:
        keep = np.asarray(quality_mask(table), dtype=bool)
        logger.info("quality cuts keep %d/%d stars", int(keep.sum()),
                    len(table))
        table, flux, ivar = table.loc[keep], flux[keep], ivar[keep]
        if len(table) == 0:
            raise ValueError("no stars survive the quality cuts")

    normalized_flux, normalized_ivar = normalize_spectra(
        dispersion, flux, ivar, args.continuum_list)
    # Bring the normalized spectra back to the host so the device copy is
    # freed; each test-step chunk re-uploads only its own slice.
    normalized_flux = np.asarray(normalized_flux)
    normalized_ivar = np.asarray(normalized_ivar)

    # The state is known a priori for every star, so the provenance columns
    # are constant: no seismic lookup, no classifier probability.
    target = STATES[args.state]

    chunk = args.chunk_size if args.chunk_size > 0 else len(table)
    catalogues = []
    for start in range(0, len(table), chunk):
        stop = min(start + chunk, len(table))
        logger.info("test step: stars %d-%d of %d", start, stop, len(table))
        catalogues.append(apply_model(
            model, table.iloc[start:stop],
            normalized_flux[start:stop], normalized_ivar[start:stop],
            source=np.full(stop - start, "preselected"),
            state_proba=np.full(stop - start, np.nan),
            target=target, test_batch_size=args.test_batch_size))
    catalogue = (catalogues[0] if len(catalogues) == 1
                 else pd.concat(catalogues, ignore_index=True))

    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    if output.endswith(".csv"):
        catalogue.to_csv(output, index=False)
    else:
        catalogue.to_parquet(output, index=False)
    logger.info("wrote %d-star age catalogue to %s", len(catalogue), output)

    finite = np.isfinite(catalogue["age_gyr"])
    physical = finite & ~catalogue["flag_unphysical_age"]
    print("\n=== ages for {0} preselected {1} stars ===".format(
        len(catalogue), STATE_LABELS[target]))
    print("  median age: {0:.2f} Gyr | 16th-84th: {1:.2f}-{2:.2f} Gyr | "
          "{3} unphysical (outside 0-20 Gyr) | median r_chi_sq {4:.2f}".format(
              float(np.median(catalogue.loc[physical, "age_gyr"])),
              float(np.percentile(catalogue.loc[physical, "age_gyr"], 16)),
              float(np.percentile(catalogue.loc[physical, "age_gyr"], 84)),
              int((~physical & finite).sum()),
              float(catalogue["r_chi_sq"].median())))


if __name__ == "__main__":
    main()
