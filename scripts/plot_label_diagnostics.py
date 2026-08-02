#!/usr/bin/env python
"""Standard label-diagnostic plots for the Cannon, in the shared style.

Predicted-vs-truth for every label coloured by the reduced chi^2 of the
spectral fit, residual-systematics grids (offsets vs labels / SNR), and a
Kiel map coloured by residual, via the shared ``stardiag`` module (sibling
checkout). Works on the OOF-predictions parquet persisted by
notebooks/train_rgb_wilett_all_missions.ipynb (cell 39) — both the
``<label>_truth``/``<label>_pred`` + ``val_chi2`` naming and the legacy
``pred_<label>`` naming are handled.

Usage:
    python scripts/plot_label_diagnostics.py               # auto-detect
    python scripts/plot_label_diagnostics.py --results <parquet>
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # AnniesLasso/scripts
REPO = HERE.parent

for _c in (REPO.parent / "stardiag", Path.home() / "code" / "stardiag",
           Path.home() / "scr_mk27" / "stardiag"):
    if (_c / "stardiag.py").exists():
        sys.path.insert(0, str(_c))
        break
else:
    sys.exit("stardiag module not found (expected a 'stardiag' checkout "
             "next to this repo)")
import stardiag  # noqa: E402

RESULTS_CANDIDATES = [
    Path.home() / "scr_mk27" / "cannon_oof_predictions_rgb_all_missions.parquet",
    Path.home() / "scr_mk27" / "cannon-results-rgb-wilett-all-missions.parquet",
    REPO / "cannon-results-rgb-wilett-all-missions.parquet",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", help="OOF predictions parquet "
                    "(default: auto-detect)")
    ap.add_argument("--plot-dir", default=str(REPO / "plots" / "diagnostics"),
                    help="where to write PNGs")
    args = ap.parse_args()

    results = (Path(args.results) if args.results else
               next((p for p in RESULTS_CANDIDATES if p.exists()), None))
    if results is None or not results.exists():
        sys.exit(f"no predictions parquet found; looked for "
                 f"{[str(p) for p in RESULTS_CANDIDATES]} — persist it from "
                 f"notebooks/train_rgb_wilett_all_missions.ipynb first")
    print(f"predictions: {results}")

    spec = stardiag.load_cannon(results)
    for p in stardiag.make_all(spec, args.plot_dir):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
