#!/usr/bin/env python3
"""Compute the F1-optimal operating point from saved test predictions.

This is the F1 convention reported for the TabFormer fraud tables in the paper:
the threshold that maximises F1 on the test scores, applied uniformly to every
model (TFM, CoLES, XGBoost). It uses the same precision-recall sweep for all of
them, so no model is favoured and nothing is retrained — it only re-reads the
``test_predictions.csv`` written by the fine-tune / baseline scripts.

Rationale: on the sparse TabFormer fraud tests the validation threshold is
unstable (0.11-1.2% positives), so F1 at the validation threshold degenerates.
Reporting the F1-optimal point for all models keeps the F1 column comparable.
On the dense datasets (rel-hm, and the paper's private dataset) the validation
threshold is used instead; this script is only for the TabFormer F1 column.

Each CSV must have columns ``label`` and ``prob``.

Usage:
  # one file
  python data/compute_f1_optimal.py results/fraud_tfm_graphsample_strat_s42/test_predictions.csv

  # several, with labels, as a table
  python data/compute_f1_optimal.py \
      "TFM+graph=results/fraud_tfm_graphsample_strat_s42/test_predictions.csv" \
      "CoLES-FT=results/coles_tab_strat_sl16/seed42/ft_test_predictions.csv" \
      "XGBoost=results/xgb_canonical_tab_strat/seed42/test_predictions.csv"
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def f1_optimal(labels, probs):
    """Max F1 over all thresholds of the PR curve (the F1-optimal operating point)."""
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    prec, rec, _ = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    # precision_recall_curve appends a trailing (prec=1, rec=0) point with no
    # threshold; drop it before taking the max.
    return float(np.nanmax(f1[:-1]) * 100.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+",
                    help="test_predictions.csv paths, optionally as NAME=path")
    args = ap.parse_args()

    width = max((len(it.split("=", 1)[0]) if "=" in it else len(os.path.basename(it))
                 for it in args.items), default=4)
    for it in args.items:
        name, path = it.split("=", 1) if "=" in it else (os.path.basename(os.path.dirname(it)) or it, it)
        if not os.path.exists(path):
            print(f"{name:<{width}}  (arquivo nao encontrado: {path})")
            continue
        d = pd.read_csv(path)
        print(f"{name:<{width}}  F1*={f1_optimal(d['label'], d['prob']):.2f}")


if __name__ == "__main__":
    main()
