#!/usr/bin/env python3
"""Evaluate rel-hm churn on the OFFICIAL RelBench test set via the RelBench API.

The official rel-hm test set has no public labels, so the only leaderboard-
comparable number comes from ``task.evaluate(pred)``. This script takes the test
predictions written by the TFM fine-tune and submits them to that API.

Requires a dataset prepared with ``prepare_relbench_hm.py --official-test``: the
test sample ids are ``test{j}``, where ``j`` is the row index in
``task.get_table("test")``, so predictions align with the task table. Rows with
no usable history (customer with < min-events before the cutoff) are filled with
the train churn base rate, exactly as the CoLES baseline does.

Usage:
  python data/eval_relbench_official.py \
    --preds results/relbench_graph_s42_full/test_predictions.csv \
    --base-rate data/processed_relbench_hm_official/base_rate.pt
"""
import argparse

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True,
                    help="test_predictions.csv do fine-tune (colunas id=test{j}, prob)")
    ap.add_argument("--base-rate", default=None,
                    help="base_rate.pt (fallback para linhas sem histórico); default 0.0")
    args = ap.parse_args()

    from relbench.tasks import get_task
    task = get_task("rel-hm", "user-churn", download=True)
    n = len(task.get_table("test").df)

    base = 0.0
    if args.base_rate:
        base = float(torch.load(args.base_rate, weights_only=False))
    pred_full = np.full(n, base, dtype=float)

    df = pd.read_csv(args.preds)
    cov = 0
    for sid, prob in zip(df["id"].astype(str), df["prob"].astype(float)):
        if sid.startswith("test"):
            try:
                j = int(sid[4:])
            except ValueError:
                continue
            if 0 <= j < n:
                pred_full[j] = prob
                cov += 1
    print(f"[oficial] cobertos {cov:,}/{n:,} do rb_test (resto = base rate {base:.4f})")

    try:
        m = task.evaluate(pred_full)
    except Exception as e:  # algumas versões exigem a target_table explícita
        print("[oficial] task.evaluate(pred) falhou:", repr(e)[:120], "— tentando com target_table")
        m = task.evaluate(pred_full, task.get_table("test"))
    print(f"[oficial] task.evaluate no rb_test: {m}")


if __name__ == "__main__":
    main()
