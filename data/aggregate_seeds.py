#!/usr/bin/env python3
"""
Agrega resultados multiseed e faz o teste pareado por semente (grafo vs seq).

Para cada semente: casa as predições por id, calcula AUC/PR-AUC/F1 dos dois modelos,
e o delta com significância (DeLong p/ AUC, bootstrap p/ PR-AUC e F1). No fim, reporta
média ± desvio de cada métrica e do delta, e em quantas sementes deu significativo.

Convenção de pastas (uma por semente):
  results/fraud_tfm_graphsample_fixed_s42/test_predictions.csv , ..._s43, ...
  results/fraud_seq_temporal_fixed_s42/test_predictions.csv , ...

Uso:
  python data/aggregate_seeds.py \
    --graph-dirs results/fraud_tfm_graphsample_fixed_s{SEED} \
    --seq-dirs   results/fraud_seq_temporal_fixed_s{SEED} \
    --seeds 42 43 44 45 46
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sig_tabformer as S   # delong, boot, prauc, best_thr, f1_boot


def load(path):
    d = pd.read_csv(path)
    return d


def recall_at_fpr(y, p, fpr=0.05):
    """Recall (TPR) com o limiar fixado pelo orçamento de FPR (sem peeking em F1)."""
    y = np.asarray(y); p = np.asarray(p)
    neg = p[y == 0]
    if len(neg) == 0 or (y == 1).sum() == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - fpr)       # limiar que deixa ~fpr dos negativos acima
    return float(((p >= thr) & (y == 1)).sum() / (y == 1).sum())


def metrics(y, p, thr=None):
    if thr is None:
        thr = S.best_thr(y, p)
    return (roc_auc_score(y, p), average_precision_score(y, p),
            f1_score(y, (p >= thr).astype(int), zero_division=0), thr,
            recall_at_fpr(y, p, 0.05))


def align(da, db):
    if "id" in da.columns and "id" in db.columns:
        m = da[["id", "label", "prob"]].merge(db[["id", "label", "prob"]], on="id", suffixes=("_a", "_b"))
        return m["label_a"].values.astype(float), m["prob_a"].values, m["prob_b"].values
    y = da["label"].values.astype(float)
    return y, da["prob"].values, db["prob"].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dirs", required=True, help="padrão com {SEED}, ex: results/..._s{SEED}")
    ap.add_argument("--seq-dirs", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--graph-file", default="test_predictions.csv",
                    help="nome do CSV no dir A (ex.: ft_test_predictions.csv p/ CoLES fine-tune)")
    ap.add_argument("--seq-file", default="test_predictions.csv",
                    help="nome do CSV no dir B")
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()

    rows = []
    for s in args.seeds:
        ga = os.path.join(args.graph_dirs.replace("{SEED}", str(s)), args.graph_file)
        sb = os.path.join(args.seq_dirs.replace("{SEED}", str(s)), args.seq_file)
        if not (os.path.exists(ga) and os.path.exists(sb)):
            print(f"[seed {s}] FALTANDO: {ga if not os.path.exists(ga) else sb}")
            continue
        y, pa, pb = align(load(ga), load(sb))
        auc_a, pr_a, f1_a, _, rec_a = metrics(y, pa)
        auc_b, pr_b, f1_b, _, rec_b = metrics(y, pb)
        dauc, z, p_auc = S.delong(y, pa, pb)
        dpr, lo, hi, p_pr = S.boot(y, pa, pb, S.prauc, B=args.B)
        rows.append({"seed": s, "auc_g": auc_a, "auc_s": auc_b, "d_auc": dauc, "p_auc": p_auc,
                     "pr_g": pr_a, "pr_s": pr_b, "d_pr": dpr, "p_pr": p_pr,
                     "f1_g": f1_a, "f1_s": f1_b, "rec_g": rec_a, "rec_s": rec_b})
        print(f"[seed {s}] AUC g={auc_a:.4f} s={auc_b:.4f} (Δ{dauc:+.2f}p, p={p_auc:.3g}) | "
              f"PR-AUC g={pr_a:.4f} s={pr_b:.4f} (Δ{dpr:+.2f}p, p={p_pr:.3g}) | "
              f"F1@orác g={f1_a:.4f} s={f1_b:.4f} | rec@FPR5 g={rec_a:.4f} s={rec_b:.4f}")

    if not rows:
        print("Nenhuma semente completa encontrada."); return
    df = pd.DataFrame(rows)
    def ms(col): return f"{df[col].mean():.4f} ± {df[col].std(ddof=1):.4f}"
    def msp(col): return f"{df[col].mean():+.2f} ± {df[col].std(ddof=1):.2f} p.p."
    print("\n==================== AGREGADO (n=%d sementes) ====================" % len(df))
    print(f"{'':8}{'grafo':>18}{'seq':>18}{'Δ (grafo-seq)':>22}")
    print(f"{'AUC':10}{ms('auc_g'):>18}{ms('auc_s'):>18}{msp('d_auc'):>22}")
    print(f"{'PR-AUC':10}{ms('pr_g'):>18}{ms('pr_s'):>18}{msp('d_pr'):>22}")
    print(f"{'F1@orác':10}{ms('f1_g'):>18}{ms('f1_s'):>18}")
    print(f"{'rec@FPR5':10}{ms('rec_g'):>18}{ms('rec_s'):>18}")
    nsig_auc = int((df['p_auc'] < 0.05).sum()); nsig_pr = int((df['p_pr'] < 0.05).sum())
    print(f"\nSementes com p<0,05 (pareado): AUC {nsig_auc}/{len(df)} | PR-AUC {nsig_pr}/{len(df)}")
    print("Reporte no paper: média ± desvio das métricas + delta médio; consistência da significância.")


if __name__ == "__main__":
    main()
