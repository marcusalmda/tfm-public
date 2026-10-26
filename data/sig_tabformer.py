"""
sig_tabformer.py — significância pareada entre dois modelos no MESMO test set.
Reusa DeLong (AUC) e bootstrap pareado (PR-AUC, F1) do _QUALIFICACAO_AMOSTRA/significancia.py.

Uso (a partir da raiz do projeto):
  python data/sig_tabformer.py \
    --a results/fraud_tfm_ar_tabformer/test_predictions.csv   --name-a "TFM+grafo" \
    --b results/fraud_causal_ar_tabformer/test_predictions.csv --name-b "TFM-seq" \
    --thr-a 0.9995 --thr-b 0.9988

Cada CSV precisa ter as colunas 'label' e 'prob' (as duas predições devem ser do MESMO
conjunto de teste, com os rótulos alinhados). Convenção do delta: A - B.
Se --thr-* não for passado, usa o melhor F1 no próprio teste (mesmo procedimento p/ os dois).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


# ---- DeLong (AUC) e bootstrap pareado (embutidos; código da versão testada) ----
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong(y, s1, s2):
    """AUC(s1)-AUC(s2) (pp), z e p bilateral (DeLong pareado)."""
    idx = np.argsort(-y, kind="stable"); y = y[idx]; m = int(y.sum())
    P = np.vstack([s1[idx], s2[idx]]); n = P.shape[1] - m
    tx = np.vstack([_midrank(P[r, :m]) for r in range(2)])
    ty = np.vstack([_midrank(P[r, m:]) for r in range(2)])
    tz = np.vstack([_midrank(P[r]) for r in range(2)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    l = np.array([[1.0, -1.0]]); var = float((l @ cov @ l.T).item())
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float((aucs[0] - aucs[1]) * 100), float(z), float(2 * stats.norm.sf(abs(z)))


def prauc(y, p):
    return average_precision_score(y, p)


def boot(y, pa, pb, metric_fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d0 = metric_fn(y, pa) - metric_fn(y, pb)
    n = len(y); deltas = []
    for _ in range(B):
        idx = rng.integers(0, n, n); yy = y[idx]
        if yy.sum() < 2 or yy.sum() > len(yy) - 2:
            continue
        deltas.append(metric_fn(yy, pa[idx]) - metric_fn(yy, pb[idx]))
    d = np.array(deltas)
    return float(d0 * 100), float(np.percentile(d, 2.5) * 100), float(np.percentile(d, 97.5) * 100), \
        float(2 * min((d > 0).mean(), (d < 0).mean()))


class _Sig:
    delong = staticmethod(delong)
    boot = staticmethod(boot)
    prauc = staticmethod(prauc)


sig = _Sig()


def load(path):
    d = pd.read_csv(path)
    return d["label"].values.astype(float), d["prob"].values


def best_thr(y, p):
    """Threshold de melhor F1, varredura vetorizada O(n log n) (exata).

    Ordena por prob desc; ao baixar o limiar, cada ponto vira positivo um a um,
    então F1(k) = 2*tp(k) / (k + P). Pega o k que maximiza.
    """
    y = np.asarray(y); p = np.asarray(p)
    P = float(y.sum())
    if P == 0:
        return 0.5
    order = np.argsort(-p, kind="stable")
    ys = y[order]; ps = p[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    f1 = 2.0 * tp / (k + P)
    i = int(np.argmax(f1))
    return float(ps[i])


def f1_boot(y, pa, pb, ta, tb, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d0 = f1_score(y, (pa >= ta).astype(int), zero_division=0) - f1_score(y, (pb >= tb).astype(int), zero_division=0)
    n = len(y); dd = []
    for _ in range(B):
        idx = rng.integers(0, n, n); yy = y[idx]
        if yy.sum() < 2 or yy.sum() > len(yy) - 2:
            continue
        dd.append(f1_score(yy, (pa[idx] >= ta).astype(int), zero_division=0)
                  - f1_score(yy, (pb[idx] >= tb).astype(int), zero_division=0))
    dd = np.array(dd)
    return float(d0 * 100), float(np.percentile(dd, 2.5) * 100), float(np.percentile(dd, 97.5) * 100), \
        float(2 * min((dd > 0).mean(), (dd < 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--thr-a", type=float, default=None)
    ap.add_argument("--thr-b", type=float, default=None)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()

    da = pd.read_csv(args.a)
    db = pd.read_csv(args.b)
    if "id" in da.columns and "id" in db.columns:
        # alinha por id (robusto a ordem diferente)
        m = da[["id", "label", "prob"]].merge(db[["id", "label", "prob"]], on="id", suffixes=("_a", "_b"))
        assert len(m) > 0, "Nenhum id em comum entre os dois CSV."
        assert np.array_equal(m["label_a"].values, m["label_b"].values), "Rótulos divergem para o mesmo id."
        y = m["label_a"].values.astype(float); pa = m["prob_a"].values; pb = m["prob_b"].values
        print(f"[align] casados por id: {len(m):,} amostras")
    else:
        ya, pa = da["label"].values.astype(float), da["prob"].values
        yb, pb = db["label"].values.astype(float), db["prob"].values
        assert len(ya) == len(yb) and np.array_equal(ya, yb), \
            "Sem coluna 'id' e ordem/rotulos diferentes — regenere as predições com id."
        y = ya
    ta = args.thr_a if args.thr_a is not None else best_thr(y, pa)
    tb = args.thr_b if args.thr_b is not None else best_thr(y, pb)

    print(f"\nComparação: {args.name_a} (A) vs {args.name_b} (B) | n={len(y)} pos={int(y.sum())} | delta = A - B\n")
    print(f"{'métrica':<10} {args.name_a:>12} {args.name_b:>12} {'Δ (A-B)':>10} {'IC95%':>20} {'p':>10}")

    auc_a, auc_b = roc_auc_score(y, pa), roc_auc_score(y, pb)
    dauc, z, pd_ = sig.delong(y, pa, pb)
    print(f"{'AUC':<10} {auc_a:>12.4f} {auc_b:>12.4f} {dauc:>9.2f}p {'(DeLong)':>20} {pd_:>10.4g}")

    pr_a, pr_b = average_precision_score(y, pa), average_precision_score(y, pb)
    dpr, lo, hi, ppr = sig.boot(y, pa, pb, sig.prauc, B=args.B)
    print(f"{'PR-AUC':<10} {pr_a:>12.4f} {pr_b:>12.4f} {dpr:>9.2f}p {f'[{lo:.2f},{hi:.2f}]':>20} {ppr:>10.4g}")

    f1_a = f1_score(y, (pa >= ta).astype(int), zero_division=0)
    f1_b = f1_score(y, (pb >= tb).astype(int), zero_division=0)
    df1, lo1, hi1, pf1 = f1_boot(y, pa, pb, ta, tb, B=args.B)
    print(f"{'F1':<10} {f1_a:>12.4f} {f1_b:>12.4f} {df1:>9.2f}p {f'[{lo1:.2f},{hi1:.2f}]':>20} {pf1:>10.4g}")
    print(f"\n(thr_a={ta:.4f}, thr_b={tb:.4f}; Δ e IC em pontos percentuais; "
          f"significativo se p<0,05 e o IC exclui zero)")


if __name__ == "__main__":
    main()
