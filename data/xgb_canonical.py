#!/usr/bin/env python3
"""
XGBoost CANÔNICO (as 59 features document-level do paper) no NOSSO split.

Reusa o MESMO extrator do paper (src.baselines.feature_extractor.aggregate_features_for_documents)
e os MESMOS hiperparâmetros do run_fraud_significance.py, mas lendo os NOSSOS arquivos de split
(sample_split_strat.pt / sample_split.pt) — assim o teste do XGBoost é idêntico ao do TFM e do
CoLES (DeLong/bootstrap pareados válidos).

Features (max_event_type_id=20, sem caps de specific/counterparty): event_count + value_processed
{sum,mean,std,min,max} + delta_t_seconds {sum,mean} + event_type_id freq 0..19 + time_hour freq 0..23
+ time_weekday freq 0..6 = 59, igual ao paper.

Uso:
  python data/xgb_canonical.py --dir data/processed_tabformer_temporal \
    --labels-name fraud_labels.pt --split-name sample_split_strat.pt \
    --out results/xgb_canonical_tab_strat/seed42/test_predictions.csv --seed 42
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raiz do projeto
import finetune_causal_ar_fraud as seqbase
from src.baselines.feature_extractor import aggregate_features_for_documents
KEYS, CAT_VOCAB, best_thr = seqbase.KEYS, seqbase.CAT_VOCAB, seqbase.best_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--labels-name", default="fraud_labels.pt")
    ap.add_argument("--split-name", default="sample_split_strat.pt")
    ap.add_argument("--max-event-type-id", type=int, default=20)  # => 59 features, igual ao paper
    ap.add_argument("--out", default="results/xgb_canonical/test_predictions.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import torch
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    seqs = torch.load(f"{args.dir}/discrete_event_sequences.pt", weights_only=False)
    labs = torch.load(f"{args.dir}/{args.labels_name}", weights_only=False)
    spl = torch.load(f"{args.dir}/{args.split_name}", weights_only=False)
    pick = lambda s: [d for d in spl if spl[d] == s and d in seqs and d in labs]
    tr_ids, va_ids, te_ids = pick("train"), pick("val"), pick("test")
    print(f"amostras: train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)} "
          f"| pos_test={np.mean([int(labs[d]) for d in te_ids]):.4f}")

    # feature_info mínimo (o extrator usa vocab de time_hour/time_weekday; default 24/7)
    feature_info = {"event_type_id": {"vocab_size": int(CAT_VOCAB.get("event_type_id", args.max_event_type_id))},
                    "time_hour": {"vocab_size": 24}, "time_weekday": {"vocab_size": 7}}
    feats = aggregate_features_for_documents(seqs, tr_ids + va_ids + te_ids, KEYS, feature_info,
                                             max_event_type_id=args.max_event_type_id)
    if "document_id" in feats.columns:
        feats = feats.set_index("document_id")
    print(f"[xgb] features: {feats.shape[1]} colunas (esperado 59 com max_event_type_id=20)")

    keep = lambda ids: [d for d in ids if d in feats.index]
    tr_ids, va_ids, te_ids = keep(tr_ids), keep(va_ids), keep(te_ids)
    Xtr, Xva, Xte = feats.loc[tr_ids], feats.loc[va_ids], feats.loc[te_ids]
    ytr = np.array([int(labs[d]) for d in tr_ids]); yva = np.array([int(labs[d]) for d in va_ids])
    yte = np.array([int(labs[d]) for d in te_ids])

    scaler = StandardScaler().fit(Xtr)
    xgb_params = {"eta": 0.05, "max_depth": 5, "subsample": 0.7, "colsample_bytree": 0.7,
                  "objective": "binary:logistic", "eval_metric": "auc", "seed": args.seed}
    dtrain = xgb.DMatrix(scaler.transform(Xtr), label=ytr)
    dval = xgb.DMatrix(scaler.transform(Xva), label=yva)
    dtest = xgb.DMatrix(scaler.transform(Xte))
    booster = xgb.train(xgb_params, dtrain, num_boost_round=500,
                        evals=[(dval, "val")], early_stopping_rounds=20, verbose_eval=False)
    it = (0, booster.best_iteration + 1)
    prob_va = booster.predict(dval, iteration_range=it)
    prob = booster.predict(dtest, iteration_range=it)

    thr = best_thr(yva, prob_va)  # limiar de F1 no val, igual ao TFM
    auc = float(roc_auc_score(yte, prob)); pr = float(average_precision_score(yte, prob))
    f1 = float(f1_score(yte, (prob >= thr).astype(int), zero_division=0))
    print(f"[xgb-canonico:{args.split_name}] teste: AUC={auc:.4f} PR-AUC={pr:.4f} F1={f1:.4f} (thr={thr:.4f})")

    out_dir = os.path.dirname(args.out)
    m = {"regime": args.split_name, "head": "xgb_canonical", "seed": args.seed,
         "auc": auc, "pr_auc": pr, "f1": f1, "thr": float(thr),
         "n_features": int(feats.shape[1]), "n_test": int(len(yte)), "pos_rate": float(yte.mean())}
    with open(os.path.join(out_dir, "metrics_xgb_canonical.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    pd.DataFrame({"id": te_ids, "label": yte, "prob": prob}).to_csv(args.out, index=False)
    print(f"[ok] salvo em {args.out}")


if __name__ == "__main__":
    main()
