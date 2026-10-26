#!/usr/bin/env python3
"""Baseline causal-AR (sequencial puro, sem grafo) para FRAUDE.

Fraude nao tem score_date por amostra, entao NAO ha event-age/qtoken aqui — o
modelo classifica a partir da sequencia crua de eventos (ultima posicao valida).
Usa o MESMO split do TFM (estratificado, seed 42, test 0.3, val 0.1 do restante),
importando create_doc_id_splits, para o teste ser comparavel.

Salva as predicoes de teste (test_predictions.csv). O baseline tabular e o
XGBoost canonico (59 features) em data/xgb_canonical.py — nao ha XGBoost inline
aqui, para manter uma unica definicao de XGBoost em todo o repositorio.

Uso:
  python finetune_causal_ar_fraud.py \
      --pretrained-checkpoint checkpoints/pretrain_causal_ar/best_model.pt \
      --sequences processed/discrete_event_sequences.pt \
      --labels processed/fraud_labels.pt \
      --output-dir results/fraud_causal_ar
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

from src.models.causal_event_model import CausalEventModel
from src.utils.evaluation import create_doc_id_splits

CAT_INDICES = [0, 3, 4, 5, 6, 7]
CONT_INDICES = [1, 2]
KEYS = ["event_type_id", "value_processed", "delta_t_seconds", "counterparty_idx",
        "specific_event_idx", "time_hour", "time_weekday", "time_day_of_year"]
CAT_VOCAB = {"event_type_id": 20, "counterparty_idx": 60000, "specific_event_idx": 10000,
             "time_hour": 25, "time_weekday": 8, "time_day_of_year": 367}
CAT_EMB = {"event_type_id": 32, "counterparty_idx": 96, "specific_event_idx": 32,
           "time_hour": 8, "time_weekday": 8, "time_day_of_year": 16}
CAT_NAMES = ["event_type_id", "counterparty_idx", "specific_event_idx",
             "time_hour", "time_weekday", "time_day_of_year"]


def clamp(v, vocab):
    return max(0, min(int(v), vocab - 1))


class FraudSeqDataset(Dataset):
    def __init__(self, sequences, labels, doc_ids, max_len, cont_mean=None, cont_std=None):
        self.samples = sequences
        self.labels = labels
        self.ids = [d for d in doc_ids if d in sequences and len(sequences[d]) >= 1]
        self.max_len = max_len
        if cont_mean is None:
            rows = []
            for d in self.ids:
                for e in self.samples[d][-max_len:]:
                    rows.append([float(e.get(KEYS[i], 0.0)) for i in CONT_INDICES])
            t = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((1, len(CONT_INDICES)))
            self.cont_mean = t.mean(0); self.cont_std = t.std(0).clamp(min=1e-6)
        else:
            self.cont_mean, self.cont_std = cont_mean, cont_std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        d = self.ids[idx]
        ev = self.samples[d][-self.max_len:]
        cat = torch.tensor([[clamp(e.get(KEYS[i], 0), CAT_VOCAB[KEYS[i]]) for i in CAT_INDICES] for e in ev],
                           dtype=torch.long)
        cont = torch.tensor([[float(e.get(KEYS[i], 0.0)) for i in CONT_INDICES] for e in ev],
                            dtype=torch.float32)
        cont = (cont - self.cont_mean) / self.cont_std
        return {"cat": cat, "cont": cont, "label": torch.tensor(int(self.labels[d]), dtype=torch.long)}


def collate(batch):
    L = max(b["cat"].size(0) for b in batch)
    nc, nk = batch[0]["cat"].size(1), batch[0]["cont"].size(1)
    cat = torch.zeros(len(batch), L, nc, dtype=torch.long)
    cont = torch.zeros(len(batch), L, nk, dtype=torch.float32)
    pad = torch.ones(len(batch), L, dtype=torch.bool)
    labels = torch.stack([b["label"] for b in batch])
    for i, b in enumerate(batch):
        n = b["cat"].size(0)
        cat[i, :n] = b["cat"]; cont[i, :n] = b["cont"]; pad[i, :n] = False
    return {"cat": cat, "cont": cont, "pad": pad, "labels": labels}


@torch.no_grad()
def evaluate(model, loader, device, causal=True):
    model.eval()
    ys, ps = [], []
    for b in loader:
        logits = model.forward_classify(b["cat"].to(device), b["cont"].to(device),
                                        padding_mask=b["pad"].to(device), causal=causal)
        ps.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy()); ys.extend(b["labels"].numpy())
    return np.asarray(ys), np.asarray(ps)


def best_thr(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[int(np.nanargmax(f1[:-1]))]) if len(thr) else 0.5


def recall_at_fpr(y, p, target=0.05):
    fpr, tpr, _ = roc_curve(y, p); i = max(np.searchsorted(fpr, target, side="right") - 1, 0)
    return float(tpr[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-checkpoint", default=None)
    ap.add_argument("--sequences", default="processed/discrete_event_sequences.pt")
    ap.add_argument("--labels", default="processed/fraud_labels.pt")
    ap.add_argument("--output-dir", default="results/fraud_causal_ar")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["stratified", "tabformer", "relbench"], default="stratified",
                    help="stratified = split estratificado por seed (TabFormer in-distribution); "
                         "tabformer = temporal por sample_time + on_grid (TabFormer OOT); "
                         "relbench = usa o split pronto do RelBench (sample_split.pt)")
    ap.add_argument("--sample-time-path", default="data/processed_tabformer/sample_time.pt")
    ap.add_argument("--sample-on-grid-path", default="data/processed_tabformer/sample_on_grid.pt")
    ap.add_argument("--sample-split-path", default="data/processed_relbench_hm/sample_split.pt")
    ap.add_argument("--objective", choices=["auto", "ar", "mem"], default="auto",
                    help="modo de atencao no fine-tune: ar=causal, mem=bidirecional; auto=le do checkpoint")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequences = torch.load(args.sequences, weights_only=False)
    labels = torch.load(args.labels, weights_only=False)
    # intersecao rotulados presentes nas sequencias, ORDENADA — igual ao TFM
    # (src/datasets.py usa sorted(...)). A ordem importa: create_doc_id_splits
    # com o mesmo seed so produz o MESMO split se a lista de entrada for identica.
    if args.split == "tabformer":
        stime = torch.load(args.sample_time_path, weights_only=False)
        ongrid = torch.load(args.sample_on_grid_path, weights_only=False)
        docs = sorted([d for d in labels if d in sequences and d in stime])
        docs.sort(key=lambda d: stime[d])
        n = len(docs); n_test = int(n * args.test_size); n_val = int((n - n_test) * args.val_size)
        n_train = n - n_test - n_val
        tr = docs[:n_train]; va = docs[n_train:n_train + n_val]
        te = [d for d in docs[n_train + n_val:] if ongrid.get(d, True)]
        print(f"Split TabFormer (temporal): train={len(tr)} val={len(va)} test={len(te)} (grade natural) "
              f"| pos_train={sum(int(labels[d]) for d in tr)} pos_test={sum(int(labels[d]) for d in te)}")
    elif args.split == "relbench":
        ssplit = torch.load(args.sample_split_path, weights_only=False)
        docs = sorted([d for d in labels if d in sequences and d in ssplit])
        tr = [d for d in docs if ssplit[d] == "train"]
        va = [d for d in docs if ssplit[d] == "val"]
        te = [d for d in docs if ssplit[d] == "test"]
        print(f"Split RelBench: train={len(tr)} val={len(va)} test={len(te)} "
              f"| pos_train={sum(int(labels[d]) for d in tr)} pos_test={sum(int(labels[d]) for d in te)}")
    else:
        docs = sorted([d for d in labels if d in sequences])
        print(f"Docs rotulados com sequencia: {len(docs)} | positivos: {sum(int(labels[d]) for d in docs)}")
        tr, va, te = create_doc_id_splits(docs, {d: int(labels[d]) for d in docs},
                                          test_size=args.test_size, validation_size=args.val_size,
                                          seed=args.seed, stratify=True)
        if os.path.exists(args.sample_on_grid_path):
            og = torch.load(args.sample_on_grid_path, weights_only=False)
            te = [d for d in te if og.get(d, True)]
            print(f"[stratified] teste restrito à grade natural (on_grid): {len(te)} amostras")
        print(f"Split estratificado: train={len(tr)} val={len(va)} test={len(te)} "
              f"| pos_test={sum(int(labels[d]) for d in te)}")

    train_ds = FraudSeqDataset(sequences, labels, tr, args.max_seq_len)
    val_ds = FraudSeqDataset(sequences, labels, va, args.max_seq_len, train_ds.cont_mean, train_ds.cont_std)
    test_ds = FraudSeqDataset(sequences, labels, te, args.max_seq_len, train_ds.cont_mean, train_ds.cont_std)
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, collate_fn=collate)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    cat_facets = [(n, CAT_VOCAB[n], CAT_EMB[n]) for n in CAT_NAMES]
    feat_targets = ([{"name": n, "kind": "cat", "col": j, "vocab": CAT_VOCAB[n]} for j, n in enumerate(CAT_NAMES)] +
                    [{"name": n, "kind": "cont", "col": j, "vocab": None}
                     for j, n in enumerate(["value_processed", "delta_t_seconds"])])
    model = CausalEventModel(cat_facets, len(CONT_INDICES), d_model=args.d_model, nhead=args.nhead,
                             num_layers=args.layers, dim_feedforward=args.ffn, max_seq_len=args.max_seq_len)
    model.set_feature_targets(feat_targets); model = model.to(device)

    ckpt_objective = "ar"
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        ck = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
        state = ck.get("model_state_dict", ck)
        ckpt_objective = ck.get("args", {}).get("objective", "ar")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Carregado pre-treino ({ckpt_objective}): missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("Sem checkpoint: scratch.")
    model.attach_classification_head(2); model.to(device)

    # modo de atencao no fine-tune: ar=causal, mem=bidirecional. auto = herda do checkpoint.
    obj = ckpt_objective if args.objective == "auto" else args.objective
    causal = (obj != "mem")
    print(f"[fine-tune] objetivo={obj} -> {'causal (AR)' if causal else 'bidirecional (MEM)'}")

    y_tr = np.array([int(labels[d]) for d in train_ds.ids])
    n = len(y_tr); w = torch.tensor([n / (2 * max(1, (y_tr == c).sum())) for c in (0, 1)],
                                    dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        for b in tqdm(train_loader, desc=f"epoch {ep+1}/{args.epochs}", leave=False):
            opt.zero_grad()
            logits = model.forward_classify(b["cat"].to(device), b["cont"].to(device),
                                            padding_mask=b["pad"].to(device), causal=causal)
            loss = loss_fn(logits, b["labels"].to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        yv, pv = evaluate(model, val_loader, device, causal=causal); auc = roc_auc_score(yv, pv)
        print(f"epoch {ep+1}: val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    yv, pv = evaluate(model, val_loader, device, causal=causal); thr = best_thr(yv, pv)
    yt, pt = evaluate(model, test_loader, device, causal=causal); pred = (pt >= thr).astype(int)
    ar = {"auc": float(roc_auc_score(yt, pt)), "pr_auc": float(average_precision_score(yt, pt)),
          "f1": float(f1_score(yt, pred, zero_division=0)), "recall_at_fpr5": recall_at_fpr(yt, pt),
          "threshold": thr, "val_auc": float(best_auc), "pretrained": bool(args.pretrained_checkpoint)}

    # Predicoes de teste (consumidas por data/xgb_canonical.py, data/aggregate_seeds.py
    # e data/compute_f1_optimal.py). O baseline XGBoost e o canonico (59 features).
    pd.DataFrame({"id": test_ds.ids, "label": yt, "prob": pt}).to_csv(
        os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    import json
    with open(os.path.join(args.output_dir, "fraud_causal_ar_report.json"), "w") as f:
        json.dump({"causal_ar": ar, "n_test": len(yt),
                   "test_positive_rate": float(yt.mean())}, f, indent=2)
    print("\n=== Fraude (teste) ===")
    print("causal-AR:", {k: round(v, 4) if isinstance(v, float) else v for k, v in ar.items()})


if __name__ == "__main__":
    main()
