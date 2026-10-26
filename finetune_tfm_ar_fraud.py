#!/usr/bin/env python3
"""TFM+AR fine-tune em FRAUDE (grafo + objetivo autorregressivo).

Substitui o resultado da qualificacao pela versao LIMPA (grafo noleak) com o
objetivo AR simetrico ao MEM. Reusa toda a logica do baseline causal-AR sem
grafo (finetune_causal_ar_fraud.py), mas:
  - constroi o CausalEventModel COM gnn_config (encoder HGT do Eduardo)
  - por batch, computa o embedding do documento (prefixo) via graph_embed
  - o prefixo entra na posicao 0 como conditioning; classifica pela ultima
    posicao valida (convencao GPT) sob mascara causal.

Mesmo split do TFM (estratificado, seed 42) para comparabilidade. Salva
predicoes de teste (id,label,prob) no mesmo formato dos outros scripts. O
baseline tabular e o XGBoost canonico (data/xgb_canonical.py); nao ha XGBoost
inline aqui.

Uso:
  python finetune_tfm_ar_fraud.py \
      --pretrained-checkpoint checkpoints/pretrain_tfm_ar/best_model.pt \
      --graph-path processed/hetero_graph_noleak.pt \
      --doc-map-path processed/doc_id_to_graph_idx.pt \
      --config-path config/finetune_fraud_config_1024_cap.yaml \
      --output-dir results/fraud_tfm_ar
"""

import argparse
import json
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
from src.pretrain import load_config
from src.utils.evaluation import create_doc_id_splits
import finetune_causal_ar_fraud as seqbase

CAT_INDICES = seqbase.CAT_INDICES
CONT_INDICES = seqbase.CONT_INDICES
KEYS = seqbase.KEYS
CAT_VOCAB = seqbase.CAT_VOCAB
CAT_EMB = seqbase.CAT_EMB
CAT_NAMES = seqbase.CAT_NAMES
clamp = seqbase.clamp
best_thr = seqbase.best_thr
recall_at_fpr = seqbase.recall_at_fpr


class FraudGraphDataset(Dataset):
    """Como FraudSeqDataset, mas guarda o indice do documento no grafo."""

    def __init__(self, sequences, labels, doc_ids, doc_map, max_len, cont_mean=None, cont_std=None):
        self.samples = sequences
        self.labels = labels
        self.doc_map = doc_map
        self.ids = [d for d in doc_ids if d in sequences and len(sequences[d]) >= 1 and d in doc_map]
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
        return {"cat": cat, "cont": cont, "doc_idx": int(self.doc_map[d]),
                "label": torch.tensor(int(self.labels[d]), dtype=torch.long)}


def collate(batch):
    L = max(b["cat"].size(0) for b in batch)
    nc, nk = batch[0]["cat"].size(1), batch[0]["cont"].size(1)
    cat = torch.zeros(len(batch), L, nc, dtype=torch.long)
    cont = torch.zeros(len(batch), L, nk, dtype=torch.float32)
    pad = torch.ones(len(batch), L, dtype=torch.bool)
    labels = torch.stack([b["label"] for b in batch])
    doc = torch.tensor([b["doc_idx"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["cat"].size(0)
        cat[i, :n] = b["cat"]; cont[i, :n] = b["cont"]; pad[i, :n] = False
    return {"cat": cat, "cont": cont, "pad": pad, "labels": labels, "doc": doc}


@torch.no_grad()
def evaluate(model, loader, graph_data, device, causal=True, readout="auto"):
    model.eval()
    # grafo é estático no eval: computa o embedding de 'Document' UMA vez e indexa por batch
    doc_emb = model.gnn_encoder(graph_data)["Document"]
    ys, ps = [], []
    for b in loader:
        prefix = doc_emb[b["doc"].to(device)]
        logits = model.forward_classify(b["cat"].to(device), b["cont"].to(device),
                                        padding_mask=b["pad"].to(device), causal=causal,
                                        prefix=prefix, readout=readout)
        ps.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy()); ys.extend(b["labels"].numpy())
    return np.asarray(ys), np.asarray(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-checkpoint", default="checkpoints/pretrain_tfm_ar/best_model.pt")
    ap.add_argument("--sequences", default="processed/discrete_event_sequences.pt")
    ap.add_argument("--labels", default="processed/fraud_labels.pt")
    ap.add_argument("--graph-path", default="processed/hetero_graph_noleak.pt")
    ap.add_argument("--doc-map-path", default="processed/doc_id_to_graph_idx.pt")
    ap.add_argument("--config-path", default="config/finetune_fraud_config_1024_cap.yaml")
    ap.add_argument("--output-dir", default="results/fraud_tfm_ar")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["stratified", "tabformer", "relbench"], default="stratified")
    ap.add_argument("--sample-time-path", default="data/processed_tabformer/sample_time.pt",
                    help="(split tabformer) {sample_id: timestamp ISO} para split temporal")
    ap.add_argument("--sample-on-grid-path", default="data/processed_tabformer/sample_on_grid.pt",
                    help="(split tabformer) {sample_id: bool}; teste usa só on_grid==True (distribuição natural)")
    ap.add_argument("--sample-split-path", default="data/processed_relbench_hm/sample_split.pt",
                    help="(split relbench) {sample_id: train/val/test} — usa o split pronto do RelBench")
    ap.add_argument("--graph-refresh-steps", type=int, default=1,
                    help="recalcula o HGT a cada K passos de treino (cache do grafo); 1 = toda vez (original). "
                         "Use K alto (ex.: 50) para grafos grandes (rel-hm).")
    ap.add_argument("--objective", choices=["auto", "ar", "mem"], default="auto")
    ap.add_argument("--readout", choices=["auto", "last", "mean"], default="auto",
                    help="agregacao p/ classificar: last=ultima posicao (GPT), "
                         "mean=media (robusto a drift), auto=depende do objetivo")
    ap.add_argument("--freeze-gnn", action="store_true",
                    help="congela o encoder de grafo (HGT) no fine-tune; diagnostico de "
                         "overfit temporal do grafo sob OOT")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph_data = torch.load(args.graph_path, map_location=device, weights_only=False)
    doc_map = torch.load(args.doc_map_path, weights_only=False)
    doc_map = {str(k): int(v) for k, v in doc_map.items()}
    cfg = load_config(args.config_path)
    gnn_cfg = cfg["gnn_model"]; tcfg = cfg["transformer_model"]
    d_model = tcfg["d_model"]

    sequences = torch.load(args.sequences, weights_only=False)
    labels = torch.load(args.labels, weights_only=False)
    # intersecao rotulados presentes nas sequencias E no grafo, ORDENADA — igual
    # ao TFM (src/datasets.py usa sorted(...)). Mesmo seed => mesmo split.
    if args.split == "tabformer":
        # Split TEMPORAL por sample_time; TESTE apenas em amostras da grade natural (on_grid).
        stime = torch.load(args.sample_time_path, weights_only=False)
        ongrid = torch.load(args.sample_on_grid_path, weights_only=False)
        docs = sorted([d for d in labels if d in sequences and str(d) in doc_map and d in stime])
        docs.sort(key=lambda d: stime[d])                      # ordena cronologicamente
        n = len(docs)
        n_test = int(n * args.test_size)
        n_val = int((n - n_test) * args.val_size)
        n_train = n - n_test - n_val
        tr = docs[:n_train]
        va = docs[n_train:n_train + n_val]
        te = [d for d in docs[n_train + n_val:] if ongrid.get(d, True)]  # teste = distribuição natural
        print(f"Split TabFormer (temporal): train={len(tr)} val={len(va)} "
              f"test={len(te)} (grade natural) | pos_train={sum(int(labels[d]) for d in tr)} "
              f"pos_test={sum(int(labels[d]) for d in te)}")
    elif args.split == "relbench":
        ssplit = torch.load(args.sample_split_path, weights_only=False)
        docs = sorted([d for d in labels if d in sequences and str(d) in doc_map and d in ssplit])
        tr = [d for d in docs if ssplit[d] == "train"]
        va = [d for d in docs if ssplit[d] == "val"]
        te = [d for d in docs if ssplit[d] == "test"]
        print(f"Split RelBench: train={len(tr)} val={len(va)} test={len(te)} "
              f"| pos_train={sum(int(labels[d]) for d in tr)} pos_test={sum(int(labels[d]) for d in te)}")
    else:
        docs = sorted([d for d in labels if d in sequences and str(d) in doc_map])
        print(f"Docs rotulados com sequencia+grafo: {len(docs)} | positivos: {sum(int(labels[d]) for d in docs)}")
        tr, va, te = create_doc_id_splits(docs, {d: int(labels[d]) for d in docs},
                                          test_size=args.test_size, validation_size=args.val_size,
                                          seed=args.seed, stratify=True)
        # TabFormer: teste na distribuição natural (só on_grid); treino/val mantêm os positivos densos
        if os.path.exists(args.sample_on_grid_path):
            og = torch.load(args.sample_on_grid_path, weights_only=False)
            te = [d for d in te if og.get(d, True)]
            print(f"[stratified] teste restrito à grade natural (on_grid): {len(te)} amostras")
        print(f"Split estratificado: train={len(tr)} val={len(va)} test={len(te)} "
              f"| pos_test={sum(int(labels[d]) for d in te)}")

    dmap = {str(d): doc_map[str(d)] for d in docs}
    train_ds = FraudGraphDataset(sequences, labels, tr, dmap, args.max_seq_len)
    val_ds = FraudGraphDataset(sequences, labels, va, dmap, args.max_seq_len, train_ds.cont_mean, train_ds.cont_std)
    test_ds = FraudGraphDataset(sequences, labels, te, dmap, args.max_seq_len, train_ds.cont_mean, train_ds.cont_std)
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, collate_fn=collate)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    cat_facets = [(n, CAT_VOCAB[n], CAT_EMB[n]) for n in CAT_NAMES]
    feat_targets = ([{"name": n, "kind": "cat", "col": j, "vocab": CAT_VOCAB[n]} for j, n in enumerate(CAT_NAMES)] +
                    [{"name": n, "kind": "cont", "col": j, "vocab": None}
                     for j, n in enumerate(["value_processed", "delta_t_seconds"])])
    gnn_input_dims = {nt: (graph_data[nt].x.size(1) if hasattr(graph_data[nt], "x")
                           and graph_data[nt].x is not None else 0)
                      for nt in graph_data.node_types}
    model = CausalEventModel(
        cat_facets, len(CONT_INDICES), d_model=d_model, nhead=tcfg["nhead"],
        num_layers=tcfg["num_encoder_layers"], dim_feedforward=tcfg["dim_feedforward"],
        max_seq_len=args.max_seq_len,
        gnn_config={**gnn_cfg, "out_channels": d_model},
        gnn_node_types=list(graph_data.node_types),
        gnn_edge_types=list(graph_data.edge_types),
        gnn_input_feature_dims=gnn_input_dims,
    )
    model.set_feature_targets(feat_targets); model = model.to(device)

    ckpt_objective = "ar"
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        ck = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
        state = ck.get("model_state_dict", ck)
        ckpt_objective = ck.get("args", {}).get("objective", "ar")
        missing, unexpected = model.load_state_dict(state, strict=False)
        gnn_missing = [k for k in missing if k.startswith("gnn_encoder")]
        print(f"Carregado TFM+AR ({ckpt_objective}): missing={len(missing)} "
              f"(gnn_missing={len(gnn_missing)}) unexpected={len(unexpected)}")
        assert len(gnn_missing) == 0, "Pesos do HGT NAO foram carregados do pre-treino!"
    else:
        print("Sem checkpoint: scratch.")
    model.attach_classification_head(2); model.to(device)

    if args.freeze_gnn and model.gnn_encoder is not None:
        n_frozen = 0
        for p in model.gnn_encoder.parameters():
            p.requires_grad = False; n_frozen += p.numel()
        print(f"[freeze-gnn] encoder de grafo CONGELADO ({n_frozen} params).")

    obj = ckpt_objective if args.objective == "auto" else args.objective
    causal = (obj != "mem")
    print(f"[fine-tune] objetivo={obj} -> {'causal (AR)' if causal else 'bidirecional (MEM)'}")

    y_tr = np.array([int(labels[d]) for d in train_ds.ids])
    n = len(y_tr); w = torch.tensor([n / (2 * max(1, (y_tr == c).sum())) for c in (0, 1)],
                                    dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc, best_state = -1.0, None
    K = max(1, args.graph_refresh_steps)
    for ep in range(args.epochs):
        model.train()
        gcache = None  # cache destacado do embedding de 'Document' entre refreshes
        for step, b in enumerate(tqdm(train_loader, desc=f"epoch {ep+1}/{args.epochs}", leave=False)):
            opt.zero_grad()
            if step % K == 0:
                doc_emb = model.gnn_encoder(graph_data)["Document"]   # recomputa HGT COM grad
                gcache = doc_emb.detach()
                prefix = doc_emb[b["doc"].to(device)]                 # grad flui p/ o HGT neste passo
            else:
                prefix = gcache[b["doc"].to(device)]                 # cache destacado (rápido, sem grad no HGT)
            logits = model.forward_classify(b["cat"].to(device), b["cont"].to(device),
                                            padding_mask=b["pad"].to(device), causal=causal,
                                            prefix=prefix, readout=args.readout)
            loss = loss_fn(logits, b["labels"].to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        yv, pv = evaluate(model, val_loader, graph_data, device, causal=causal, readout=args.readout); auc = roc_auc_score(yv, pv)
        print(f"epoch {ep+1}: val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    yv, pv = evaluate(model, val_loader, graph_data, device, causal=causal, readout=args.readout); thr = best_thr(yv, pv)
    yt, pt = evaluate(model, test_loader, graph_data, device, causal=causal, readout=args.readout); pred = (pt >= thr).astype(int)
    tfm = {"auc": float(roc_auc_score(yt, pt)), "pr_auc": float(average_precision_score(yt, pt)),
           "f1": float(f1_score(yt, pred, zero_division=0)), "recall_at_fpr5": recall_at_fpr(yt, pt),
           "threshold": thr, "val_auc": float(best_auc), "pretrained": bool(args.pretrained_checkpoint)}

    # Predicoes de teste (id,label,prob) — consumidas por data/xgb_canonical.py,
    # data/aggregate_seeds.py, data/sig_tabformer.py e data/compute_f1_optimal.py.
    # O baseline tabular e o XGBoost canonico (59 features) em data/xgb_canonical.py.
    pd.DataFrame({"id": test_ds.ids, "label": yt, "prob": pt}).to_csv(
        os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    with open(os.path.join(args.output_dir, "fraud_tfm_ar_report.json"), "w") as f:
        json.dump({"tfm_ar": tfm, "n_test": len(yt),
                   "test_positive_rate": float(yt.mean()), "split": args.split}, f, indent=2)
    print("\n=== Fraude TFM+AR (teste) ===")
    print("TFM+AR :", {k: round(v, 4) if isinstance(v, float) else v for k, v in tfm.items()})


if __name__ == "__main__":
    main()
