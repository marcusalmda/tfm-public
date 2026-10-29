#!/usr/bin/env python3
"""
TFM + grafo com AMOSTRAGEM DE VIZINHANÇA TEMPORAL (rel-hm churn).

Mantém o HGT; troca o full-batch por NeighborLoader temporal (edge_time): por batch,
amostra um subgrafo recente (só passado) ao redor dos clientes-semente -> recupera o
sinal colaborativo (2-hop), cabe na GPU e não vaza futuro.

Reusa os vocabulários/helpers do finetune_causal_ar_fraud (seqbase) e o CausalEventModel.
Split vem do RelBench (sample_split.pt). Métrica principal: AUROC.

Uso:
  python finetune_tfm_churn_graphsample.py \
    --dir data/processed_relbench_hm_temporal \
    --pretrained-checkpoint checkpoints/pretrain_causal_ar_hm/best_model.pt \
    --config-path config/finetune_fraud_config_1024_cap.yaml \
    --max-seq-len 32 --batch-size 256 --epochs 8 --num-neighbors 15 10 \
    --output-dir results/churn_tfm_graphsample
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

from src.models.causal_event_model import CausalEventModel
from src.pretrain import load_config
import finetune_causal_ar_fraud as seqbase

CAT_INDICES, CONT_INDICES = seqbase.CAT_INDICES, seqbase.CONT_INDICES
KEYS, CAT_VOCAB, CAT_EMB, CAT_NAMES = seqbase.KEYS, seqbase.CAT_VOCAB, seqbase.CAT_EMB, seqbase.CAT_NAMES
clamp, best_thr, recall_at_fpr = seqbase.clamp, seqbase.best_thr, seqbase.recall_at_fpr


def seq_to_tensors(events, max_len, cont_mean, cont_std):
    ev = events[-max_len:]
    cat = torch.tensor([[clamp(e.get(KEYS[i], 0), CAT_VOCAB[KEYS[i]]) for i in CAT_INDICES] for e in ev], dtype=torch.long)
    cont = torch.tensor([[float(e.get(KEYS[i], 0.0)) for i in CONT_INDICES] for e in ev], dtype=torch.float32)
    cont = (cont - cont_mean) / cont_std
    return cat, cont


def collate_seqs(seq_list, max_len, cont_mean, cont_std):
    items = [seq_to_tensors(s, max_len, cont_mean, cont_std) for s in seq_list]
    L = max(c.size(0) for c, _ in items)
    nc, nk = items[0][0].size(1), items[0][1].size(1)
    cat = torch.zeros(len(items), L, nc, dtype=torch.long)
    cont = torch.zeros(len(items), L, nk, dtype=torch.float32)
    pad = torch.ones(len(items), L, dtype=torch.bool)
    for i, (c, k) in enumerate(items):
        n = c.size(0)
        cat[i, :n] = c; cont[i, :n] = k; pad[i, :n] = False
    return cat, cont, pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed_relbench_hm_temporal")
    ap.add_argument("--labels-name", default="churn_labels.pt",
                    help="nome do arquivo de rótulos em --dir (ex.: fraud_labels.pt)")
    ap.add_argument("--split-name", default="sample_split.pt",
                    help="nome do arquivo de split em --dir (ex.: sample_split_strat.pt)")
    ap.add_argument("--pretrained-checkpoint", default=None)
    ap.add_argument("--config-path", default="config/finetune_fraud_config_1024_cap.yaml")
    ap.add_argument("--max-seq-len", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--num-neighbors", type=int, nargs="+", default=[15, 10])
    ap.add_argument("--readout", default="auto")
    ap.add_argument("--output-dir", default="results/churn_tfm_graphsample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- dados ----
    g = torch.load(f"{args.dir}/graph_data.pt", weights_only=False)
    for et in g.edge_types:                       # expõe edge_time como 'time' p/ o loader
        if hasattr(g[et], "edge_time"):
            g[et].time = g[et].edge_time
    sequences = torch.load(f"{args.dir}/discrete_event_sequences.pt", weights_only=False)
    labels = torch.load(f"{args.dir}/{args.labels_name}", weights_only=False)
    s2doc = torch.load(f"{args.dir}/sample_to_graph_idx.pt", weights_only=False)
    s2time = torch.load(f"{args.dir}/sample_time.pt", weights_only=False)
    s2split = torch.load(f"{args.dir}/{args.split_name}", weights_only=False)

    def arrays(split):
        ids = [s for s in s2split if s2split[s] == split and s in sequences and s in s2doc and len(sequences[s]) >= 1]
        docn = torch.tensor([s2doc[s] for s in ids], dtype=torch.long)
        tim = torch.tensor([int(pd.Timestamp(s2time[s]).timestamp()) for s in ids], dtype=torch.long)
        lab = torch.tensor([int(labels[s]) for s in ids], dtype=torch.long)
        return ids, docn, tim, lab

    tr_ids, tr_doc, tr_tim, tr_lab = arrays("train")
    va_ids, va_doc, va_tim, va_lab = arrays("val")
    te_ids, te_doc, te_tim, te_lab = arrays("test")
    print(f"amostras: train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)} "
          f"| churn train={tr_lab.float().mean():.3f} test={te_lab.float().mean():.3f}")

    # normalização contínua a partir do treino
    rows = []
    for s in tr_ids[:50000]:
        for e in sequences[s][-args.max_seq_len:]:
            rows.append([float(e.get(KEYS[i], 0.0)) for i in CONT_INDICES])
    t = torch.tensor(rows, dtype=torch.float32)
    cont_mean, cont_std = t.mean(0), t.std(0).clamp(min=1e-6)

    def make_loader(docn, tim, shuffle):
        return NeighborLoader(g, num_neighbors=args.num_neighbors, time_attr="time",
                              input_nodes=("Document", docn), input_time=tim,
                              batch_size=args.batch_size, disjoint=True, shuffle=shuffle)

    tr_loader = make_loader(tr_doc, tr_tim, True)
    va_loader = make_loader(va_doc, va_tim, False)
    te_loader = make_loader(te_doc, te_tim, False)

    # ---- modelo (HGT a partir da metadata do grafo) ----
    cfg = load_config(args.config_path); gnn_cfg = cfg["gnn_model"]; tcfg = cfg["transformer_model"]
    d_model = tcfg["d_model"]
    cat_facets = [(n, CAT_VOCAB[n], CAT_EMB[n]) for n in CAT_NAMES]
    feat_targets = ([{"name": n, "kind": "cat", "col": j, "vocab": CAT_VOCAB[n]} for j, n in enumerate(CAT_NAMES)] +
                    [{"name": n, "kind": "cont", "col": j, "vocab": None}
                     for j, n in enumerate(["value_processed", "delta_t_seconds"])])
    gnn_input_dims = {nt: (g[nt].x.size(1) if hasattr(g[nt], "x") and g[nt].x is not None else 0) for nt in g.node_types}
    model = CausalEventModel(cat_facets, len(CONT_INDICES), d_model=d_model, nhead=tcfg["nhead"],
                             num_layers=tcfg["num_encoder_layers"], dim_feedforward=tcfg["dim_feedforward"],
                             max_seq_len=args.max_seq_len, gnn_config={**gnn_cfg, "out_channels": d_model},
                             gnn_node_types=list(g.node_types), gnn_edge_types=list(g.edge_types),
                             gnn_input_feature_dims=gnn_input_dims)
    model.set_feature_targets(feat_targets)
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        ck = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
        state = ck.get("model_state_dict", ck)
        miss, unexp = model.load_state_dict(state, strict=False)
        print(f"checkpoint carregado: missing={len(miss)} unexpected={len(unexp)} (HGT pode iniciar aleatório)")
    model.attach_classification_head(2); model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_batch(batch, ids_ref):
        batch = batch.to(device)
        bs = batch["Document"].batch_size
        prefix = model.gnn_encoder(batch)["Document"][:bs]           # embedding dos clientes-semente
        sids = [ids_ref[i] for i in batch["Document"].input_id.tolist()]
        cat, cont, pad = collate_seqs([sequences[s] for s in sids], args.max_seq_len, cont_mean, cont_std)
        y = torch.tensor([int(labels[s]) for s in sids], dtype=torch.long)
        logits = model.forward_classify(cat.to(device), cont.to(device), padding_mask=pad.to(device),
                                        causal=True, prefix=prefix, readout=args.readout)
        return logits, y, sids

    @torch.no_grad()
    def evaluate(loader, ids_ref):
        model.eval(); ys, ps, sid_order = [], [], []
        for batch in loader:
            logits, y, sids = run_batch(batch, ids_ref)
            ps.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy()); ys.extend(y.numpy()); sid_order.extend(sids)
        return np.asarray(ys), np.asarray(ps), sid_order

    best_auc, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        for batch in tqdm(tr_loader, desc=f"epoch {ep+1}/{args.epochs}", leave=False):
            opt.zero_grad()
            logits, y, _ = run_batch(batch, tr_ids)
            loss = loss_fn(logits, y.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        yv, pv, _ = evaluate(va_loader, va_ids); auc = roc_auc_score(yv, pv)
        print(f"epoch {ep+1}: val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    yv, pv, _ = evaluate(va_loader, va_ids); thr = best_thr(yv, pv)
    yt, pt, tids = evaluate(te_loader, te_ids)
    pred = (pt >= thr).astype(int)
    res = {"auc": float(roc_auc_score(yt, pt)), "pr_auc": float(average_precision_score(yt, pt)),
           "f1": float(f1_score(yt, pred, zero_division=0)), "recall_at_fpr5": recall_at_fpr(yt, pt),
           "threshold": float(thr), "val_auc": float(best_auc)}
    print("\n=== Churn TFM+grafo (neighbor sampling) — teste ===")
    print(res)
    pd.DataFrame({"id": tids, "label": yt, "prob": pt}).to_csv(f"{args.output_dir}/test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
