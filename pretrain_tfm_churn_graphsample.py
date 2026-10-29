#!/usr/bin/env python3
"""
Pré-treino AR do TFM COM grafo via amostragem de vizinhança temporal (rel-hm).

Igual ao pretrain_tfm_ar (objetivo AR: prever o próximo evento), mas o Graph Token
vem de um SUBGRAFO amostrado (NeighborLoader temporal), não do full-batch.
Produz um checkpoint com o HGT JÁ TREINADO, para o finetune_tfm_churn_graphsample
partir dele (em vez de HGT aleatório).

Uso:
  python pretrain_tfm_churn_graphsample.py \
    --dir data/processed_relbench_hm_temporal \
    --config-path config/finetune_fraud_config_1024_cap.yaml \
    --max-seq-len 32 --batch-size 256 --epochs 15 --num-neighbors 15 10 \
    --output-dir checkpoints/pretrain_tfm_graphsample
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from src.models.causal_event_model import CausalEventModel
from src.pretrain import load_config
import finetune_causal_ar_fraud as seqbase
import pretrain_tfm_ar as pt
from finetune_tfm_churn_graphsample import collate_seqs

CAT_INDICES, CONT_INDICES = seqbase.CAT_INDICES, seqbase.CONT_INDICES
KEYS, CAT_VOCAB, CAT_EMB, CAT_NAMES = seqbase.KEYS, seqbase.CAT_VOCAB, seqbase.CAT_EMB, seqbase.CAT_NAMES
FEATURE_TARGETS = pt.FEATURE_TARGETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed_relbench_hm_temporal")
    ap.add_argument("--config-path", default="config/finetune_fraud_config_1024_cap.yaml")
    ap.add_argument("--max-seq-len", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--num-neighbors", type=int, nargs="+", default=[15, 10])
    ap.add_argument("--loss-position-rate", type=float, default=0.15)
    ap.add_argument("--output-dir", default="checkpoints/pretrain_tfm_graphsample")
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint .pt p/ continuar o pré-treino (carrega só os pesos; AdamW reinicia)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = torch.load(f"{args.dir}/graph_data.pt", weights_only=False)
    for et in g.edge_types:
        if hasattr(g[et], "edge_time"):
            g[et].time = g[et].edge_time
    sequences = torch.load(f"{args.dir}/discrete_event_sequences.pt", weights_only=False)
    s2doc = torch.load(f"{args.dir}/sample_to_graph_idx.pt", weights_only=False)
    s2time = torch.load(f"{args.dir}/sample_time.pt", weights_only=False)
    s2split = torch.load(f"{args.dir}/sample_split.pt", weights_only=False)

    # pré-treino usa train+val (sem rótulo; é autossupervisionado)
    ids = [s for s in s2split if s2split[s] in ("train", "val") and s in sequences
           and s in s2doc and len(sequences[s]) >= 2]
    docn = torch.tensor([s2doc[s] for s in ids], dtype=torch.long)
    tim = torch.tensor([int(pd.Timestamp(s2time[s]).timestamp()) for s in ids], dtype=torch.long)
    print(f"[pretrain] amostras: {len(ids):,}")

    rows = []
    for s in ids[:50000]:
        for e in sequences[s][-args.max_seq_len:]:
            rows.append([float(e.get(KEYS[i], 0.0)) for i in CONT_INDICES])
    t = torch.tensor(rows, dtype=torch.float32)
    cont_mean, cont_std = t.mean(0), t.std(0).clamp(min=1e-6)

    loader = NeighborLoader(g, num_neighbors=args.num_neighbors, time_attr="time",
                            input_nodes=("Document", docn), input_time=tim,
                            batch_size=args.batch_size, disjoint=True, shuffle=True)

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
    if args.resume_from and os.path.exists(args.resume_from):
        ck = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        state = ck.get("model_state_dict", ck)
        miss, unexp = model.load_state_dict(state, strict=False)
        print(f"[resume] pesos carregados de {args.resume_from}: missing={len(miss)} unexpected={len(unexp)} "
              f"(AdamW reinicia; warm-start)")
    model = model.to(device)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = float("inf")
    for ep in range(args.epochs):
        model.train(); tot = []
        for batch in tqdm(loader, desc=f"AR+G(sample) ep {ep+1}/{args.epochs}", leave=False):
            batch = batch.to(device)
            bs = batch["Document"].batch_size
            prefix = model.gnn_encoder(batch)["Document"][:bs]
            sids = [ids[i] for i in batch["Document"].input_id.tolist()]
            cat, cont, pad = collate_seqs([sequences[s] for s in sids], args.max_seq_len, cont_mean, cont_std)
            cat, cont, pad = cat.to(device), cont.to(device), pad.to(device)
            opt.zero_grad()
            out, pad_adj, npf = model.encode(cat, cont, padding_mask=pad, causal=True, prefix=prefix)
            pred_hidden = out[:, :-1]
            valid = ~pad
            sel = valid & (torch.rand_like(valid, dtype=torch.float32) < args.loss_position_rate)
            if sel.sum() == 0:
                continue
            preds = model.apply_next_heads(pred_hidden[sel])
            tgt_cat = {t["name"]: cat[:, :, t["col"]][sel] for t in FEATURE_TARGETS if t["kind"] == "cat"}
            tgt_cont = {t["name"]: cont[:, :, t["col"]][sel] for t in FEATURE_TARGETS if t["kind"] == "cont"}
            l_cat = out.new_zeros(()); l_cont = out.new_zeros(())
            for name, tgt in tgt_cat.items():
                l_cat = l_cat + ce(preds[name], tgt)
            for name, tgt in tgt_cont.items():
                l_cont = l_cont + mse(preds[name].squeeze(-1), tgt)
            loss = (l_cat + l_cont) / len(FEATURE_TARGETS)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot.append(float(loss.item()))
        m = float(np.mean(tot)) if tot else float("inf")
        print(f"epoch {ep+1}: loss={m:.4f}")
        if m < best:
            best = m
            torch.save({"model_state_dict": model.state_dict(), "args": {"objective": "ar"}},
                       f"{args.output_dir}/best_model.pt")
    print(f"Pronto. Melhor loss={best:.4f}. Em {args.output_dir}")


if __name__ == "__main__":
    main()
