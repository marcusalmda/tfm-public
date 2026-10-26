#!/usr/bin/env python3
"""TFM + AR: pre-treino com GRAFO + objetivo autorregressivo (ou MEM).

Junta o encoder de grafo (HGT, importado do Eduardo sem modificacao) ao modelo
sequencial causal. O embedding do documento entra como CONDITIONING na posicao 0
da sequencia; sob mascara causal ele so atende a si mesmo, mas os eventos atendem
de volta -> veem grafo + passado. Classificacao downstream pela ultima posicao.

Alinhamento da perda AR COM grafo (posicao 0 = grafo):
  sequencia no encoder: [g, e_0, ..., e_{L-1}]  (L+1 posicoes)
  out[0]=g preve e_0 ; out[k]=e_{k-1} preve e_k
  => pred_hidden = out[:, :-1] (posicoes 0..L-1) preve TODOS os L eventos.
(Sem grafo o alvo seria events[:,1:]; com grafo e events[:, :].)

Uso:
  python pretrain_tfm_ar.py --objective ar \
      --sequences processed/discrete_event_sequences.pt \
      --graph-path processed/hetero_graph_noleak.pt \
      --doc-map-path processed/doc_id_to_graph_idx.pt \
      --config-path config/finetune_fraud_config_1024_cap.yaml \
      --output-dir checkpoints/pretrain_tfm_ar
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.causal_event_model import CausalEventModel
from src.pretrain import load_config
import pretrain_causal_ar as base   # reusa CAT_*, FEATURE_TARGETS, mask tokens

CAT_INDICES = base.CAT_INDICES
CONT_INDICES = base.CONT_INDICES
KEYS = base.KEYS
CAT_VOCAB = base.CAT_VOCAB
CAT_EMB = base.CAT_EMB
CAT_NAMES = base.CAT_NAMES
FEATURE_TARGETS = base.FEATURE_TARGETS
CAT_MASK_TOKEN = base.CAT_MASK_TOKEN
NUM_EVENT_TYPES = base.NUM_EVENT_TYPES


class GraphSeqDataset(Dataset):
    """Como ARSequenceDataset, mas guarda o indice do documento no grafo."""

    def __init__(self, sequences_path, doc_map, max_len, cont_mean=None, cont_std=None):
        self.samples = torch.load(sequences_path, weights_only=False)
        self.doc_map = {str(k): v for k, v in doc_map.items()}
        self.ids = [k for k, v in self.samples.items()
                    if isinstance(v, list) and len(v) >= 2 and str(k) in self.doc_map]
        self.max_len = max_len
        if cont_mean is None:
            rows = []
            for sid in self.ids:
                for e in self.samples[sid][-max_len:]:
                    rows.append([float(e.get(KEYS[i], 0.0)) for i in CONT_INDICES])
            t = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((1, len(CONT_INDICES)))
            self.cont_mean = t.mean(0); self.cont_std = t.std(0).clamp(min=1e-6)
        else:
            self.cont_mean, self.cont_std = cont_mean, cont_std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        events = self.samples[sid][-self.max_len:]
        cat = torch.tensor([[base.clamp(e.get(KEYS[i], 0), CAT_VOCAB[KEYS[i]]) for i in CAT_INDICES]
                            for e in events], dtype=torch.long)
        cont = torch.tensor([[float(e.get(KEYS[i], 0.0)) for i in CONT_INDICES] for e in events],
                            dtype=torch.float32)
        cont = (cont - self.cont_mean) / self.cont_std
        return {"cat": cat, "cont": cont, "doc_idx": int(self.doc_map[str(sid)])}


def collate(batch):
    L = max(b["cat"].size(0) for b in batch)
    nc, nk = batch[0]["cat"].size(1), batch[0]["cont"].size(1)
    cat = torch.zeros(len(batch), L, nc, dtype=torch.long)
    cont = torch.zeros(len(batch), L, nk, dtype=torch.float32)
    pad = torch.ones(len(batch), L, dtype=torch.bool)
    doc = torch.tensor([b["doc_idx"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["cat"].size(0)
        cat[i, :n] = b["cat"]; cont[i, :n] = b["cont"]; pad[i, :n] = False
    return {"cat": cat, "cont": cont, "pad": pad, "doc": doc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="processed/discrete_event_sequences.pt")
    ap.add_argument("--graph-path", default="processed/hetero_graph_noleak.pt")
    ap.add_argument("--doc-map-path", default="processed/doc_id_to_graph_idx.pt")
    ap.add_argument("--config-path", default="config/finetune_fraud_config_1024_cap.yaml")
    ap.add_argument("--output-dir", default="checkpoints/pretrain_tfm_ar")
    ap.add_argument("--objective", choices=["ar", "mem"], default="ar")
    ap.add_argument("--mask-continuous", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--loss-position-rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--graph-refresh-steps", type=int, default=1,
                    help="recalcula o HGT a cada K passos (cache do grafo); 1 = toda vez. "
                         "Use K alto (ex.: 50) para grafos grandes (rel-hm).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | objetivo={args.objective} | grafo={args.graph_path}")

    graph_data = torch.load(args.graph_path, map_location=device, weights_only=False)
    doc_map = torch.load(args.doc_map_path, weights_only=False)
    cfg = load_config(args.config_path)
    gnn_cfg = cfg["gnn_model"]; tcfg = cfg["transformer_model"]
    d_model = tcfg["d_model"]

    ds = GraphSeqDataset(args.sequences, doc_map, args.max_seq_len)
    print(f"Sequencias com grafo: {len(ds)} | cont_mean={ds.cont_mean.tolist()}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)

    gnn_input_dims = {nt: (graph_data[nt].x.size(1) if hasattr(graph_data[nt], "x")
                           and graph_data[nt].x is not None else 0)
                      for nt in graph_data.node_types}
    cat_facets = [(KEYS[i], CAT_VOCAB[KEYS[i]], CAT_EMB[KEYS[i]]) for i in CAT_INDICES]
    model = CausalEventModel(
        cat_facets, len(CONT_INDICES), d_model=d_model, nhead=tcfg["nhead"],
        num_layers=tcfg["num_encoder_layers"], dim_feedforward=tcfg["dim_feedforward"],
        max_seq_len=args.max_seq_len, num_event_types=NUM_EVENT_TYPES,
        gnn_config={**gnn_cfg, "out_channels": d_model},
        gnn_node_types=list(graph_data.node_types),
        gnn_edge_types=list(graph_data.edge_types),
        gnn_input_feature_dims=gnn_input_dims,
    )
    model.set_feature_targets(FEATURE_TARGETS)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(loader))

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ce = nn.CrossEntropyLoss(); mse = nn.MSELoss()

    best, hist = float("inf"), []
    K = max(1, args.graph_refresh_steps)
    for ep in range(args.epochs):
        model.train()
        tot, tcat, tcont = [], [], []
        gcache = None   # cache destacado do embedding de 'Document' entre refreshes
        for step, b in enumerate(tqdm(loader, desc=f"{args.objective.upper()}+G ep {ep+1}/{args.epochs}", leave=False)):
            cat = b["cat"].to(device); cont = b["cont"].to(device)
            pad = b["pad"].to(device); doc = b["doc"].to(device)
            opt.zero_grad()
            if step % K == 0:
                _demb = model.gnn_encoder(graph_data)["Document"]  # recomputa HGT COM grad
                gcache = _demb.detach()
                prefix = _demb[doc]                                # grad flui p/ o HGT
            else:
                prefix = gcache[doc]                               # cache destacado (rápido)

            if args.objective == "ar":
                out, pad_adj, npf = model.encode(cat, cont, padding_mask=pad, causal=True, prefix=prefix)
                pred_hidden = out[:, :-1]                          # (B, L, d) — posicoes 0..L-1
                valid = ~pad                                      # (B, L) alvo = TODOS os eventos
                sel = valid & (torch.rand_like(valid, dtype=torch.float32) < args.loss_position_rate)
                if sel.sum() == 0:
                    continue
                preds = model.apply_next_heads(pred_hidden[sel])
                tgt_cat = {t["name"]: cat[:, :, t["col"]][sel] for t in FEATURE_TARGETS if t["kind"] == "cat"}
                tgt_cont = {t["name"]: cont[:, :, t["col"]][sel] for t in FEATURE_TARGETS if t["kind"] == "cont"}
            else:
                valid = ~pad
                sel = valid & (torch.rand_like(valid, dtype=torch.float32) < args.loss_position_rate)
                if sel.sum() == 0:
                    continue
                cat_in = cat.clone(); cont_in = cont.clone()
                for j, n in enumerate(CAT_NAMES):
                    cat_in[:, :, j][sel] = CAT_MASK_TOKEN[n]
                if args.mask_continuous:
                    for j in range(cont_in.size(-1)):
                        cont_in[:, :, j][sel] = 0.0
                out, pad_adj, npf = model.encode(cat_in, cont_in, padding_mask=pad, causal=False, prefix=prefix)
                event_out = out[:, npf:]                           # remove a posicao do grafo
                preds = model.apply_next_heads(event_out[sel])
                tgt_cat = {t["name"]: cat[:, :, t["col"]][sel] for t in FEATURE_TARGETS if t["kind"] == "cat"}
                tgt_cont = {t["name"]: cont[:, :, t["col"]][sel] for t in FEATURE_TARGETS if t["kind"] == "cont"}

            l_cat = out.new_zeros(()); l_cont = out.new_zeros(())
            for name, tgt in tgt_cat.items():
                l_cat = l_cat + ce(preds[name], tgt)
            for name, tgt in tgt_cont.items():
                l_cont = l_cont + mse(preds[name].squeeze(-1), tgt)
            loss = (l_cat + l_cont) / len(FEATURE_TARGETS)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot.append(float(loss.item())); tcat.append(float(l_cat.item())); tcont.append(float(l_cont.item()))

        m = float(np.mean(tot)); mc = float(np.mean(tcat)); mk = float(np.mean(tcont))
        hist.append({"epoch": ep + 1, "loss": m, "loss_cat": mc, "loss_cont": mk})
        print(f"epoch {ep+1}: loss={m:.4f} (cat={mc:.4f} cont={mk:.4f})")
        ckpt = {"model_state_dict": model.state_dict(), "epoch": ep + 1,
                "cont_mean": ds.cont_mean, "cont_std": ds.cont_std,
                "cat_facets": cat_facets, "args": vars(args)}
        torch.save(ckpt, os.path.join(args.output_dir, "last_model.pt"))
        if m < best:
            best = m
            torch.save(ckpt, os.path.join(args.output_dir, "best_model.pt"))
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"Pronto. Melhor loss={best:.4f}. Em {args.output_dir}")


if __name__ == "__main__":
    main()
