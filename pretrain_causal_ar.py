#!/usr/bin/env python3
"""Passo A — pre-treino autorregressivo (next-event) do modelo causal, sem grafo.

Objetivo: a partir de cada posicao, prever o TIPO do proximo evento (cross-entropy)
e o intervalo delta_t ate ele (MSE sobre valor normalizado). Baseline estilo nuFormer.

Roda sobre a subamostra (com charges), alinhando pre-treino e fine-tune:
  processed/discrete_event_sequences.pt

Uso:
  python pretrain_causal_ar.py --sequences processed/discrete_event_sequences.pt \
      --output-dir checkpoints/pretrain_causal_ar --epochs 300
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

# Ordem das features na sequencia discreta (mesma do pipeline)
CAT_INDICES = [0, 3, 4, 5, 6, 7]   # event_type_id, counterparty, specific, hour, weekday, dayofyear
CONT_INDICES = [1, 2]              # value_processed, delta_t_seconds
KEYS = ["event_type_id", "value_processed", "delta_t_seconds", "counterparty_idx",
        "specific_event_idx", "time_hour", "time_weekday", "time_day_of_year"]
CAT_VOCAB = {"event_type_id": 20, "counterparty_idx": 60000, "specific_event_idx": 10000,
             "time_hour": 25, "time_weekday": 8, "time_day_of_year": 367}
CAT_EMB = {"event_type_id": 32, "counterparty_idx": 96, "specific_event_idx": 32,
           "time_hour": 8, "time_weekday": 8, "time_day_of_year": 16}
CAT_NAMES = ["event_type_id", "counterparty_idx", "specific_event_idx",
             "time_hour", "time_weekday", "time_day_of_year"]
CONT_NAMES = ["value_processed", "delta_t_seconds"]
NUM_EVENT_TYPES = 20
# Mask token de cada facet categorico = ultimo indice do vocab (vocab-1),
# exatamente como no MEM do Eduardo (feature_info: mask_token = vocab_size-1).
CAT_MASK_TOKEN = {n: CAT_VOCAB[n] - 1 for n in CAT_NAMES}

# Alvos do objetivo autorregressivo de evento completo (simetrico ao MEM):
# 6 atributos categoricos (CE) + 2 continuos (MSE). col = posicao no tensor.
FEATURE_TARGETS = (
    [{"name": n, "kind": "cat", "col": j, "vocab": CAT_VOCAB[n]} for j, n in enumerate(CAT_NAMES)] +
    [{"name": n, "kind": "cont", "col": j, "vocab": None} for j, n in enumerate(CONT_NAMES)]
)


def clamp(v, vocab):
    return max(0, min(int(v), vocab - 1))


class ARSequenceDataset(Dataset):
    def __init__(self, sequences_path, max_len, cont_mean=None, cont_std=None):
        self.samples = torch.load(sequences_path, weights_only=False)
        self.ids = [k for k, v in self.samples.items() if isinstance(v, list) and len(v) >= 2]
        self.max_len = max_len
        if cont_mean is None:
            rows = []
            for sid in self.ids:
                for e in self.samples[sid][-max_len:]:
                    rows.append([float(e.get(KEYS[i], 0.0)) for i in CONT_INDICES])
            t = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((1, len(CONT_INDICES)))
            self.cont_mean = t.mean(0)
            self.cont_std = t.std(0).clamp(min=1e-6)
        else:
            self.cont_mean, self.cont_std = cont_mean, cont_std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        events = self.samples[self.ids[idx]][-self.max_len:]
        cat, cont = [], []
        for e in events:
            cat.append([clamp(e.get(KEYS[i], 0), CAT_VOCAB[KEYS[i]]) for i in CAT_INDICES])
            cont.append([float(e.get(KEYS[i], 0.0)) for i in CONT_INDICES])
        cat = torch.tensor(cat, dtype=torch.long)
        cont = (torch.tensor(cont, dtype=torch.float32) - self.cont_mean) / self.cont_std
        return {"cat": cat, "cont": cont}


def collate(batch):
    L = max(b["cat"].size(0) for b in batch)
    nc, nk = batch[0]["cat"].size(1), batch[0]["cont"].size(1)
    cat = torch.zeros(len(batch), L, nc, dtype=torch.long)
    cont = torch.zeros(len(batch), L, nk, dtype=torch.float32)
    pad = torch.ones(len(batch), L, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["cat"].size(0)
        cat[i, :n] = b["cat"]; cont[i, :n] = b["cont"]
        pad[i, :n] = False
    return {"cat": cat, "cont": cont, "pad": pad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="processed/discrete_event_sequences.pt")
    ap.add_argument("--output-dir", default="checkpoints/pretrain_causal_ar")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)   # alinhado ao pre-treino da qualificacao
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--lambda-dt", type=float, default=1.0,
                    help="[DEPRECADO] mantido por compat; a perda usa media por atributo (paridade MEM)")
    ap.add_argument("--loss-position-rate", type=float, default=0.15,
                    help="fracao de posicoes na perda: AR=subamostra p/ casar o MEM; MEM=prob de mascaramento")
    ap.add_argument("--objective", choices=["ar", "mem"], default="ar",
                    help="ar = autorregressivo causal (proximo evento); mem = mascarado bidirecional (Eduardo)")
    ap.add_argument("--mask-continuous", action="store_true",
                    help="MEM: zera contínuos nas posicoes mascaradas (canonical usa True)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = ARSequenceDataset(args.sequences, args.max_seq_len)
    print(f"Sequencias: {len(ds)} | cont_mean={ds.cont_mean.tolist()} cont_std={ds.cont_std.tolist()}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)

    cat_facets = [(KEYS[i], CAT_VOCAB[KEYS[i]], CAT_EMB[KEYS[i]]) for i in CAT_INDICES]
    model = CausalEventModel(cat_facets, len(CONT_INDICES), d_model=args.d_model, nhead=args.nhead,
                             num_layers=args.layers, dim_feedforward=args.ffn,
                             max_seq_len=args.max_seq_len, num_event_types=NUM_EVENT_TYPES)
    model.set_feature_targets(FEATURE_TARGETS)  # 6 cabecas CE + 2 MSE (simetrico ao MEM)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(loader))

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * prog))  # cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    best = float("inf")
    hist = []
    for ep in range(args.epochs):
        model.train()
        tot, tot_cat, tot_cont = [], [], []
        for b in tqdm(loader, desc=f"{args.objective.upper()} epoch {ep+1}/{args.epochs}", leave=False):
            cat = b["cat"].to(device); cont = b["cont"].to(device); pad = b["pad"].to(device)
            opt.zero_grad()

            if args.objective == "ar":
                # ---- AUTORREGRESSIVO (causal): estado k preve evento k+1 ----
                hidden, _, _ = model.encode(cat, cont, padding_mask=pad, causal=True)
                pred_hidden = hidden[:, :-1]
                valid = ~pad[:, 1:]
                sub = torch.rand_like(valid, dtype=torch.float32) < args.loss_position_rate
                sel = valid & sub
                if sel.sum() == 0:
                    continue
                preds = model.apply_next_heads(pred_hidden[sel])
                tgt_cat = {t["name"]: cat[:, 1:, t["col"]][sel] for t in FEATURE_TARGETS if t["kind"] == "cat"}
                tgt_cont = {t["name"]: cont[:, 1:, t["col"]][sel] for t in FEATURE_TARGETS if t["kind"] == "cont"}
            else:
                # ---- MEM (bidirecional): mascara 15% dos eventos e reconstroi ----
                # (fiel ao Eduardo: categoricos -> mask_token; continuos -> zerados
                #  se --mask-continuous; label = valor original; perda so nos mascarados)
                valid = ~pad
                sel = valid & (torch.rand_like(valid, dtype=torch.float32) < args.loss_position_rate)
                if sel.sum() == 0:
                    continue
                cat_in = cat.clone(); cont_in = cont.clone()
                for j, n in enumerate(CAT_NAMES):
                    col = cat_in[:, :, j]
                    col[sel] = CAT_MASK_TOKEN[n]                  # substitui pelo mask token do facet
                if args.mask_continuous:
                    for j in range(cont_in.size(-1)):
                        cont_in[:, :, j][sel] = 0.0
                hidden, _, _ = model.encode(cat_in, cont_in, padding_mask=pad, causal=False)
                preds = model.apply_next_heads(hidden[sel])       # reconstroi posicoes mascaradas
                tgt_cat = {t["name"]: cat[:, :, t["col"]][sel] for t in FEATURE_TARGETS if t["kind"] == "cat"}
                tgt_cont = {t["name"]: cont[:, :, t["col"]][sel] for t in FEATURE_TARGETS if t["kind"] == "cont"}

            # Perda comum aos dois: CE (categoricos) + MSE (continuos), media por atributo
            l_cat = hidden.new_zeros(())
            l_cont = hidden.new_zeros(())
            for name, tgt in tgt_cat.items():
                l_cat = l_cat + ce(preds[name], tgt)
            for name, tgt in tgt_cont.items():
                l_cont = l_cont + mse(preds[name].squeeze(-1), tgt)
            loss = (l_cat + l_cont) / len(FEATURE_TARGETS)        # media, como no MEM
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot.append(float(loss.item())); tot_cat.append(float(l_cat.item())); tot_cont.append(float(l_cont.item()))
        m = float(np.mean(tot)); mc = float(np.mean(tot_cat)); mk = float(np.mean(tot_cont))
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
    print(f"Pronto. Melhor loss={best:.4f}. Checkpoints em {args.output_dir}")


if __name__ == "__main__":
    main()
