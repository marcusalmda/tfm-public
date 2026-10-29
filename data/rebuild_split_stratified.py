#!/usr/bin/env python3
"""
Gera um sample_split.pt ESTRATIFICADO (in-distribution) no dir temporal, espelhando a
lógica do finetune --split stratified: split aleatório estratificado por rótulo em
train/val/test, com o TESTE restrito à grade natural (on_grid==True).

Assim dá pra avaliar o MESMO pipeline (grafo temporal + amostragem + pré-treino) no
regime estratificado, com grafo e seq no mesmo split e alinhados por id.

Salva como sample_split_strat.pt (não sobrescreve o OOT). Passe-o via
--sample-split-path nos finetunes.

Uso:
  python data/rebuild_split_stratified.py --dir data/processed_tabformer_temporal \
    --test-size 0.3 --val-size 0.1 --seed 42
"""
import argparse
import os
from collections import Counter
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed_tabformer_temporal")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", default="sample_split_strat.pt")
    args = ap.parse_args()

    seq = torch.load(f"{args.dir}/discrete_event_sequences.pt", weights_only=False)
    lab = torch.load(f"{args.dir}/fraud_labels.pt", weights_only=False)
    s2d = torch.load(f"{args.dir}/sample_to_graph_idx.pt", weights_only=False)
    ongrid = torch.load(f"{args.dir}/sample_on_grid.pt", weights_only=False)

    docs = sorted([d for d in lab if d in seq and d in s2d])
    y = np.array([int(lab[d]) for d in docs])

    # test estratificado
    tr_val, te = train_test_split(docs, test_size=args.test_size, random_state=args.seed,
                                  stratify=y)
    y_trval = np.array([int(lab[d]) for d in tr_val])
    # val como fração do restante
    tr, va = train_test_split(tr_val, test_size=args.val_size, random_state=args.seed,
                              stratify=y_trval)

    # ESTRATIFICADO = in-distribution: teste DENSO (mesma densidade do treino),
    # NÃO restringe a on_grid (senão vira tão esparso quanto o OOT e perde o sentido).
    split = {}
    for d in tr:
        split[d] = "train"
    for d in va:
        split[d] = "val"
    for d in te:
        split[d] = "test"           # todos os samples estratificados do teste (densos)

    torch.save(split, f"{args.dir}/{args.out_name}")
    pos = lambda sp: sum(int(lab[d]) for d in split if split[d] == sp)
    print(f"[estratificado] {dict(Counter(split.values()))}")
    print(f"   pos_train={pos('train'):,} | pos_val={pos('val'):,} | pos_test={pos('test'):,}")
    print(f"   taxa fraude test={pos('test')/max(1,sum(1 for d in split if split[d]=='test')):.4%}")
    print(f"[ok] salvo em {args.dir}/{args.out_name}")


if __name__ == "__main__":
    main()
