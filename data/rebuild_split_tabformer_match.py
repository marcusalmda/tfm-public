#!/usr/bin/env python3
"""
Reconstrói sample_split.pt no dir temporal para bater EXATAMENTE com o finetune --split tabformer.

Diferença corrigida: as janelas de fraude off-grid da CAUDA (período de teste) são
DESCARTADAS (como no --split tabformer), e não movidas para o treino. Também filtra
amostras por sequência+label+grafo+tempo, igual ao finetune, pra os cortes cronológicos
baterem. Não mexe no grafo — só regrava sample_split.pt.

Uso:
  python data/rebuild_split_tabformer_match.py \
    --dir data/processed_tabformer_temporal --test-size 0.3 --val-size 0.1
"""
import argparse
import os
from collections import Counter
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed_tabformer_temporal")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--val-size", type=float, default=0.1)
    args = ap.parse_args()

    seq = torch.load(f"{args.dir}/discrete_event_sequences.pt", weights_only=False)
    lab = torch.load(f"{args.dir}/fraud_labels.pt", weights_only=False)
    s2d = torch.load(f"{args.dir}/sample_to_graph_idx.pt", weights_only=False)
    stime = torch.load(f"{args.dir}/sample_time.pt", weights_only=False)
    ongrid = torch.load(f"{args.dir}/sample_on_grid.pt", weights_only=False)

    # MESMO filtro do finetune --split tabformer: label + sequência + grafo + tempo
    docs = [d for d in lab if d in seq and d in s2d and d in stime]
    docs.sort(key=lambda d: stime[d])                    # cronológico
    n = len(docs)
    n_test = int(n * args.test_size)
    n_val = int((n - n_test) * args.val_size)
    n_train = n - n_test - n_val

    split = {}
    for d in docs[:n_train]:
        split[d] = "train"                               # inclui off-grid desta faixa
    for d in docs[n_train:n_train + n_val]:
        split[d] = "val"
    dropped = 0
    for d in docs[n_train + n_val:]:
        if ongrid.get(d, True):
            split[d] = "test"                            # teste = só grade natural
        else:
            dropped += 1                                 # off-grid da cauda: DESCARTA (não vai p/ treino)

    torch.save(split, f"{args.dir}/sample_split.pt")
    pos = lambda sp: sum(int(lab[d]) for d in split if split[d] == sp)
    print(f"[split corrigido] {dict(Counter(split.values()))} | off-grid da cauda descartados={dropped:,}")
    print(f"   pos_train={pos('train'):,} | pos_val={pos('val'):,} | pos_test={pos('test'):,}")
    print(f"[ok] sample_split.pt regravado em {args.dir} (agora idêntico ao --split tabformer)")


if __name__ == "__main__":
    main()
