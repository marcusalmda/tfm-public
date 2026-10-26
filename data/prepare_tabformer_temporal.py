#!/usr/bin/env python3
"""
Grafo TEMPORAL de fraude (Caminho 2) — replica a receita do churn no TabFormer.

Em vez do grafo estático (Document/Account/Merchant/User), constrói um grafo
BIPARTIDO TEMPORAL:  Document(usuário) --TRANSACTED(t)--> Merchant, uma aresta por
transação, com edge_time. Isso habilita:
  (C) grafo temporal por transação  (recência + frequência, sem dedup)
  (B) amostragem de vizinhança temporal (NeighborLoader) -> 2-hop colaborativo
      (usuários que compartilham comerciantes = sinal de quadrilha), sem vazar futuro
  (A) pré-treino do HGT (depois, via pretrain_tfm_churn_graphsample.py)

AUTOCONTIDO: reconstrói o lado da sequência com as MESMAS funções do prepare_tabformer
(build_discrete_maps/build_windowed), então basta o CSV + os mesmos --window/--stride
usados no experimento original. Gera também sample_split.pt com o split temporal do
finetune (--split tabformer). Saídas em --outdir, no formato que os scripts graphsample leem.

Uso:
  python data/prepare_tabformer_temporal.py \
    --tx    data/public/tabformer_credit_card/credit_card_transactions-ibm_v2.csv \
    --users data/public/tabformer_credit_card/sd254_users.csv \
    --outdir data/processed_tabformer_temporal \
    --window 10 --stride 10 [--nrows N]
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_tabformer as P  # load_transactions, build_discrete_maps, build_windowed, encode_node_features


def build_temporal_graph(tx, users_path):
    print("[graph] construindo grafo BIPARTIDO TEMPORAL (Document--TRANSACTED(t)-->Merchant) ...")
    doc_ids = sorted(tx["document_id"].unique())
    mer_ids = sorted(tx["merchant_id"].dropna().unique())
    doc_idx = {d: i for i, d in enumerate(doc_ids)}
    mer_idx = {m: i for i, m in enumerate(mer_ids)}

    data = HeteroData()
    data["Document"].num_nodes = len(doc_ids)
    data["Merchant"].num_nodes = len(mer_ids)

    # features de Document (intrínsecas do usuário — iguais ao grafo "raso"); Merchant sem feature
    try:
        users = pd.read_csv(users_path, low_memory=False)
        users.columns = [c.strip() for c in users.columns]
        users = users.reset_index().rename(columns={"index": "User"})
        for c in ["Per Capita Income - Zipcode", "Yearly Income - Person", "Total Debt"]:
            if c in users.columns:
                users[c] = P.money_to_float(users[c])
        users["document_id"] = "U" + users["User"].astype(str)
        users = users.set_index("document_id").reindex(doc_ids).reset_index()
        data["Document"].x = torch.tensor(P.encode_node_features(
            users, ["Gender"],
            ["Current Age", "Retirement Age", "Birth Year", "FICO Score", "Num Credit Cards",
             "Per Capita Income - Zipcode", "Yearly Income - Person", "Total Debt", "Latitude", "Longitude"]),
            dtype=torch.float32)
    except Exception as e:
        print(f"[graph] AVISO Document sem features ({e})")
        data["Document"].x = torch.zeros((len(doc_ids), 1), dtype=torch.float32)
    data["Merchant"].x = torch.zeros((len(mer_ids), 1), dtype=torch.float32)  # raso (sem enriquecer)

    # arestas POR TRANSAÇÃO com tempo (sem dedup)
    t = tx.dropna(subset=["merchant_id"]).copy()
    src = t["document_id"].map(doc_idx).to_numpy()
    dst = t["merchant_id"].map(mer_idx).to_numpy()
    etime = (pd.to_datetime(t["timestamp"]).astype("int64") // 10**9).to_numpy()  # epoch s
    ei = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    et = torch.tensor(etime, dtype=torch.long)
    data["Document", "TRANSACTED", "Merchant"].edge_index = ei
    data["Document", "TRANSACTED", "Merchant"].edge_time = et
    data["Merchant", "REV_TRANSACTED", "Document"].edge_index = ei.flip(0).contiguous()
    data["Merchant", "REV_TRANSACTED", "Document"].edge_time = et.clone()

    print(data)
    print(f"[graph] arestas por transação (c/ tempo): {ei.size(1):,}")
    return data, doc_idx


def temporal_split(sample_time, on_grid, test_size, val_size, valid=None):
    """IDÊNTICO ao finetune --split tabformer: ordena por tempo; teste = cauda on_grid;
    off-grid da cauda é DESCARTADO (não vai pro treino). `valid` = conjunto de sids
    elegíveis (com sequência+label+grafo), pra os cortes cronológicos baterem."""
    keys = [s for s in sample_time if (valid is None or s in valid)]
    sids = sorted(keys, key=lambda s: sample_time[s])
    n = len(sids)
    n_test = int(n * test_size)
    n_val = int((n - n_test) * val_size)
    n_train = n - n_test - n_val
    split = {}
    for s in sids[:n_train]:
        split[s] = "train"
    for s in sids[n_train:n_train + n_val]:
        split[s] = "val"
    for s in sids[n_train + n_val:]:
        if on_grid.get(s, True):
            split[s] = "test"
        # off-grid da cauda: DESCARTA (não entra em nenhum split)
    print(f"[split] {dict(Counter(split.values()))}")
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx", required=True)
    ap.add_argument("--users", required=True)
    ap.add_argument("--outdir", default="data/processed_tabformer_temporal")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--min-events", type=int, default=2)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--val-size", type=float, default=0.1)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tx = P.load_transactions(args.tx, nrows=args.nrows, fold_errors=False)
    ev_map, merch_map, mcc_map = P.build_discrete_maps(tx)

    graph, doc_idx = build_temporal_graph(tx, args.users)
    torch.save(graph, os.path.join(args.outdir, "graph_data.pt"))
    torch.save(doc_idx, os.path.join(args.outdir, "doc_id_to_graph_idx.pt"))

    sequences, labels, sample_to_doc, sample_time, on_grid = P.build_windowed(
        tx, args.window, args.stride, args.min_events, ev_map, merch_map, mcc_map)
    sample_to_graph_idx = {sid: doc_idx[d] for sid, d in sample_to_doc.items() if d in doc_idx}

    torch.save(sequences, os.path.join(args.outdir, "discrete_event_sequences.pt"))
    torch.save(labels, os.path.join(args.outdir, "fraud_labels.pt"))
    torch.save(sample_to_graph_idx, os.path.join(args.outdir, "sample_to_graph_idx.pt"))
    torch.save(sample_time, os.path.join(args.outdir, "sample_time.pt"))
    torch.save(on_grid, os.path.join(args.outdir, "sample_on_grid.pt"))

    valid = set(d for d in labels if d in sequences and d in sample_to_graph_idx and d in sample_time)
    split = temporal_split(sample_time, on_grid, args.test_size, args.val_size, valid=valid)
    torch.save(split, os.path.join(args.outdir, "sample_split.pt"))

    print(f"\n[ok] {args.outdir}/ : graph_data.pt (temporal) | discrete_event_sequences.pt | "
          "fraud_labels.pt | sample_to_graph_idx.pt | sample_time.pt | sample_on_grid.pt | sample_split.pt")


if __name__ == "__main__":
    main()
