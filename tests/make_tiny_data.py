#!/usr/bin/env python3
"""Generate a tiny SYNTHETIC dataset in the exact tensor format the pipeline
expects, so the whole repo can be smoke-tested without the real TabFormer /
rel-hm data. NOT real data — only for verifying the code runs end to end.

Writes to tests/tiny_data/:
  discrete_event_sequences.pt  {id: [ {8 event fields}, ... ]}
  fraud_labels.pt              {id: 0/1}
  graph_data.pt                HeteroData (Document + Counterparty, both directions)
  sample_to_graph_idx.pt       {id: Document node idx}   (= doc_map / Graph Token)
  doc_id_to_graph_idx.pt       {id: Document node idx}
  sample_time.pt               {id: ISO timestamp}       (temporal split)
  sample_on_grid.pt            {id: True}
  sample_split.pt              {id: train/val/test}
  sample_split_strat.pt        copy of sample_split (xgb_canonical --split-name)
"""
import datetime
import os
import random

import torch
from torch_geometric.data import HeteroData

random.seed(0)
torch.manual_seed(0)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_data")
os.makedirs(OUT, exist_ok=True)

N = 60       # samples (= Document nodes)
NCP = 15     # counterparty nodes
base = datetime.datetime(2020, 1, 1)

seqs, labels, split, stime, ongrid = {}, {}, {}, {}, {}
for i in range(N):
    sid = str(i)
    L = random.randint(2, 8)
    seqs[sid] = [{
        "event_type_id": random.randint(0, 19),
        "value_processed": round(random.uniform(-5, 5), 3),
        "delta_t_seconds": round(random.uniform(0, 1e5), 1),
        "counterparty_idx": random.randint(0, NCP - 1),
        "specific_event_idx": random.randint(0, 40),
        "time_hour": random.randint(0, 23),
        "time_weekday": random.randint(0, 6),
        "time_day_of_year": random.randint(0, 365),
    } for _ in range(L)]
    stime[sid] = (base + datetime.timedelta(days=i)).isoformat()
    ongrid[sid] = True

# split by index (60/15/25) and assign labels alternating WITHIN each split,
# so every split has both classes (AUROC/F1 need it).
order = [str(i) for i in range(N)]
n_tr, n_va = int(N * 0.6), int(N * 0.15)
groups = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
for name, ids in groups.items():
    for j, sid in enumerate(ids):
        split[sid] = name
        labels[sid] = j % 2            # guarantees both classes per split

doc_map = {str(i): i for i in range(N)}

g = HeteroData()
g["Document"].x = torch.randn(N, 4)
g["Counterparty"].x = torch.randn(NCP, 3)
src, dst = [], []
for i in range(N):
    for _ in range(random.randint(1, 3)):
        src.append(i); dst.append(random.randint(0, NCP - 1))
ei = torch.tensor([src, dst], dtype=torch.long)
g["Document", "to", "Counterparty"].edge_index = ei
g["Counterparty", "to", "Document"].edge_index = ei.flip(0)

torch.save(seqs, f"{OUT}/discrete_event_sequences.pt")
torch.save(labels, f"{OUT}/fraud_labels.pt")
torch.save(g, f"{OUT}/graph_data.pt")
torch.save(doc_map, f"{OUT}/sample_to_graph_idx.pt")
torch.save(doc_map, f"{OUT}/doc_id_to_graph_idx.pt")
torch.save(stime, f"{OUT}/sample_time.pt")
torch.save(ongrid, f"{OUT}/sample_on_grid.pt")
torch.save(split, f"{OUT}/sample_split.pt")
torch.save(split, f"{OUT}/sample_split_strat.pt")

pos = sum(labels.values())
print(f"[tiny] {N} samples ({pos} pos / {N - pos} neg), {NCP} counterparties -> {OUT}")
