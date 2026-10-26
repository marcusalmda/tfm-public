"""
prepare_relbench_hm.py — adaptador RelBench rel-hm (user-churn) -> formato TFM.

Gera, em --outdir, os mesmos artefatos que o prepare_tabformer (para reusar
pretrain_tfm_ar / finetune_tfm_ar_fraud / sig_tabformer):
    discrete_event_sequences.pt  {sample_id: [ {evento 8 campos}, ... ]}
    churn_labels.pt              {sample_id: 0/1}
    graph_data.pt                HeteroData (Document=cliente, Article; BOUGHT + REV)
    doc_id_to_graph_idx.pt       {customer_id: idx do nó Document}
    sample_to_graph_idx.pt       {sample_id: idx do nó Document}
    sample_time.pt               {sample_id: timestamp ISO}
    sample_split.pt              {sample_id: "train"/"val"/"test"}

Convenções:
    sample_id = f"{customer_id}@{timestamp}"  (um exemplo = cliente numa data de referência)
    sequência = transações do cliente com t_dat < timestamp (últimas max_seq_len)
    rótulo    = churn (0/1) da tarefa user-churn do RelBench
    split     = train/val do RelBench vira train+val internos; val do RelBench vira TESTE
                (o test do RelBench não tem rótulo público)

Evento (8 campos, ordem dos scripts de fine-tuning):
    event_type_id=sales_channel | value_processed=sign*log1p(price) |
    delta_t_seconds=log1p(Δt s) | counterparty_idx=article (capado) |
    specific_event_idx=product_type (capado) | time_hour=0 | time_weekday | time_day_of_year

Uso (container com relbench instalado e HOME apontando p/ o cache):
    python data/prepare_relbench_hm.py --outdir data/processed_relbench_hm \
        --max-seq-len 32 --max-train-samples 300000 --val-frac 0.1
    (use --max-train-samples menor p/ smoke test)
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

CAP_ARTICLE = 60000     # counterparty_idx
CAP_TYPE = 10000        # specific_event_idx


def encode_node_features(df, cat_cols, num_cols):
    cat_cols = [c for c in cat_cols if c in df.columns and df[c].nunique(dropna=True) > 1]
    num_cols = [c for c in num_cols if c in df.columns and df[c].notna().any()]
    tr = []
    if cat_cols:
        tr.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50), cat_cols))
    if num_cols:
        tr.append(("num", StandardScaler(), num_cols))
    if not tr:
        return np.zeros((len(df), 1), dtype=np.float32)
    w = df.copy()
    for c in num_cols:
        w[c] = pd.to_numeric(w[c], errors="coerce")
        w[c] = w[c].fillna(w[c].median() if w[c].notna().any() else 0.0)
    for c in cat_cols:
        w[c] = w[c].astype(str).fillna("NA")
    return np.asarray(ColumnTransformer(tr).fit_transform(w[cat_cols + num_cols]), dtype=np.float32)


def event_dicts_for_customer(sub):
    """sub: df de 1 cliente já ordenado por data. Retorna (lista de dicts, datas np)."""
    price = sub["price"].fillna(0.0).values.astype(float)
    dt = sub["t_dat"].diff().dt.total_seconds().fillna(0.0).values
    et = sub["evt"].values
    cp = sub["art_idx"].values
    sp = sub["type_idx"].values
    wd = sub["t_dat"].dt.weekday.values
    doy = (sub["t_dat"].dt.dayofyear - 1).values
    vlog = np.sign(price) * np.log1p(np.abs(price))
    dlog = np.log1p(np.clip(dt, 0, None))
    evs = []
    for i in range(len(sub)):
        evs.append({
            "event_type_id": int(et[i]), "value_processed": float(vlog[i]),
            "delta_t_seconds": float(dlog[i]), "counterparty_idx": int(cp[i]),
            "specific_event_idx": int(sp[i]), "time_hour": 0,
            "time_weekday": int(wd[i]), "time_day_of_year": int(doy[i]),
        })
    return evs, sub["t_dat"].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/processed_relbench_hm")
    ap.add_argument("--max-seq-len", type=int, default=32)
    ap.add_argument("--max-train-samples", type=int, default=300000,
                    help="subamostra os exemplos de treino (churn tem 3.8M) p/ caber no tempo de GPU")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fração do train do RelBench p/ val interno")
    ap.add_argument("--min-events", type=int, default=2)
    ap.add_argument("--prune-to-samples", action="store_true",
                    help="mantém no grafo só os clientes que aparecem nas amostras (+ seus artigos) — "
                         "encolhe o grafo p/ o HGT de 2 camadas caber na GPU")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    print("[rb] carregando rel-hm ...")
    db = get_dataset("rel-hm", download=True).get_db()
    tx = db.table_dict["transactions"].df.copy()
    cust = db.table_dict["customer"].df.copy()
    art = db.table_dict["article"].df.copy()
    task = get_task("rel-hm", "user-churn", download=True)
    rb_train = task.get_table("train").df.copy()
    rb_val = task.get_table("val").df.copy()   # será o nosso TESTE (tem rótulo)

    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    tx["customer_id"] = tx["customer_id"].astype(str)
    tx["article_id"] = tx["article_id"].astype(str)
    for d in (rb_train, rb_val):
        d["customer_id"] = d["customer_id"].astype(str)
        d["timestamp"] = pd.to_datetime(d["timestamp"])

    # subamostra o train do RelBench JÁ AQUI (determinístico) para sabermos quais clientes usar
    if args.max_train_samples and len(rb_train) > args.max_train_samples:
        rb_train = rb_train.sample(args.max_train_samples, random_state=args.seed).reset_index(drop=True)

    # poda: restringe o grafo aos clientes que aparecem nas amostras (train subamostrado + val/teste)
    if args.prune_to_samples:
        used = set(rb_train["customer_id"]) | set(rb_val["customer_id"])
        before_c, before_e = tx["customer_id"].nunique(), len(tx)
        tx = tx[tx["customer_id"].isin(used)].copy()
        cust = cust[cust["customer_id"].astype(str).isin(used)].copy()
        print(f"[rb] poda: clientes {before_c:,}->{tx['customer_id'].nunique():,} | "
              f"transações {before_e:,}->{len(tx):,}")

    # ---- vocabs de evento ----
    tx["evt"] = tx["sales_channel_id"].fillna(0).astype(int) + 1            # 1/2 -> 2/3 (0=pad)
    art_rank = tx["article_id"].value_counts()
    art_map = {a: (i + 1 if i + 1 < CAP_ARTICLE else CAP_ARTICLE - 1) for i, a in enumerate(art_rank.index)}
    tx["art_idx"] = tx["article_id"].map(art_map).fillna(0).astype(int)
    art["article_id"] = art["article_id"].astype(str)
    type_col = "product_type_no" if "product_type_no" in art.columns else "product_type_name"
    art2type = art.set_index("article_id")[type_col].astype(str).to_dict()
    tvals = sorted(set(art2type.values()))
    type_map = {t: (i + 1 if i + 1 < CAP_TYPE else CAP_TYPE - 1) for i, t in enumerate(tvals)}
    tx["type_idx"] = tx["article_id"].map(art2type).map(type_map).fillna(0).astype(int)
    print(f"[rb] canais={tx['evt'].nunique()} | artigos={len(art_map):,} | tipos={len(type_map)}")

    # ---- grafo: Document=customer, Article; BOUGHT + REV ----
    tx = tx.sort_values(["customer_id", "t_dat"])
    cust_ids = sorted(tx["customer_id"].unique())
    art_ids = sorted(tx["article_id"].unique())
    cidx = {c: i for i, c in enumerate(cust_ids)}
    aidx = {a: i for i, a in enumerate(art_ids)}
    data = HeteroData()
    data["Document"].num_nodes = len(cust_ids)
    data["Article"].num_nodes = len(art_ids)
    cust["customer_id"] = cust["customer_id"].astype(str)
    cfeat = cust.set_index("customer_id").reindex(cust_ids).reset_index()
    data["Document"].x = torch.tensor(encode_node_features(
        cfeat, ["club_member_status", "fashion_news_frequency", "FN", "Active"], ["age"]), dtype=torch.float32)
    afeat = art.set_index("article_id").reindex(art_ids).reset_index()
    data["Article"].x = torch.tensor(encode_node_features(
        afeat,
        # mais colunas categóricas do artigo -> nó mais rico (encode_node_features ignora as ausentes)
        ["product_group_name", "colour_group_name", type_col,
         "department_name", "index_name", "index_group_name", "garment_group_name",
         "graphical_appearance_name", "perceived_colour_value_name", "perceived_colour_master_name",
         "section_name"],
        []), dtype=torch.float32)
    # arestas POR TRANSAÇÃO (sem dedup) com timestamp -> recupera frequência/recência
    # e permite amostragem de vizinhança TEMPORAL (só passado) no treino.
    bought = tx[["customer_id", "article_id", "t_dat"]].dropna(subset=["customer_id", "article_id", "t_dat"])
    src = bought["customer_id"].map(cidx).to_numpy()
    dst = bought["article_id"].map(aidx).to_numpy()
    etime = (bought["t_dat"].astype("int64") // 10**9).to_numpy()   # epoch (segundos)
    ei = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    et = torch.tensor(etime, dtype=torch.long)
    data["Document", "BOUGHT", "Article"].edge_index = ei
    data["Document", "BOUGHT", "Article"].edge_time = et
    data["Article", "REV_BOUGHT", "Document"].edge_index = ei.flip(0).contiguous()
    data["Article", "REV_BOUGHT", "Document"].edge_time = et
    print(f"[graph] arestas BOUGHT (por transação, c/ tempo): {ei.size(1):,}")
    print(data)

    # ---- eventos por cliente (dicts compartilhados) ----
    print("[rb] construindo eventos por cliente ...")
    cust_events, cust_dates = {}, {}
    for cid, sub in tx.groupby("customer_id", sort=False):
        evs, dates = event_dicts_for_customer(sub)
        cust_events[cid] = evs
        cust_dates[cid] = pd.to_datetime(dates)

    # ---- montar amostras a partir das tabelas de tarefa ----
    def build_samples(df, split, subsample=None):
        rows = df[df["customer_id"].isin(cust_events)].copy()
        if "churn" not in rows.columns:
            return {}, {}, {}, {}, {}
        if subsample and len(rows) > subsample:
            rows = rows.sample(subsample, random_state=args.seed)
        seqs, labs, s2doc, s2time, s2split = {}, {}, {}, {}, {}
        for cid, ts, ch in zip(rows["customer_id"], rows["timestamp"], rows["churn"]):
            dates = cust_dates[cid]
            k = int((dates < ts).sum())           # nº de eventos antes da data de referência
            if k < args.min_events:
                continue
            evs = cust_events[cid][:k][-args.max_seq_len:]
            sid = f"{cid}@{ts.date()}"
            seqs[sid] = evs; labs[sid] = int(ch)
            s2doc[sid] = cidx[cid]; s2time[sid] = ts.isoformat(); s2split[sid] = split
        return seqs, labs, s2doc, s2time, s2split

    # train do RelBench -> train + val interno; val do RelBench -> test
    tr = rb_train
    va_mask = rng.random(len(tr)) < args.val_frac
    seqs, labs, s2doc, s2time, s2split = {}, {}, {}, {}, {}
    for part, split, sub in [(tr[~va_mask], "train", None),   # rb_train já foi subamostrado acima
                             (tr[va_mask], "val", None),
                             (rb_val, "test", None)]:
        a, b, c, d, e = build_samples(part, split, sub)
        seqs.update(a); labs.update(b); s2doc.update(c); s2time.update(d); s2split.update(e)

    from collections import Counter
    cnt = Counter(s2split.values())
    pos = {s: sum(labs[k] for k in labs if s2split[k] == s) for s in ["train", "val", "test"]}
    print(f"[rb] amostras: {dict(cnt)}")
    for s in ["train", "val", "test"]:
        n = cnt.get(s, 0)
        print(f"   {s}: {n:,} | churn%={pos[s]/max(n,1):.2%}")

    torch.save(seqs, f"{args.outdir}/discrete_event_sequences.pt")
    torch.save(labs, f"{args.outdir}/churn_labels.pt")
    torch.save(data, f"{args.outdir}/graph_data.pt")
    torch.save(cidx, f"{args.outdir}/doc_id_to_graph_idx.pt")
    torch.save(s2doc, f"{args.outdir}/sample_to_graph_idx.pt")
    torch.save(s2time, f"{args.outdir}/sample_time.pt")
    torch.save(s2split, f"{args.outdir}/sample_split.pt")
    print(f"\n[ok] artefatos em {args.outdir}/ (sequências, churn_labels, graph, doc/sample maps, time, split)")


if __name__ == "__main__":
    main()
