"""
prepare_tabformer.py
====================
Adaptador do dataset IBM TabFormer (cartão de crédito) para o formato consumido
pelos scripts de fine-tuning do TFM (finetune_causal_ar_fraud.py / finetune_tfm_ar_fraud.py).

Lê os 3 CSVs do espelho Kaggle `ealtman2019/credit-card-transactions`:
    - credit_card_transactions-ibm_v2.csv   (transações)
    - sd254_users.csv                        (atributos do usuário  -> nó Document)
    - sd254_cards.csv                        (atributos do cartão   -> nó Account)

Produz, em --outdir:
    - discrete_event_sequences.pt   {id: [ {event dict, 8 campos}, ... ]}   (torch.save)
    - fraud_labels.pt               {id: 0/1}                                (torch.save)
    - graph_data.pt                 HeteroData (Document/Account/Merchant/User)
    - doc_id_to_graph_idx.pt        {document_id: idx do nó 'Document'}
    (modo windowed, adicionalmente:)
    - sample_to_graph_idx.pt        {sample_id: idx do nó 'Document'}  (== doc_map p/ o Graph Token)
    - sample_time.pt                {sample_id: timestamp ISO}         (p/ split temporal)
    - sample_on_grid.pt             {sample_id: bool}   True=grade natural (TESTE); False=fraude extra (só TREINO)

Formato de cada evento (8 campos, ordem = KEYS dos scripts de fine-tuning):
    event_type_id, value_processed, delta_t_seconds, counterparty_idx,
    specific_event_idx, time_hour, time_weekday, time_day_of_year
onde:
    value_processed = sign(Amount)*log1p(|Amount|)
    delta_t_seconds = log1p(Δt em segundos, dentro da sequência)
    counterparty_idx = id do comerciante (0=N/A), limitado a < CAP_COUNTERPARTY
    specific_event_idx = índice do MCC (0=N/A), limitado a < CAP_SPECIFIC
    time_day_of_year = dia do ano - 1 (0-based)

Mapeamento p/ o grafo (nomes de tipo de nó batem com o que o modelo espera):
    Document = usuário (nó lido pelo graph_embed / Graph Token)
    Account  = cartão
    Merchant = comerciante
    User     = nó de login sintético

Uso (docker csitf210:full):
    python data/prepare_tabformer.py \
        --tx data/public/tabformer_credit_card/credit_card_transactions-ibm_v2.csv \
        --users data/public/tabformer_credit_card/sd254_users.csv \
        --cards data/public/tabformer_credit_card/sd254_cards.csv \
        --outdir data/processed_tabformer --task windowed --window 10 --stride 10
    (use --nrows 3000000 para teste rápido)
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# limites de vocabulário categórico usados pelos scripts de fine-tuning (CAT_VOCAB)
CAP_COUNTERPARTY = 60000   # counterparty_idx  (comerciante)
CAP_SPECIFIC = 10000       # specific_event_idx (MCC)


# --------------------------------------------------------------------------- #
def money_to_float(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
            .replace({"": np.nan, "nan": np.nan}).astype(float))


def yesno_to_int(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower() == "yes").astype(int)


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    ts = (df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-"
          + df["Day"].astype(str).str.zfill(2) + " " + df["Time"].astype(str))
    return pd.to_datetime(ts, errors="coerce")


def encode_node_features(df, cat_cols, num_cols) -> np.ndarray:
    cat_cols = [c for c in cat_cols if c in df.columns and df[c].nunique(dropna=True) > 1]
    num_cols = [c for c in num_cols if c in df.columns and df[c].notna().any()]
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if not transformers:
        return np.zeros((len(df), 1), dtype=np.float32)
    work = df.copy()
    for c in num_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
        work[c] = work[c].fillna(work[c].median() if work[c].notna().any() else 0.0)
    for c in cat_cols:
        work[c] = work[c].astype(str).fillna("Unknown")
    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    return np.asarray(ct.fit_transform(work[cat_cols + num_cols]), dtype=np.float32)


# --------------------------------------------------------------------------- #
def load_transactions(path: str, nrows=None, fold_errors=False) -> pd.DataFrame:
    print(f"[tx] lendo {path} ...")
    usecols = ["User", "Card", "Year", "Month", "Day", "Time", "Amount",
               "Use Chip", "Merchant Name", "Merchant City", "Merchant State", "Zip",
               "MCC", "Errors?", "Is Fraud?"]
    try:
        df = pd.read_csv(path, nrows=nrows, usecols=lambda c: c.strip() in usecols, low_memory=False)
    except Exception as e:
        print(f"[tx] leitura direta falhou ({e}); tentando em chunks ...")
        parts, got = [], 0
        for ch in pd.read_csv(path, usecols=lambda c: c.strip() in usecols,
                              chunksize=1_000_000, low_memory=False):
            parts.append(ch)
            got += len(ch)
            if nrows and got >= nrows:
                break
        df = pd.concat(parts, ignore_index=True)
        if nrows:
            df = df.iloc[:nrows]
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = build_timestamp(df)
    df["amount"] = money_to_float(df["Amount"])
    df["is_fraud"] = yesno_to_int(df["Is Fraud?"])
    df["document_id"] = "U" + df["User"].astype(str)                        # entidade (nó Document)
    df["account_id"] = "U" + df["User"].astype(str) + "_C" + df["Card"].astype(str)
    df["login_user_id"] = "login_U" + df["User"].astype(str)               # nó User sintético
    df["merchant_id"] = "M::" + df["Merchant Name"].astype(str).str.strip()
    df["use_chip"] = df["Use Chip"].astype(str).str.strip()
    df["mcc"] = df["MCC"].astype(str).str.strip()
    # Errors?: vazio = sem erro; qualquer texto = houve erro técnico na transação
    err = df["Errors?"].astype(str).str.strip() if "Errors?" in df.columns else pd.Series([""] * len(df))
    df["has_error"] = (~err.isin(["", "nan", "None"])).astype(int)
    # OPCIONAL: dobrar o sinal de erro no event_type. Desligado por padrão —
    # as classes de erro são raras e degradaram o pré-treino/transfer (ver análise).
    if fold_errors:
        df["event_type_str"] = df["use_chip"] + np.where(df["has_error"] == 1, "|ERR", "")
    else:
        df["event_type_str"] = df["use_chip"]
    # localização (sinal espacial -> vira nó de grafo)
    city = df["Merchant City"].astype(str).str.strip() if "Merchant City" in df.columns else pd.Series(["NA"] * len(df))
    state = df["Merchant State"].astype(str).str.strip() if "Merchant State" in df.columns else pd.Series([""] * len(df))
    df["city_id"] = "LOC::" + city + "/" + state
    df = df.dropna(subset=["timestamp"])
    print(f"[tx] {len(df):,} transações | {df['document_id'].nunique():,} usuários | "
          f"{df['account_id'].nunique():,} cartões | fraude={df['is_fraud'].mean():.4%} | "
          f"erros={df['has_error'].mean():.4%} | locais={df['city_id'].nunique():,}")
    return df


# --------------------------------------------------------------------------- #
def build_discrete_maps(tx: pd.DataFrame):
    """Vocabulários numéricos: event_type (com flag de erro), comerciante (capado) e MCC (capado)."""
    ev = {e: i for i, e in enumerate(sorted(tx["event_type_str"].dropna().unique()), start=1)}

    # comerciante por frequência: 1..CAP-1 (mais raros colapsam no último id)
    freq = tx["merchant_id"].value_counts()
    merch = {}
    for rank, m in enumerate(freq.index, start=1):
        merch[m] = rank if rank < CAP_COUNTERPARTY else (CAP_COUNTERPARTY - 1)

    mcc_vals = sorted(tx["mcc"].dropna().unique())
    mcc = {m: (i if i < CAP_SPECIFIC else CAP_SPECIFIC - 1) for i, m in enumerate(mcc_vals, start=1)}
    print(f"[map] event_types={len(ev)} | comerciantes={len(merch)} (cap {CAP_COUNTERPARTY}) | mcc={len(mcc)} (cap {CAP_SPECIFIC})")
    return ev, merch, mcc


def per_card_event_dicts(g: pd.DataFrame, ev_map, merch_map, mcc_map):
    """Lista de dicts de evento (8 campos) para um cartão, ordenada no tempo."""
    g = g.sort_values("timestamp")
    ts = g["timestamp"]
    amount = g["amount"].fillna(0.0).values
    dt = ts.diff().dt.total_seconds().fillna(0.0).values
    et = g["event_type_str"].map(lambda x: ev_map.get(x, 0)).values
    cp = g["merchant_id"].map(lambda x: merch_map.get(x, 0)).values
    sp = g["mcc"].map(lambda x: mcc_map.get(x, 0)).values
    hour = ts.dt.hour.values
    wday = ts.dt.weekday.values
    doy = (ts.dt.dayofyear - 1).values
    val_proc = np.sign(amount) * np.log1p(np.abs(amount))
    dt_proc = np.log1p(np.clip(dt, 0, None))
    events = []
    for i in range(len(g)):
        events.append({
            "event_type_id": int(et[i]),
            "value_processed": float(val_proc[i]),
            "delta_t_seconds": float(dt_proc[i]),
            "counterparty_idx": int(cp[i]),
            "specific_event_idx": int(sp[i]),
            "time_hour": int(hour[i]),
            "time_weekday": int(wday[i]),
            "time_day_of_year": int(doy[i]),
        })
    return events, g["is_fraud"].values.astype(int), ts.values


# --------------------------------------------------------------------------- #
def build_windowed(tx, window, stride, min_events, ev_map, merch_map, mcc_map):
    print(f"[seq] modo=windowed — janela={window}, stride={stride}, min_events={min_events} ...")
    acc_to_doc = tx.drop_duplicates("account_id").set_index("account_id")["document_id"].to_dict()
    sequences, labels, sample_to_doc, sample_time, on_grid = {}, {}, {}, {}, {}
    n_grid_pos = n_grid = n_extra_pos = 0

    for acc_id, g in tx.groupby("account_id", sort=False):
        events, y, ts = per_card_event_dicts(g, ev_map, merch_map, mcc_map)
        n = len(events)
        grid = set(range(n - 1, min_events - 2, -stride))

        def add(i, grid_flag):
            if i < 0 or i + 1 < min_events:
                return None
            sid = f"{acc_id}#{i}"
            if sid in sequences:
                return None
            start = max(0, i - window + 1)
            sequences[sid] = events[start:i + 1]              # slice compartilha os dicts (economia de memória)
            labels[sid] = int(y[i])
            sample_to_doc[sid] = acc_to_doc[acc_id]
            sample_time[sid] = pd.Timestamp(ts[i]).isoformat()
            on_grid[sid] = bool(grid_flag)
            return sid

        for i in grid:
            if add(i, True) is not None:
                n_grid += 1
                if y[i] == 1:
                    n_grid_pos += 1
        for i in np.where(y == 1)[0]:
            if int(i) not in grid and add(int(i), False) is not None:
                n_extra_pos += 1

    total_pos = int(sum(labels.values()))
    print(f"[seq] {len(sequences):,} janelas | grade={n_grid:,} "
          f"(pos_grade={n_grid_pos:,} = {n_grid_pos/max(n_grid,1):.4%}, natural p/ TESTE) "
          f"| +{n_extra_pos:,} fraudes fora da grade (só TREINO) | pos_total={total_pos:,}")
    return sequences, labels, sample_to_doc, sample_time, on_grid


def build_entity(tx, max_seq_len, ev_map, merch_map, mcc_map):
    print(f"[seq] modo=entity — uma sequência por usuário (max_len={max_seq_len}) ...")
    sequences, labels = {}, {}
    for doc_id, g in tx.groupby("document_id", sort=False):
        events, y, _ = per_card_event_dicts(g, ev_map, merch_map, mcc_map)  # ordena por tempo
        sequences[doc_id] = events[-max_seq_len:]
        labels[doc_id] = int(y.max())   # 1 se qualquer transação for fraude (degenerado no TabFormer)
    print(f"[seq] {len(sequences):,} sequências | pos={int(sum(labels.values())):,}")
    return sequences, labels


# --------------------------------------------------------------------------- #
def build_graph(tx, users_path, cards_path):
    print("[graph] construindo HeteroData ...")
    data = HeteroData()
    doc_ids = sorted(tx["document_id"].unique())
    acc_ids = sorted(tx["account_id"].unique())
    mer_ids = sorted(tx["merchant_id"].dropna().unique())
    usr_ids = sorted(tx["login_user_id"].unique())
    loc_ids = sorted(tx["city_id"].dropna().unique()) if "city_id" in tx.columns else []
    doc_idx = {d: i for i, d in enumerate(doc_ids)}
    acc_idx = {a: i for i, a in enumerate(acc_ids)}
    mer_idx = {m: i for i, m in enumerate(mer_ids)}
    usr_idx = {u: i for i, u in enumerate(usr_ids)}
    loc_idx = {c: i for i, c in enumerate(loc_ids)}

    data["Document"].num_nodes = len(doc_ids)
    data["Account"].num_nodes = len(acc_ids)
    data["Merchant"].num_nodes = len(mer_ids)
    data["User"].num_nodes = len(usr_ids)
    if loc_ids:
        data["Location"].num_nodes = len(loc_ids)

    # features de Document (sd254_users)
    try:
        users = pd.read_csv(users_path, low_memory=False)
        users.columns = [c.strip() for c in users.columns]
        users = users.reset_index().rename(columns={"index": "User"})
        for c in ["Per Capita Income - Zipcode", "Yearly Income - Person", "Total Debt"]:
            if c in users.columns:
                users[c] = money_to_float(users[c])
        users["document_id"] = "U" + users["User"].astype(str)
        users = users.set_index("document_id").reindex(doc_ids).reset_index()
        data["Document"].x = torch.tensor(encode_node_features(
            users, ["Gender"],
            ["Current Age", "Retirement Age", "Birth Year", "FICO Score", "Num Credit Cards",
             "Per Capita Income - Zipcode", "Yearly Income - Person", "Total Debt", "Latitude", "Longitude"]),
            dtype=torch.float32)
    except Exception as e:
        print(f"[graph] AVISO Document sem features ({e})")
        data["Document"].x = torch.zeros((len(doc_ids), 1), dtype=torch.float32)

    # features de Account (sd254_cards)
    try:
        cards = pd.read_csv(cards_path, low_memory=False)
        cards.columns = [c.strip() for c in cards.columns]
        if "Credit Limit" in cards.columns:
            cards["Credit Limit"] = money_to_float(cards["Credit Limit"])
        cidx = "CARD INDEX" if "CARD INDEX" in cards.columns else "Card Index"
        cards["account_id"] = "U" + cards["User"].astype(str) + "_C" + cards[cidx].astype(str)
        cards = cards.set_index("account_id").reindex(acc_ids).reset_index()
        data["Account"].x = torch.tensor(encode_node_features(
            cards, ["Card Brand", "Card Type", "Has Chip", "Card on Dark Web"],
            ["Credit Limit", "Cards Issued", "Year PIN last Changed"]), dtype=torch.float32)
    except Exception as e:
        print(f"[graph] AVISO Account sem features ({e})")
        data["Account"].x = torch.zeros((len(acc_ids), 1), dtype=torch.float32)

    gen = torch.Generator().manual_seed(42)
    data["Merchant"].x = torch.randn((len(mer_ids), 16), generator=gen)
    data["User"].x = torch.randn((len(usr_ids), 16), generator=gen)
    if loc_ids:
        data["Location"].x = torch.randn((len(loc_ids), 16), generator=gen)

    pares = tx[["document_id", "account_id", "login_user_id", "merchant_id", "city_id"]].drop_duplicates()
    da = pares[["document_id", "account_id"]].drop_duplicates()
    data["Document", "OWNS_ACCOUNT", "Account"].edge_index = torch.tensor(
        [da["document_id"].map(doc_idx).tolist(), da["account_id"].map(acc_idx).tolist()], dtype=torch.long)
    au = pares[["account_id", "login_user_id"]].drop_duplicates()
    data["Account", "HAS_USER", "User"].edge_index = torch.tensor(
        [au["account_id"].map(acc_idx).tolist(), au["login_user_id"].map(usr_idx).tolist()], dtype=torch.long)
    am = pares.dropna(subset=["merchant_id"])[["account_id", "merchant_id"]].drop_duplicates()
    data["Account", "TRANSACTS_WITH", "Merchant"].edge_index = torch.tensor(
        [am["account_id"].map(acc_idx).tolist(), am["merchant_id"].map(mer_idx).tolist()], dtype=torch.long)
    if loc_ids:
        # comerciante localizado em cidade/UF (sinal espacial — STA-GT)
        ml = tx[["merchant_id", "city_id"]].dropna().drop_duplicates()
        data["Merchant", "LOCATED_IN", "Location"].edge_index = torch.tensor(
            [ml["merchant_id"].map(mer_idx).tolist(), ml["city_id"].map(loc_idx).tolist()], dtype=torch.long)

    # arestas reversas (REV_): o HGTConv exige que todo tipo de nó seja destino de
    # alguma aresta; sem isso, 'Document' (só origem) não recebe embedding.
    for (src, rel, dst) in list(data.edge_types):
        rev = (dst, "REV_" + rel, src)
        if rev not in data.edge_types:
            data[rev].edge_index = data[src, rel, dst].edge_index.flip(0).contiguous()

    print(data)
    return data, doc_idx


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Adaptador IBM TabFormer -> formato TFM (discreto).")
    ap.add_argument("--tx", required=True)
    ap.add_argument("--users", required=True)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--outdir", default="data/processed_tabformer")
    ap.add_argument("--task", choices=["windowed", "entity"], default="windowed")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--min-events", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=128, help="cap de comprimento (modo entity)")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--fold-errors", action="store_true",
                    help="dobra o Errors? no event_type (padrão off — degradou o modelo nos testes)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tx = load_transactions(args.tx, nrows=args.nrows, fold_errors=args.fold_errors)
    ev_map, merch_map, mcc_map = build_discrete_maps(tx)

    graph, doc_idx = build_graph(tx, args.users, args.cards)
    torch.save(graph, os.path.join(args.outdir, "graph_data.pt"))
    torch.save(doc_idx, os.path.join(args.outdir, "doc_id_to_graph_idx.pt"))

    if args.task == "entity":
        sequences, labels = build_entity(tx, args.max_seq_len, ev_map, merch_map, mcc_map)
        torch.save(sequences, os.path.join(args.outdir, "discrete_event_sequences.pt"))
        torch.save(labels, os.path.join(args.outdir, "fraud_labels.pt"))
        print(f"\n[ok] modo=entity | {args.outdir}/ : discrete_event_sequences.pt | fraud_labels.pt | "
              "graph_data.pt | doc_id_to_graph_idx.pt")
    else:
        sequences, labels, sample_to_doc, sample_time, on_grid = build_windowed(
            tx, args.window, args.stride, args.min_events, ev_map, merch_map, mcc_map)
        sample_to_graph_idx = {sid: doc_idx[d] for sid, d in sample_to_doc.items() if d in doc_idx}
        torch.save(sequences, os.path.join(args.outdir, "discrete_event_sequences.pt"))
        torch.save(labels, os.path.join(args.outdir, "fraud_labels.pt"))
        torch.save(sample_to_graph_idx, os.path.join(args.outdir, "sample_to_graph_idx.pt"))
        torch.save(sample_time, os.path.join(args.outdir, "sample_time.pt"))
        torch.save(on_grid, os.path.join(args.outdir, "sample_on_grid.pt"))
        print(f"\n[ok] modo=windowed | {args.outdir}/ :")
        print("     discrete_event_sequences.pt | fraud_labels.pt | graph_data.pt | doc_id_to_graph_idx.pt |")
        print("     sample_to_graph_idx.pt | sample_time.pt | sample_on_grid.pt")
        print("     REGRAS fine-tuning: split TEMPORAL por sample_time; TESTE=on_grid==True (natural); "
              "TREINO=todas do período (positivos densos); desbalanceamento só no treino.")


if __name__ == "__main__":
    main()
