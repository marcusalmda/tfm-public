#!/usr/bin/env python3
"""
Baseline CoLES JUSTO para CHURN (rel-hm) — histórico completo por cliente.
Espelha o coles_fullhist.py (fraude), trocando a fonte de dados pra rel-hm.

Pipeline: (1) sequências completas por cliente do rel-hm; (2) pré-treino CoLES
(contrastivo, GRU) excluindo período de teste; (3) embedding do prefixo até o
seed_time de cada amostra; (4) downstream LightGBM (+ fine-tune opcional).
Mesmo split/rótulos/métricas que o nosso TFM de churn; alinhado por id.

Uso:
  python data/coles_relbench.py --proc-dir data/processed_relbench_hm_temporal \
    --labels-name churn_labels.pt --split-name sample_split.pt --cutoff-mode oot \
    [--finetune] --out results/coles_churn_fullhist/test_predictions.csv
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import finetune_causal_ar_fraud as seqbase
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_relbench_hm as R

CAT_INDICES, CONT_INDICES = seqbase.CAT_INDICES, seqbase.CONT_INDICES
KEYS, CAT_VOCAB, CAT_EMB = seqbase.KEYS, seqbase.CAT_VOCAB, seqbase.CAT_EMB
CONT_NAMES = ["value_processed", "delta_t_seconds"]
CAP_ARTICLE, CAP_TYPE = 60000, 10000


def events_to_rec(rid, events, label=0, cm=None, cs=None):
    r = {"id": rid, "target": int(label), "event_time": torch.arange(len(events), dtype=torch.long)}
    for i in CAT_INDICES:
        n = KEYS[i]
        r[n] = torch.tensor([int(min(e.get(n, 0), CAT_VOCAB[n] - 1)) for e in events], dtype=torch.long)
    for i in CONT_INDICES:
        n = KEYS[i]
        v = np.array([float(e.get(n, 0.0)) for e in events], dtype=np.float32)
        if cm is not None:
            v = (v - cm[n]) / cs[n]
        r[n] = torch.tensor(v, dtype=torch.float32)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", default="data/processed_relbench_hm_temporal")
    ap.add_argument("--labels-name", default="churn_labels.pt")
    ap.add_argument("--split-name", default="sample_split.pt")
    ap.add_argument("--cutoff-mode", choices=["oot", "none"], default="oot")
    ap.add_argument("--pre-max", type=int, default=500)
    ap.add_argument("--inf-max", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--test-official", action="store_true",
                    help="avalia no rb_test OFICIAL via task.evaluate (número comparável ao leaderboard)")
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--out", default="results/coles_churn_fullhist/test_predictions.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import pytorch_lightning as pl
    pl.seed_everything(args.seed)
    from ptls.data_load.datasets import MemoryMapDataset, inference_data_loader
    from ptls.frames import PtlsDataModule
    from ptls.frames.coles import CoLESModule, ColesDataset
    from ptls.frames.coles.split_strategy import SampleSlices
    from ptls.nn import TrxEncoder, RnnSeqEncoder

    labs = torch.load(f"{args.proc_dir}/{args.labels_name}", weights_only=False)
    spl = torch.load(f"{args.proc_dir}/{args.split_name}", weights_only=False)
    stime = torch.load(f"{args.proc_dir}/sample_time.pt", weights_only=False)
    cids_sample = set(s.rsplit("@", 1)[0] for s in spl)       # clientes do sample 300k (pré-treino)

    # ---- rel-hm: transações + vocabs (replica prepare_relbench_hm) ----
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    print("[rb] carregando rel-hm ...")
    db = get_dataset("rel-hm", download=True).get_db()
    task = get_task("rel-hm", "user-churn", download=True)
    rb_test = None
    cids_load = set(cids_sample)
    if args.test_official:
        rb_test = task.get_table("test").df.copy()
        rb_test["customer_id"] = rb_test["customer_id"].astype(str)
        rb_test["timestamp"] = pd.to_datetime(rb_test["timestamp"])
        cids_load |= set(rb_test["customer_id"])              # inclui clientes do teste oficial (só p/ inferência)
    tx = db.table_dict["transactions"].df.copy()
    art = db.table_dict["article"].df.copy()
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    tx["customer_id"] = tx["customer_id"].astype(str)
    tx["article_id"] = tx["article_id"].astype(str)
    tx = tx[tx["customer_id"].isin(cids_load)].copy()        # clientes do sample (+ teste oficial, se pedido)
    tx["evt"] = tx["sales_channel_id"].fillna(0).astype(int) + 1
    art_rank = tx["article_id"].value_counts()
    art_map = {a: (i + 1 if i + 1 < CAP_ARTICLE else CAP_ARTICLE - 1) for i, a in enumerate(art_rank.index)}
    tx["art_idx"] = tx["article_id"].map(art_map).fillna(0).astype(int)
    art["article_id"] = art["article_id"].astype(str)
    type_col = "product_type_no" if "product_type_no" in art.columns else "product_type_name"
    art2type = art.set_index("article_id")[type_col].astype(str).to_dict()
    tvals = sorted(set(art2type.values()))
    type_map = {t: (i + 1 if i + 1 < CAP_TYPE else CAP_TYPE - 1) for i, t in enumerate(tvals)}
    tx["type_idx"] = tx["article_id"].map(art2type).map(type_map).fillna(0).astype(int)
    tx = tx.sort_values(["customer_id", "t_dat"])
    print(f"[rb] transações={len(tx):,} | clientes={tx['customer_id'].nunique():,}")

    # ---- sequências completas por cliente ----
    cust_events, cust_ts = {}, {}
    for cid, sub in tx.groupby("customer_id", sort=False):
        evs, ts = R.event_dicts_for_customer(sub)
        cust_events[cid] = evs
        cust_ts[cid] = pd.to_datetime(pd.Series(ts)).values
    print(f"[coles-rb] clientes com sequência: {len(cust_events):,}")

    # stats contínuos (amostra)
    sv = {KEYS[i]: [] for i in CONT_INDICES}
    for evs in cust_events.values():
        for e in evs:
            for i in CONT_INDICES:
                sv[KEYS[i]].append(float(e.get(KEYS[i], 0.0)))
        if len(sv[KEYS[CONT_INDICES[0]]]) >= 300000:
            break
    cont_mean = {n: float(np.mean(v)) for n, v in sv.items()}
    cont_std = {n: float(max(np.std(v), 1e-6)) for n, v in sv.items()}

    def card_to_feats(events):
        d = {}
        for i in CAT_INDICES:
            n = KEYS[i]
            d[n] = torch.tensor([int(min(e.get(n, 0), CAT_VOCAB[n] - 1)) for e in events], dtype=torch.long)
        for i in CONT_INDICES:
            n = KEYS[i]
            v = np.array([float(e.get(n, 0.0)) for e in events], dtype=np.float32)
            d[n] = torch.tensor((v - cont_mean[n]) / cont_std[n], dtype=torch.float32)
        return d
    card_feats = {c: card_to_feats(ev) for c, ev in cust_events.items()}

    te_ids = [s for s in spl if spl[s] == "test"]
    if args.cutoff_mode == "oot":
        cutoff = min(pd.Timestamp(stime[s]) for s in te_ids).to_datetime64()
        print(f"[coles-rb] OOT corte de teste: {cutoff}")
    else:
        cutoff = np.datetime64("2100-01-01")

    # ---- pré-treino CoLES ----
    pre_recs = []
    for cid in cids_sample:                                   # pré-treino SÓ no sample 300k (justo c/ o nosso TFM)
        evs = cust_events.get(cid)
        if evs is None:
            continue
        k = int((cust_ts[cid] < cutoff).sum())
        pref = evs[:k][-args.pre_max:]
        if len(pref) >= 2:
            pre_recs.append(events_to_rec(f"c_{cid}", pref, cm=cont_mean, cs=cont_std))
    print(f"[coles-rb] sequências de pré-treino (sample 300k): {len(pre_recs):,}")

    trx = TrxEncoder(embeddings={KEYS[i]: {"in": CAT_VOCAB[KEYS[i]], "out": CAT_EMB[KEYS[i]]} for i in CAT_INDICES},
                     numeric_values={n: "identity" for n in CONT_NAMES})
    seq_encoder = RnnSeqEncoder(trx_encoder=trx, hidden_size=args.hidden, type="gru")
    model = CoLESModule(seq_encoder=seq_encoder,
                        optimizer_partial=partial(torch.optim.Adam, lr=1e-3, weight_decay=1e-5),
                        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.9))
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(pre_recs))
    nv = max(1, int(0.1 * len(pre_recs)))
    splitter = SampleSlices(split_count=5, cnt_min=10, cnt_max=min(200, args.pre_max))
    train_ds = ColesDataset(MemoryMapDataset([pre_recs[i] for i in perm[nv:]], i_filters=[]), splitter=splitter)
    valid_ds = ColesDataset(MemoryMapDataset([pre_recs[i] for i in perm[:nv]], i_filters=[]), splitter=splitter)
    dm = PtlsDataModule(train_data=train_ds, valid_data=valid_ds, train_batch_size=args.batch_size,
                        valid_batch_size=args.batch_size, train_num_workers=8, valid_num_workers=8)
    from pytorch_lightning.callbacks import EarlyStopping
    pl.Trainer(max_epochs=args.epochs, accelerator="gpu", devices=1, enable_checkpointing=False, logger=False,
               callbacks=[EarlyStopping(monitor="recall_top_k", mode="max", patience=3, strict=False)]).fit(model, dm)

    # ---- embeddings por amostra (prefixo até seed_time) ----
    all_sids = list(stime.keys())
    _t64 = pd.to_datetime(pd.Series([stime[s] for s in all_sids])).values
    stime64 = dict(zip(all_sids, _t64))

    def sample_prefix_rec(sid):
        cid = sid.rsplit("@", 1)[0]
        cf = card_feats.get(cid)
        if cf is None or sid not in stime64:
            return None
        k = int(np.searchsorted(cust_ts[cid], stime64[sid], side="right"))
        if k < 1:
            return None
        a = max(0, k - args.inf_max)
        L = k - a
        r = {"id": sid, "target": int(labs.get(sid, 0)), "event_time": torch.arange(L, dtype=torch.long)}
        for i in CAT_INDICES:
            r[KEYS[i]] = cf[KEYS[i]][a:k]
        for i in CONT_INDICES:
            r[KEYS[i]] = cf[KEYS[i]][a:k]
        return r

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_encoder.to(device).eval()

    @torch.no_grad()
    def embed(ids):
        recs = [r for r in (sample_prefix_rec(s) for s in ids) if r is not None]
        kept = [r["id"] for r in recs]
        embs = []
        for b in inference_data_loader(recs, num_workers=0, batch_size=512):   # num_workers=0: preserva ordem
            embs.append(seq_encoder(b.to(device)).detach().cpu().numpy())
        return kept, np.concatenate(embs, 0)

    tr_ids = [s for s in spl if spl[s] == "train"]
    va_ids = [s for s in spl if spl[s] == "val"]
    tr_k, Xtr = embed(tr_ids); va_k, Xva = embed(va_ids); te_k, Xte = embed(te_ids)
    ytr = np.array([int(labs[s]) for s in tr_k]); yva = np.array([int(labs[s]) for s in va_k])
    yte = np.array([int(labs[s]) for s in te_k])
    print(f"[coles-rb] emb train={Xtr.shape} val={Xva.shape} test={Xte.shape} | churn test={yte.mean():.3f}")

    import json
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    best_thr = seqbase.best_thr
    out_dir = os.path.dirname(args.out)

    def report(head, y, p, p_val, y_val):
        """AUROC, PR-AUC e F1 (threshold escolhido no VAL, igual ao TFM). Grava metrics_<head>.json."""
        thr = best_thr(y_val, p_val)
        auc = float(roc_auc_score(y, p)); pr = float(average_precision_score(y, p))
        f1 = float(f1_score(y, (p >= thr).astype(int), zero_division=0))
        print(f"[coles-rb] teste ({head}): AUROC={auc:.4f} PR-AUC={pr:.4f} F1={f1:.4f} (thr={thr:.4f})")
        m = {"regime": "rel-hm-churn", "head": head, "seed": args.seed, "auc": auc, "pr_auc": pr, "f1": f1,
             "thr": float(thr), "epochs": args.epochs, "ft_epochs": args.ft_epochs if args.finetune else 0,
             "n_test": int(len(y)), "pos_rate": float(np.mean(y)), "pre_max": args.pre_max, "inf_max": args.inf_max}
        with open(os.path.join(out_dir, f"metrics_{head}.json"), "w") as fh:
            json.dump(m, fh, indent=2)
        return m

    dtr, dva = lgb.Dataset(Xtr, label=ytr), lgb.Dataset(Xva, label=yva)
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 63,
              "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "verbose": -1, "seed": args.seed}
    booster = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    prob = booster.predict(Xte, num_iteration=booster.best_iteration)
    prob_va = booster.predict(Xva, num_iteration=booster.best_iteration)   # threshold F1 vem do val
    report("frozen", yte, prob, prob_va, yva)
    pd.DataFrame({"id": te_k, "label": yte, "prob": prob}).to_csv(args.out, index=False)
    print(f"[ok] salvo em {args.out}")

    # ---- teste OFICIAL (rb_test via task.evaluate) ----
    if args.test_official and rb_test is not None:
        print("\n[oficial] avaliando no rb_test OFICIAL via task.evaluate ...")
        recs, rows = [], []
        for j, (cid, ts) in enumerate(zip(rb_test["customer_id"].values, rb_test["timestamp"].values)):
            cf = card_feats.get(cid)
            if cf is None:
                continue
            k = int(np.searchsorted(cust_ts[cid], np.datetime64(pd.Timestamp(ts)), side="right"))
            if k < 1:
                continue
            a = max(0, k - args.inf_max); L = k - a
            r = {"id": f"t{j}", "target": 0, "event_time": torch.arange(L, dtype=torch.long)}
            for i in CAT_INDICES:
                r[KEYS[i]] = cf[KEYS[i]][a:k]
            for i in CONT_INDICES:
                r[KEYS[i]] = cf[KEYS[i]][a:k]
            recs.append(r); rows.append(j)
        E = []
        for b in inference_data_loader(recs, num_workers=0, batch_size=512):
            E.append(seq_encoder(b.to(device)).detach().cpu().numpy())
        E = np.concatenate(E, 0)
        p = booster.predict(E, num_iteration=booster.best_iteration)
        pred_full = np.full(len(rb_test), float(ytr.mean()), dtype=float)  # sem histórico -> base rate
        for idx, pr in zip(rows, p):
            pred_full[idx] = pr
        print(f"[oficial] cobertos {len(rows):,}/{len(rb_test):,} do rb_test")
        try:
            m = task.evaluate(pred_full)
        except Exception as e:
            print("[oficial] task.evaluate(pred) falhou:", repr(e)[:100], "— tentando com target_table")
            m = task.evaluate(pred_full, task.get_table("test"))
        print(f"[oficial] task.evaluate (CoLES congelado+GBM) no rb_test: {m}")

    if args.finetune:
        print("\n[ft] fine-tune supervisionado (churn) ...")
        torch.backends.cudnn.enabled = False
        seq_encoder.train()
        from ptls.frames.supervised import SeqToTargetDataset, SequenceToTarget
        from ptls.nn import Head
        import torchmetrics
        recs_for = lambda ids: [r for r in (sample_prefix_rec(s) for s in ids) if r is not None]
        tr_recs, va_recs, te_recs = recs_for(tr_ids), recs_for(va_ids), recs_for(te_ids)
        head = Head(input_size=args.hidden, objective="classification", num_classes=2)
        ftm = SequenceToTarget(seq_encoder=seq_encoder, head=head, loss=torch.nn.NLLLoss(),
                               metric_list=[torchmetrics.AUROC(task="multiclass", num_classes=2)],
                               optimizer_partial=partial(torch.optim.Adam, lr=1e-4, weight_decay=1e-5),
                               lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma=0.5))
        mk = lambda recs: SeqToTargetDataset(MemoryMapDataset(recs, i_filters=[]), target_col_name="target", target_dtype=torch.long)
        ftdm = PtlsDataModule(train_data=mk(tr_recs), valid_data=mk(va_recs), train_batch_size=256,
                              valid_batch_size=512, train_num_workers=8, valid_num_workers=8)
        pl.Trainer(max_epochs=args.ft_epochs, accelerator="gpu", devices=1, enable_checkpointing=False, logger=False).fit(ftm, ftdm)
        ftm.to(device).eval()

        @torch.no_grad()
        def ft_predict(recs):
            ps = []
            for b in inference_data_loader(recs, num_workers=0, batch_size=512):   # num_workers=0: preserva ordem
                out = ftm(b.to(device))
                p = torch.softmax(out, dim=1)[:, 1] if out.ndim == 2 else torch.sigmoid(out).squeeze(-1)
                ps.append(p.detach().cpu().numpy())
            return np.concatenate(ps, 0)

        ft_ids = [r["id"] for r in te_recs]
        ft_prob = ft_predict(te_recs); ft_prob_va = ft_predict(va_recs)
        ft_y = np.array([int(labs[s]) for s in ft_ids]); ft_y_va = np.array([int(labs.get(r["id"], 0)) for r in va_recs])
        report("finetune", ft_y, ft_prob, ft_prob_va, ft_y_va)
        ft_out = os.path.join(os.path.dirname(args.out), "ft_test_predictions.csv")
        pd.DataFrame({"id": ft_ids, "label": ft_y, "prob": ft_prob}).to_csv(ft_out, index=False)
        print(f"[ok] fine-tune salvo em {ft_out}")


if __name__ == "__main__":
    main()
