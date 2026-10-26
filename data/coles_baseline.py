#!/usr/bin/env python3
"""
Baseline CoLES (pytorch-lifestream 0.7.0) no NOSSO split.

Pré-treino auto-supervisionado contrastivo (CoLES, encoder GRU) sobre as nossas
sequências discretas; depois extrai embeddings e treina uma cabeça downstream
(LightGBM) no train, prevendo no test. Salva test_predictions.csv com 'id' pra
comparar (sig) com o nosso TFM no MESMO conjunto de teste.

É um baseline auto-supervisionado de SEQUÊNCIA (sem grafo) — contraste direto com o
nosso TFM (sequência + grafo).

Uso (fraude OOT):
  python data/coles_baseline.py --dir data/processed_tabformer_temporal \
    --labels-name fraud_labels.pt --split-name sample_split.pt \
    --max-seq-len 16 --out results/coles_fraud_oot/test_predictions.csv
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raiz do projeto
import finetune_causal_ar_fraud as seqbase
CAT_INDICES, CONT_INDICES = seqbase.CAT_INDICES, seqbase.CONT_INDICES
KEYS, CAT_VOCAB, CAT_EMB, CAT_NAMES = seqbase.KEYS, seqbase.CAT_VOCAB, seqbase.CAT_EMB, seqbase.CAT_NAMES
best_thr = seqbase.best_thr
CONT_NAMES = ["value_processed", "delta_t_seconds"]


def build_records(sequences, labels, split, sids, max_len, min_len=2):
    """Um registro ptls por sample: event_time + 6 categóricos + 2 contínuos (arrays).
    min_len=2 no pré-treino (CoLES precisa fatiar; igual ao nosso TFM pretrain);
    min_len=1 na inferência/teste (igual ao filtro de teste do TFM)."""
    recs = []
    for sid in sids:
        ev = sequences[sid][-max_len:]
        if len(ev) < min_len:
            continue
        r = {"id": sid, "target": int(labels[sid]),
             "event_time": torch.arange(len(ev), dtype=torch.long)}
        for i in CAT_INDICES:
            name = KEYS[i]
            r[name] = torch.tensor([int(min(e.get(name, 0), CAT_VOCAB[name] - 1)) for e in ev], dtype=torch.long)
        for i in CONT_INDICES:
            name = KEYS[i]
            r[name] = torch.tensor([float(e.get(name, 0.0)) for e in ev], dtype=torch.float32)
        recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--labels-name", default="fraud_labels.pt")
    ap.add_argument("--split-name", default="sample_split.pt")
    ap.add_argument("--max-seq-len", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=15)   # simétrico ao nosso; subir só se não convergir
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--out", default="results/coles_baseline/test_predictions.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import pytorch_lightning as pl
    pl.seed_everything(args.seed)

    from ptls.data_load.datasets import MemoryMapDataset
    from ptls.frames import PtlsDataModule
    from ptls.frames.coles import CoLESModule, ColesDataset
    from ptls.frames.coles.split_strategy import SampleSlices
    from ptls.nn import TrxEncoder, RnnSeqEncoder

    seqs = torch.load(f"{args.dir}/discrete_event_sequences.pt", weights_only=False)
    labs = torch.load(f"{args.dir}/{args.labels_name}", weights_only=False)
    spl = torch.load(f"{args.dir}/{args.split_name}", weights_only=False)
    tr_ids = [s for s in spl if spl[s] == "train" and s in seqs and s in labs]
    va_ids = [s for s in spl if spl[s] == "val" and s in seqs and s in labs]
    te_ids = [s for s in spl if spl[s] == "test" and s in seqs and s in labs]
    print(f"amostras: train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)}")

    pretrain_recs = build_records(seqs, labs, spl, tr_ids + va_ids, args.max_seq_len)
    print(f"[coles] registros de pré-treino: {len(pretrain_recs):,}")

    trx = TrxEncoder(
        embeddings={KEYS[i]: {"in": CAT_VOCAB[KEYS[i]], "out": CAT_EMB[KEYS[i]]} for i in CAT_INDICES},
        numeric_values={n: "identity" for n in CONT_NAMES},
    )
    seq_encoder = RnnSeqEncoder(trx_encoder=trx, hidden_size=args.hidden, type="gru")
    model = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=1e-3, weight_decay=1e-5),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.9),
    )

    train_ds = ColesDataset(
        MemoryMapDataset(pretrain_recs, i_filters=[]),
        splitter=SampleSlices(split_count=5, cnt_min=max(2, args.max_seq_len // 4), cnt_max=args.max_seq_len),
    )
    dm = PtlsDataModule(train_data=train_ds, train_batch_size=args.batch_size, train_num_workers=8)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu", devices=1,
                         enable_checkpointing=False, logger=False)
    trainer.fit(model, dm)

    # ---- embeddings por sample (inferência) ----
    from ptls.data_load.datasets import inference_data_loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_encoder.to(device).eval()

    @torch.no_grad()
    def embed(ids):
        recs = build_records(seqs, labs, spl, ids, args.max_seq_len, min_len=1)  # teste: ≥1, igual ao TFM
        kept = [r["id"] for r in recs]
        embs = []
        for b in inference_data_loader(recs, num_workers=0, batch_size=512):   # num_workers=0: preserva ordem
            embs.append(seq_encoder(b.to(device)).detach().cpu().numpy())
        return kept, np.concatenate(embs, 0)

    tr_k, Xtr = embed(tr_ids); va_k, Xva = embed(va_ids); te_k, Xte = embed(te_ids)
    ytr = np.array([int(labs[s]) for s in tr_k]); yva = np.array([int(labs[s]) for s in va_k])
    yte = np.array([int(labs[s]) for s in te_k])
    print(f"[coles] emb train={Xtr.shape} val={Xva.shape} test={Xte.shape} | pos test={yte.mean():.4f}")

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    out_dir = os.path.dirname(args.out)

    def report(head, y, p, p_val, y_val):
        """AUC, PR-AUC e F1 (threshold escolhido no VAL, igual ao TFM). Grava metrics_<head>.json."""
        thr = best_thr(y_val, p_val)
        auc = float(roc_auc_score(y, p)); pr = float(average_precision_score(y, p))
        f1 = float(f1_score(y, (p >= thr).astype(int), zero_division=0))
        print(f"[coles:{args.split_name}] teste ({head}): AUC={auc:.4f} PR-AUC={pr:.4f} F1={f1:.4f} (thr={thr:.4f})")
        m = {"regime": args.split_name, "head": head, "seed": args.seed, "auc": auc, "pr_auc": pr, "f1": f1,
             "thr": float(thr), "epochs": args.epochs, "ft_epochs": args.ft_epochs if args.finetune else 0,
             "n_test": int(len(y)), "pos_rate": float(np.mean(y))}
        with open(os.path.join(out_dir, f"metrics_{head}.json"), "w") as fh:
            json.dump(m, fh, indent=2)
        return m

    dtr = lgb.Dataset(Xtr, label=ytr); dva = lgb.Dataset(Xva, label=yva)
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
              "num_leaves": 63, "feature_fraction": 0.7, "bagging_fraction": 0.7,
              "bagging_freq": 1, "verbose": -1, "seed": args.seed}
    booster = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    prob = booster.predict(Xte, num_iteration=booster.best_iteration)
    prob_va = booster.predict(Xva, num_iteration=booster.best_iteration)   # threshold F1 vem do val
    report("frozen", yte, prob, prob_va, yva)
    pd.DataFrame({"id": te_k, "label": yte, "prob": prob}).to_csv(args.out, index=False)
    print(f"[ok] salvo em {args.out}")

    # ---- fine-tune end-to-end (opcional) ----
    if args.finetune:
        print("\n[ft] fine-tune supervisionado ...")
        torch.backends.cudnn.enabled = False
        seq_encoder.train()
        from ptls.frames.supervised import SeqToTargetDataset, SequenceToTarget
        from ptls.nn import Head
        import torchmetrics
        recs_for = lambda ids, ml: build_records(seqs, labs, spl, ids, args.max_seq_len, min_len=ml)
        tr_recs, va_recs, te_recs = recs_for(tr_ids, 2), recs_for(va_ids, 2), recs_for(te_ids, 1)
        head = Head(input_size=args.hidden, objective="classification", num_classes=2)
        ftm = SequenceToTarget(seq_encoder=seq_encoder, head=head, loss=torch.nn.NLLLoss(),
                               metric_list=[torchmetrics.AUROC(task="multiclass", num_classes=2)],
                               optimizer_partial=partial(torch.optim.Adam, lr=1e-4, weight_decay=1e-5),
                               lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma=0.5))
        mk = lambda recs: SeqToTargetDataset(MemoryMapDataset(recs, i_filters=[]), target_col_name="target", target_dtype=torch.long)
        ftdm = PtlsDataModule(train_data=mk(tr_recs), valid_data=mk(va_recs), train_batch_size=256,
                              valid_batch_size=512, train_num_workers=8, valid_num_workers=8)
        pl.Trainer(max_epochs=args.ft_epochs, accelerator="gpu", devices=1,
                   enable_checkpointing=False, logger=False).fit(ftm, ftdm)
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
        ft_y = np.array([int(labs[s]) for s in ft_ids]); ft_y_va = np.array([int(labs[r["id"]]) for r in va_recs])
        report("finetune", ft_y, ft_prob, ft_prob_va, ft_y_va)
        ft_out = os.path.join(out_dir, "ft_test_predictions.csv")
        pd.DataFrame({"id": ft_ids, "label": ft_y, "prob": ft_prob}).to_csv(ft_out, index=False)
        print(f"[ok] fine-tune salvo em {ft_out}")


if __name__ == "__main__":
    main()
