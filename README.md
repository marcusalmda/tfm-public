# TFM — Self-Supervised Graph-Sequence Pretraining for Financial Event Modeling

Public reproduction code for the paper *"Self-Supervised Graph-Sequence
Pretraining for Financial Event Modeling"*. The Transactional Foundation Model
(TFM) encodes each document's relational neighbourhood with a heterogeneous
graph transformer (HGT) and prefixes that embedding as the first token of a
sequence Transformer, pretrained with an **autoregressive (AR) next-event**
objective and fine-tuned for fraud / churn.

This repository reproduces the **public-benchmark** results only:

| Dataset | Task | Source |
|---|---|---|
| IBM TabFormer | card-fraud detection (stratified + out-of-time) | Kaggle `ealtman2019/credit-card-transactions` |
| RelBench rel-hm | user churn | `relbench` package (`rel-hm`, `user-churn`) |

> The third dataset in the paper (a private banking corpus) is **not** included:
> it cannot be redistributed. All scripts here run on the two public datasets.
>
> Only the **AR** objective is included, because that is the objective used on
> the public datasets in the paper (the MEM-vs-AR comparison was run on the
> private dataset only). The single tabular baseline is the **canonical
> XGBoost** (59 document-level features); there is no "inline" XGBoost variant.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch` / `torch-geometric` should be installed following the official
instructions for your CUDA version. A single modern GPU is enough.

All scripts expect the repository root on `PYTHONPATH` (several `data/` scripts
import modules at the repo root), so run them from the repo root like:

```bash
PYTHONPATH=. python finetune_causal_ar_fraud.py ...
PYTHONPATH=. python data/coles_baseline.py ...
```

## Repository layout

```
src/                       shared model + utilities
  models/                  CausalEventModel, HGT encoder, TFM, Transformer encoder
  baselines/               canonical feature extractor (59 features)
  utils/                   evaluation helpers (splits, metrics)
  datasets.py, pretrain.py
pretrain_causal_ar.py      AR pretraining, sequence-only (no graph)
pretrain_tfm_ar.py         AR pretraining, graph-conditioned (HGT prefix token)
finetune_tfm_ar_fraud.py   binary fine-tune WITH graph (TabFormer fraud / rel-hm churn)
finetune_causal_ar_fraud.py  binary fine-tune, sequence-only / no-graph ablation
config/finetune_fraud_config_1024_cap.yaml  HGT + Transformer config (graph path)
data/
  prepare_tabformer.py          Kaggle CSVs -> TFM event/graph tensors
  prepare_tabformer_temporal.py temporal windows + sample_time / on_grid
  rebuild_split_tabformer_match.py  stratified split matched to the TFM
  prepare_relbench_hm.py        RelBench rel-hm -> TFM event tensors
  coles_baseline.py             CoLES baseline for TabFormer (frozen + fine-tune)
  coles_relbench.py             CoLES baseline for rel-hm
  xgb_canonical.py              canonical XGBoost (59 features)
  aggregate_seeds.py            multiseed aggregation + paired significance
  sig_tabformer.py              DeLong (AUROC) + paired bootstrap (PR-AUC, F1)
  compute_f1_optimal.py         F1-optimal operating point from predictions
```

## Metrics and the F1 convention

Ranking metrics are **AUROC** and **PR-AUC**. For the thresholded **F1**, the
convention differs by dataset but is **uniform across all models within a
dataset** (no model is favoured):

- **rel-hm** (dense, ~81% positive): F1 at the **validation** threshold.
- **TabFormer** fraud (sparse, 0.11–1.2% positive): the **F1-optimal** operating
  point, applied to every model (TFM, CoLES, XGBoost). At such low prevalence
  the validation threshold is unstable and F1 at it degenerates, so the paper
  reports the F1-optimal point uniformly. Compute it from the saved
  `test_predictions.csv` with `data/compute_f1_optimal.py` — nothing is
  retrained.

```bash
PYTHONPATH=. python data/compute_f1_optimal.py \
  "TFM+graph=results/fraud_tfm_graphsample_strat_s42/test_predictions.csv" \
  "CoLES-FT=results/coles_tab_strat_sl16/seed42/ft_test_predictions.csv" \
  "XGBoost=results/xgb_canonical_tab_strat/seed42/test_predictions.csv"
```

## Reproducing — TabFormer (fraud)

1. **Data.** Download the three Kaggle CSVs
   (`credit_card_transactions-ibm_v2.csv`, `sd254_users.csv`, `sd254_cards.csv`)
   into `data/public/tabformer_credit_card/`, then build tensors and splits:

   ```bash
   PYTHONPATH=. python data/prepare_tabformer.py \
     --tx data/public/tabformer_credit_card/credit_card_transactions-ibm_v2.csv \
     --users data/public/tabformer_credit_card/sd254_users.csv \
     --cards data/public/tabformer_credit_card/sd254_cards.csv \
     --outdir data/processed_tabformer
   PYTHONPATH=. python data/prepare_tabformer_temporal.py \
     --tx data/public/tabformer_credit_card/credit_card_transactions-ibm_v2.csv \
     --users data/public/tabformer_credit_card/sd254_users.csv \
     --outdir data/processed_tabformer
   PYTHONPATH=. python data/rebuild_split_tabformer_match.py --dir data/processed_tabformer
   ```

2. **Pretrain (AR).** Graph-conditioned and sequence-only variants (the graph
   run reads the HGT/Transformer sizes from the config):

   ```bash
   PYTHONPATH=. python pretrain_tfm_ar.py --objective ar \
     --sequences data/processed_tabformer/discrete_event_sequences.pt \
     --graph-path data/processed_tabformer/graph_data.pt \
     --doc-map-path data/processed_tabformer/sample_to_graph_idx.pt \
     --config-path config/finetune_fraud_config_1024_cap.yaml \
     --max-seq-len 16 --output-dir checkpoints/tabformer_ar_graph
   PYTHONPATH=. python pretrain_causal_ar.py --objective ar \
     --sequences data/processed_tabformer/discrete_event_sequences.pt \
     --max-seq-len 16 --output-dir checkpoints/tabformer_ar_seq
   ```

3. **Fine-tune** (run per seed 42–46 for the stratified five-seed mean; OOT is a
   single run). `--split stratified` = in-distribution, `--split tabformer` = OOT.
   `finetune_tfm_ar_fraud.py` is the graph model; `finetune_causal_ar_fraud.py` is
   the sequence-only (no-graph) ablation:

   ```bash
   # with graph (TFM+AR, graph)
   PYTHONPATH=. python finetune_tfm_ar_fraud.py --split stratified --seed 42 \
     --pretrained-checkpoint checkpoints/tabformer_ar_graph/best_model.pt \
     --sequences data/processed_tabformer/discrete_event_sequences.pt \
     --labels data/processed_tabformer/fraud_labels.pt \
     --graph-path data/processed_tabformer/graph_data.pt \
     --doc-map-path data/processed_tabformer/sample_to_graph_idx.pt \
     --config-path config/finetune_fraud_config_1024_cap.yaml \
     --max-seq-len 16 --output-dir results/fraud_tfm_graph_strat_s42
   # no graph (ablation)
   PYTHONPATH=. python finetune_causal_ar_fraud.py --split stratified --seed 42 \
     --pretrained-checkpoint checkpoints/tabformer_ar_seq/best_model.pt \
     --sequences data/processed_tabformer/discrete_event_sequences.pt \
     --labels data/processed_tabformer/fraud_labels.pt \
     --max-seq-len 16 --output-dir results/fraud_tfm_seq_strat_s42
   ```

4. **Baselines** (same splits / context length):

   ```bash
   PYTHONPATH=. python data/coles_baseline.py --dir data/processed_tabformer \
     --max-seq-len 16 --finetune --out results/coles_tab_strat_sl16/seed42 --seed 42
   PYTHONPATH=. python data/xgb_canonical.py --dir data/processed_tabformer \
     --out results/xgb_canonical_tab_strat/seed42 --seed 42
   ```

5. **Significance / aggregation** and the F1-optimal column:

   ```bash
   PYTHONPATH=. python data/aggregate_seeds.py \
     --graph-dirs results/fraud_tfm_graph_strat_s{SEED} \
     --seq-dirs   results/fraud_tfm_seq_strat_s{SEED} --seeds 42 43 44 45 46
   PYTHONPATH=. python data/sig_tabformer.py --a <predA.csv> --b <predB.csv>
   ```

## Reproducing — rel-hm (churn)

```bash
# 1. data (needs the relbench package; downloads rel-hm to its cache)
PYTHONPATH=. python data/prepare_relbench_hm.py --outdir data/processed_relbench_hm

# 2. fine-tune TFM (binary churn), context of 32 events
PYTHONPATH=. python finetune_causal_ar_fraud.py --split relbench --seed 42 \
  --pretrained-checkpoint checkpoints/relbench_ar_graph/best_model.pt \
  --sequences data/processed_relbench_hm/discrete_event_sequences.pt \
  --labels data/processed_relbench_hm/churn_labels.pt \
  --max-seq-len 32 --output-dir results/churn_tfm_graph_s42

# 3. CoLES baseline
PYTHONPATH=. python data/coles_relbench.py --proc-dir data/processed_relbench_hm \
  --pre-max 32 --inf-max 32 --finetune --out results/coles_relbench/seed42 --seed 42
```

(Pretrain the rel-hm AR checkpoints with `pretrain_tfm_ar.py` /
`pretrain_causal_ar.py` using `--max-seq-len 32` and the rel-hm tensors, as in
step 2 of the TabFormer section.)

## Smoke test (tiny synthetic data)

To verify the whole pipeline runs without the real datasets, a self-contained
smoke test generates a tiny synthetic dataset in the exact tensor format and
runs AR pretraining (sequence + graph), both fine-tunes, canonical XGBoost and
the F1-optimal step on CPU in about a minute:

```bash
pip install -r requirements.txt        # needs torch, torch-geometric, xgboost
bash tests/smoke_test.sh                # prints "SMOKE TEST PASSED"
```

It checks that the code runs end to end and writes predictions; it does not
check accuracy (60 random samples, 1 epoch). `tests/make_tiny_data.py` builds
the fixture and `tests/smoke_config.yaml` is a tiny HGT/Transformer config used
only by the test.

## Notes

- Exact checkpoint paths, seeds and output directories above are templates;
  adjust them to your run. Hyperparameters match the paper (HGT 2 layers / 4
  heads / width 128; Transformer 8 layers / 8 heads / width 128; AdamW).
- The `test_predictions.csv` written by the fine-tune and baseline scripts has
  columns `id,label,prob` and is the single source consumed by
  `aggregate_seeds.py`, `sig_tabformer.py`, `xgb_canonical.py` and
  `compute_f1_optimal.py`.
- This code is released under the MIT License (see `LICENSE`).
