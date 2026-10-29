#!/usr/bin/env bash
# Quick REAL-data check on IBM TabFormer (stratified fraud), single seed, small
# sample, few epochs. Verifies the full graph pipeline runs on real data and
# produces sane numbers. NOT a paper run (no multiseed, tiny slice).
#
# Edit the three CSV paths below, then run from the repo root inside your
# container, e.g.:
#   $DKS bash tests/quick_tabformer.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

# ---- EDIT THESE to your TabFormer CSVs ----
TX=${TX:-data/public/tabformer_credit_card/credit_card_transactions-ibm_v2.csv}
U=${U:-data/public/tabformer_credit_card/sd254_users.csv}
C=${C:-data/public/tabformer_credit_card/sd254_cards.csv}
# -------------------------------------------
NROWS=${NROWS:-300000}        # rows of the transactions CSV to read (small = fast)
CFG=config/finetune_fraud_config_1024_cap.yaml
D=data/qtab
mkdir -p results ckpt

echo "== 1. prepare (small slice, context 16) =="
python3 data/prepare_tabformer.py --task windowed --window 16 --stride 16 --nrows "$NROWS" \
  --tx "$TX" --users "$U" --cards "$C" --outdir "$D"

echo "== 2. stratified split file (for canonical XGBoost) =="
python3 data/rebuild_split_stratified.py --dir "$D"

echo "== 3. AR pretrain, graph (2 epochs) =="
python3 pretrain_tfm_ar.py --objective ar \
  --sequences "$D/discrete_event_sequences.pt" \
  --graph-path "$D/graph_data.pt" --doc-map-path "$D/sample_to_graph_idx.pt" \
  --config-path "$CFG" --max-seq-len 16 --epochs 2 --output-dir ckpt/qtab

echo "== 4. fine-tune, graph, stratified, seed 42 (3 epochs) =="
python3 finetune_tfm_ar_fraud.py --split stratified --seed 42 --max-seq-len 16 --epochs 3 \
  --pretrained-checkpoint ckpt/qtab/last_model.pt \
  --sequences "$D/discrete_event_sequences.pt" --labels "$D/fraud_labels.pt" \
  --graph-path "$D/graph_data.pt" --doc-map-path "$D/sample_to_graph_idx.pt" \
  --config-path "$CFG" --output-dir results/qtab_graph_strat_s42

echo "== 5. canonical XGBoost (59 features, same stratified split) =="
python3 data/xgb_canonical.py --dir "$D" --labels-name fraud_labels.pt \
  --split-name sample_split_strat.pt --out results/qtab_xgb/test_predictions.csv

echo "== 6. F1-optimal (uniform) =="
python3 data/compute_f1_optimal.py \
  "TFM=results/qtab_graph_strat_s42/test_predictions.csv" \
  "XGB=results/qtab_xgb/test_predictions.csv"

echo "QUICK TABFORMER CHECK DONE"
