#!/usr/bin/env bash
# End-to-end smoke test on a tiny SYNTHETIC dataset (no real data, CPU, ~1 min).
# Verifies the whole repo runs: data fixture -> AR pretrain (seq + graph) ->
# fine-tune (no-graph + graph) -> canonical XGBoost -> F1-optimal.
# It checks that the pipeline RUNS and produces predictions; it does NOT check
# accuracy (60 random samples, 1 epoch).
set -euo pipefail
cd "$(dirname "$0")/.."            # repo root
export PYTHONPATH=.
D=tests/tiny_data
O=tests/out
CFG=tests/smoke_config.yaml
rm -rf "$O"; mkdir -p "$O"

echo "== 0. fixture =="
python tests/make_tiny_data.py

echo "== 1. AR pretrain, sequence-only =="
python pretrain_causal_ar.py --objective ar \
  --sequences $D/discrete_event_sequences.pt \
  --d-model 32 --nhead 2 --layers 1 --ffn 64 --epochs 1 --batch-size 16 --max-seq-len 8 \
  --output-dir $O/pre_seq

echo "== 2. fine-tune, no graph =="
python finetune_causal_ar_fraud.py --split stratified --seed 42 \
  --pretrained-checkpoint $O/pre_seq/last_model.pt \
  --sequences $D/discrete_event_sequences.pt --labels $D/fraud_labels.pt \
  --d-model 32 --nhead 2 --layers 1 --ffn 64 --epochs 1 --batch-size 16 --max-seq-len 8 \
  --output-dir $O/ft_seq

echo "== 3. AR pretrain, graph-conditioned =="
python pretrain_tfm_ar.py --objective ar \
  --sequences $D/discrete_event_sequences.pt \
  --graph-path $D/graph_data.pt --doc-map-path $D/sample_to_graph_idx.pt \
  --config-path $CFG --epochs 1 --batch-size 16 --max-seq-len 8 \
  --output-dir $O/pre_graph

echo "== 4. fine-tune, with graph =="
python finetune_tfm_ar_fraud.py --split stratified --seed 42 \
  --pretrained-checkpoint $O/pre_graph/last_model.pt \
  --sequences $D/discrete_event_sequences.pt --labels $D/fraud_labels.pt \
  --graph-path $D/graph_data.pt --doc-map-path $D/sample_to_graph_idx.pt \
  --config-path $CFG --epochs 1 --batch-size 16 --max-seq-len 8 \
  --output-dir $O/ft_graph

echo "== 5. canonical XGBoost (59 features) =="
python data/xgb_canonical.py --dir $D --labels-name fraud_labels.pt \
  --split-name sample_split_strat.pt --out $O/xgb/test_predictions.csv

echo "== 6. F1-optimal (uniform) =="
python data/compute_f1_optimal.py \
  "graph=$O/ft_graph/test_predictions.csv" \
  "seq=$O/ft_seq/test_predictions.csv" \
  "xgb=$O/xgb/test_predictions.csv"

echo "== 7. assert outputs exist =="
for f in $O/ft_seq/test_predictions.csv $O/ft_graph/test_predictions.csv $O/xgb/test_predictions.csv; do
  if [ -s "$f" ]; then echo "  OK  $f"; else echo "  MISSING $f"; exit 1; fi
done
echo "SMOKE TEST PASSED"
