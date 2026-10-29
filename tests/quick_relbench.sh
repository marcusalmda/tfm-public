#!/usr/bin/env bash
# Quick REAL-data check on RelBench rel-hm (churn), single seed, small sample,
# few epochs. Verifies the full graph pipeline runs on real data. NOT a paper
# run. Needs the `relbench` package (downloads rel-hm to its cache on first use;
# set HOME to a writable cache dir, as in your container).
#
# Run from the repo root inside your container, e.g.:
#   $DKS bash tests/quick_relbench.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

NSAMP=${NSAMP:-5000}           # rel-hm train customers to keep (small = fast)
CFG=config/finetune_fraud_config_1024_cap.yaml
D=data/qrel
mkdir -p results ckpt

echo "== 1. prepare (small sample, prune graph) =="
python3 data/prepare_relbench_hm.py --max-seq-len 32 \
  --max-train-samples "$NSAMP" --prune-to-samples --outdir "$D"

echo "== 2. AR pretrain, graph (2 epochs) =="
python3 pretrain_tfm_ar.py --objective ar \
  --sequences "$D/discrete_event_sequences.pt" \
  --graph-path "$D/graph_data.pt" --doc-map-path "$D/sample_to_graph_idx.pt" \
  --config-path "$CFG" --max-seq-len 32 --epochs 2 --output-dir ckpt/qrel

echo "== 3. fine-tune, graph, relbench split, seed 42 (3 epochs) =="
python3 finetune_tfm_ar_fraud.py --split relbench --seed 42 --max-seq-len 32 --epochs 3 \
  --pretrained-checkpoint ckpt/qrel/last_model.pt \
  --sequences "$D/discrete_event_sequences.pt" --labels "$D/churn_labels.pt" \
  --graph-path "$D/graph_data.pt" --doc-map-path "$D/sample_to_graph_idx.pt" \
  --sample-split-path "$D/sample_split.pt" \
  --config-path "$CFG" --output-dir results/qrel_graph_s42

echo "== 4. F1-optimal =="
python3 data/compute_f1_optimal.py "TFM=results/qrel_graph_s42/test_predictions.csv"

echo "QUICK REL-HM CHECK DONE"
