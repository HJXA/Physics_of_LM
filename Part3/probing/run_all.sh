#!/bin/bash
# Convenience script to run the full probing pipeline.
#
# Usage:
#   ./run_all.sh [bioS_variant] [checkpoint_path]
#
# Examples:
#   ./run_all.sh bioS_single
#   ./run_all.sh bioS_multi checkpoints/bioS_multi/step-00005000/lit_model.pth

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROBING_DIR="$BASE_DIR/probing"
cd "$PROBING_DIR"

# Default values
VARIANT="${1:-bioS_single}"
CHECKPOINT="${2:-$BASE_DIR/checkpoints/llama2}"
SWANLAB_FLAG="${3:---no-swanlab}"

echo "============================================="
echo "  Probing Pipeline"
echo "============================================="
echo "  Variant:    $VARIANT"
echo "  Checkpoint: $CHECKPOINT"
echo "============================================="

# --- Step 1: Q-Probing ---
echo ""
echo ">>> Running Q-Probing for $VARIANT..."
python -m probing.train_q_probe \
    --variant "$VARIANT" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "probe_weights/q_probe_${VARIANT}" \
    --batch_size 200 \
    --max_steps 30000 \
    --swanlab

# --- Step 2: Evaluate Q-Probe ---
echo ""
echo ">>> Evaluating Q-Probe for $VARIANT..."
python -m probing.evaluate \
    --probe_mode q \
    --probe_path "probe_weights/q_probe_${VARIANT}/final.pt" \
    --checkpoint "$CHECKPOINT" \
    --eval_variant "$VARIANT" \
    --output_file "results/q_probe_${VARIANT}.json"

# --- Step 3: P-Probing (optional, slower) ---
echo ""
echo ">>> Running P-Probing for $VARIANT..."
python -m probing.train_p_probe \
    --variant "$VARIANT" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "probe_weights/p_probe_${VARIANT}" \
    --batch_size 200 \
    --max_steps 30000 \
    --swanlab

# --- Step 4: Evaluate P-Probe ---
echo ""
echo ">>> Evaluating P-Probe for $VARIANT..."
python -m probing.evaluate \
    --probe_mode p \
    --probe_path "probe_weights/p_probe_${VARIANT}/final.pt" \
    --checkpoint "$CHECKPOINT" \
    --eval_variant "$VARIANT" \
    --output_file "results/p_probe_${VARIANT}.json"

# --- Step 5: Generate Plots ---
echo ""
echo ">>> Generating plots..."
python -m probing.plot_results \
    --result_file "results/q_probe_${VARIANT}.json" \
    --mode q

python -m probing.plot_results \
    --result_file "results/p_probe_${VARIANT}.json" \
    --mode p

echo ""
echo "============================================="
echo "  Pipeline complete for $VARIANT"
echo "============================================="
echo "  Results: probing/results/"
echo "  Plots:   probing/plots/"
echo "============================================="
