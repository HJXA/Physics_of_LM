#!/bin/bash

set -e
set -o pipefail

# =========================
# 可修改参数
# =========================
MODEL_ROOT="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/QA/bioS_multi/sft_llama2_2026_04_06_11_18_57_2026_04_07_22_47_37"
TEST_DIR="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/test"
PY_SCRIPT="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/eval/eval.py"
RESULT_DIR="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/eval/result"

BATCH_SIZE=8192
MAX_INPUT_LENGTH=128
MAX_NEW_TOKENS=32

export CUDA_VISIBLE_DEVICES=2

echo "======================================="
echo "Start evaluation: $(date)"
echo "Model root: $MODEL_ROOT"
echo "======================================="

# =========================
# 解析 model_type / model_name
# =========================
model_name=$(basename "$MODEL_ROOT")
model_type=$(basename "$(dirname "$MODEL_ROOT")")

echo "Model type: $model_type"
echo "Model name: $model_name"

# =========================
# 检查 checkpoint
# =========================
CKPT_LIST=$(ls -d ${MODEL_ROOT}/checkpoint-* 2>/dev/null | sort -V)

if [ -z "$CKPT_LIST" ]; then
    echo "❌ No checkpoints found in $MODEL_ROOT"
    exit 1
fi

# =========================
# 遍历 checkpoint
# =========================
for CKPT in $CKPT_LIST; do
    CKPT_NAME=$(basename $CKPT)

    echo ""
    echo "======================================="
    echo "Evaluating: $CKPT_NAME"
    echo "======================================="

    # =========================
    # ✅ 精确 skip 逻辑（修复版）
    # =========================
    RESULT_CKPT_DIR="${RESULT_DIR}/${model_type}/${model_name}/${CKPT_NAME}"

    if [ -d "$RESULT_CKPT_DIR" ]; then
        EXIST=$(ls ${RESULT_CKPT_DIR}/metrics_*.txt 2>/dev/null | head -n 1)

        if [ ! -z "$EXIST" ]; then
            echo "⚠️ Skip (already evaluated): $CKPT_NAME"
            continue
        fi
    fi

    # =========================
    # 开始评测
    # =========================
    START_TIME=$(date +%s)

    python "$PY_SCRIPT" \
        --model_path "$CKPT" \
        --test_dir "$TEST_DIR" \
        --batch_size $BATCH_SIZE \
        --max_input_length $MAX_INPUT_LENGTH \
        --max_new_tokens $MAX_NEW_TOKENS

    END_TIME=$(date +%s)
    COST_TIME=$((END_TIME - START_TIME))

    echo "✅ Done: $CKPT_NAME (time: ${COST_TIME}s)"

done

echo ""
echo "======================================="
echo "🎉 All checkpoints evaluated!"
echo "End time: $(date)"
echo "======================================="