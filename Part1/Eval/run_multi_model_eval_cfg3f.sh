#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="/ruilab/jxhe/miniconda3/envs/PoL/bin:$PATH"

############################
# 直接在这里修改所有配置参数
############################
TEST_DATA_PATH="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/datasets/512_Padding/cfg3f/test.parquet"
CONFIG_PATH="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/Lano-cfg/configs/cfg3f.json"
MODELS=(
  "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/gpt_standard_Pretrain/cfg3f/checkpoint-100000"
  "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/llama_rope/cfg3f/checkpoint-100000"
  "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/llama_wpe/cfg3f/checkpoint-100000"
)


NUM_EVAL_SAMPLES=20000
PREFIX_LEN=50
BATCH_SIZE=192
TEMPERATURE="1.0"
MAX_NEW_TOKENS=512
SAVE_ROOT="${SCRIPT_DIR}/result"
JUDGE_WORKERS=32
BADCASE_WORKERS=32
CUDA_DEVICES="0"

if [[ -z "${TEST_DATA_PATH}" ]]; then
  echo "[ERROR] 请在脚本顶部配置 TEST_DATA_PATH"
  exit 1
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "[ERROR] 请在脚本顶部的 MODELS 数组中至少配置一个模型路径"
  exit 1
fi

if [[ ! -f "${TEST_DATA_PATH}" ]]; then
  echo "[ERROR] 测试数据不存在: ${TEST_DATA_PATH}"
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] CFG 配置不存在: ${CONFIG_PATH}"
  exit 1
fi

if [[ -n "${CUDA_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

mkdir -p "${SAVE_ROOT}"

echo "[INFO] 共 ${#MODELS[@]} 个模型待评测"
echo "[INFO] 数据集: ${TEST_DATA_PATH}"
echo "[INFO] 配置: ${CONFIG_PATH}"
echo "[INFO] 结果根目录: ${SAVE_ROOT}"

for model_path in "${MODELS[@]}"; do
  echo ""
  echo "============================================================"
  echo "[INFO] 开始评测模型: ${model_path}"
  echo "============================================================"

  if [[ ! -e "${model_path}" ]]; then
    echo "[ERROR] 模型路径不存在: ${model_path}"
    exit 1
  fi

  model_base="$(basename "${model_path}")"
  model_parent_base="$(basename "$(dirname "${model_path}")")"
  model_grandparent_base="$(basename "$(dirname "$(dirname "${model_path}")")")"

  if [[ "${model_base}" == checkpoint* ]]; then
    model_type="${model_grandparent_base}"
    dataset_name="${model_parent_base}"
    checkpoints="${model_base}"
  else
    model_type="${model_base}"
    dataset_name="${model_parent_base}"
    checkpoints="End"
  fi

  sample_dir="${SAVE_ROOT}/${model_type}/${dataset_name}/${checkpoints}/samples_${NUM_EVAL_SAMPLES}"
  generated_path="${sample_dir}/generated_temp${TEMPERATURE}_all.jsonl"
  bad_case_path="${sample_dir}/bad_cases.jsonl"

  echo "[STEP 1/3] generate_cfg_samples.py"
  python "${SCRIPT_DIR}/generate_cfg_samples.py" \
    --test_data_path "${TEST_DATA_PATH}" \
    --model_path "${model_path}" \
    --num_eval_samples "${NUM_EVAL_SAMPLES}" \
    --prefix_len "${PREFIX_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --save_generated_path "${SAVE_ROOT}"

  if [[ ! -f "${generated_path}" ]]; then
    echo "[ERROR] 生成文件不存在（路径推断可能不匹配）: ${generated_path}"
    exit 1
  fi

  echo "[STEP 2/3] evaluate_cfg_accuracy.py"
  python "${SCRIPT_DIR}/evaluate_cfg_accuracy.py" \
    --config_path "${CONFIG_PATH}" \
    --p "${generated_path}" \
    --judge_workers "${JUDGE_WORKERS}"

  if [[ ! -f "${bad_case_path}" ]]; then
    echo "[ERROR] bad case 文件不存在: ${bad_case_path}"
    exit 1
  fi

  echo "[STEP 3/3] calc_bad_case_edit_distance.py"
  python "${SCRIPT_DIR}/calc_bad_case_edit_distance.py" \
    --config_path "${CONFIG_PATH}" \
    --p "${bad_case_path}" \
    --num_workers "${BADCASE_WORKERS}"

  echo "[INFO] 模型评测完成: ${model_path}"
done

echo ""
echo "[DONE] 全部模型评测完成。"