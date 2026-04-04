#!/usr/bin/env bash

# 发生错误时立即退出，避免服务在半配置状态运行。
set -euo pipefail

# 如果你使用 conda，可以先手动激活环境：conda activate swift
# 这里不强制写死环境激活，避免脚本覆盖你的本地环境策略。

# 指定仅使用第 0 张 GPU（即 cuda:0）。
export CUDA_VISIBLE_DEVICES=0

# 建议显式设置 HF_ENDPOINT（如果你所在环境需要镜像）。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 设置模型路径：优先使用本地下载目录。
MODEL_PATH="${1:-./checkpoints/Qwen3-8B}"

# 设置服务暴露的模型名（客户端请求时使用该名称）。
SERVED_MODEL_NAME="${2:-Qwen3-8B}"

# 可选端口，默认 8000。
PORT="${3:-8000}"

# 打印启动信息。
echo "Starting vLLM on CUDA device 0"
echo "MODEL_PATH=${MODEL_PATH}"
echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "PORT=${PORT}"

# 启动 vLLM OpenAI 兼容服务。
# --dtype auto: 让 vLLM 自动选择合适精度；
# --gpu-memory-utilization 可按显存情况调小，例如 0.85。
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype auto \
  --gpu-memory-utilization 0.5
