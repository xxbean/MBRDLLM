#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_PATH=${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}
TASK=${TASK:-gsm8k_cot_zeroshot}
LIMIT_ARG=${LIMIT_ARG:-}
COMMON_ARGS="model_path='${MODEL_PATH}',constraints_text=\"200:The|201:answer|202:is\",gen_length=256,steps=256,block_length=32,answer_length=5"

mkdir -p logs

echo "[1/3] Full-step LLaDA on ${TASK}"
accelerate launch eval_llada.py \
  --tasks "${TASK}" \
  --model llada_dist \
  --model_args "${COMMON_ARGS},early_exit_method=none,metrics_log_path=logs/gsm8k_full.jsonl" \
  ${LIMIT_ARG}

echo "[2/3] Prophet confidence-gap early exit on ${TASK}"
accelerate launch eval_llada.py \
  --tasks "${TASK}" \
  --model llada_dist \
  --model_args "${COMMON_ARGS},early_exit_method=prophet,early_threshold=7.5,mid_threshold=5.0,late_threshold=2.5,metrics_log_path=logs/gsm8k_prophet.jsonl" \
  ${LIMIT_ARG}

echo "[3/3] MBMBR-risk early exit on ${TASK}"
accelerate launch eval_llada.py \
  --tasks "${TASK}" \
  --model llada_dist \
  --model_args "${COMMON_ARGS},early_exit_method=mbr,mbr_candidate_k=8,mbr_token_topk=3,mbr_risk_early=0.05,mbr_risk_mid=0.10,mbr_risk_late=0.20,metrics_log_path=logs/gsm8k_mbr.jsonl" \
  ${LIMIT_ARG}
