#!/usr/bin/env bash
# Chat with Qwen3.8-Flash-Next on Strix Halo through exllamav3's own engine, with the measured-
# best settings: MTP self-speculation (ndt=3, dynamic, confidence 0.6), grouped-MoE k-split 2,
# MTP verify batches routed to the expert-deduplicating kernel.
#
#   bash scripts/run.sh                       # interactive chat
#   bash scripts/run.sh -prompt "..."         # one prompt, exit
#   NDT=2 DC=0.4 bash scripts/run.sh          # the higher-acceptance / slightly slower-mean point
#   CACHE=262144 bash scripts/run.sh         # the model's full window
#   CACHE=32768 CQ= bash scripts/run.sh       # short context, fp16 cache (+2 tok/s)
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/exllamav3-amd}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-EXL3}"
NDT="${NDT:-3}"
DC="${DC:-0.6}"
CACHE="${CACHE:-204800}"   # 200k default; the full 262144 also fits (both need the Q4 cache below)
CQ="${CQ:-4}"                   # Q4 KV cache (-2 tok/s vs fp16). CQ= (empty) -> fp16, which only loads up to ~100k

cd "$REPO_DIR"
source env.sh                          # LD_PRELOAD of Ubuntu's HSA runtime + venv PATH. Not optional.
export EXL3_MOE_CFG="${EXL3_MOE_CFG:-2}"
export EXL3_HIP_PREFILL_MIN_ROWS="${EXL3_HIP_PREFILL_MIN_ROWS:-2}"
# Defaults already on in the fork on ROCm; listed so they are visible:
export EXL3_HIP_SKINNY_GEMM="${EXL3_HIP_SKINNY_GEMM:-1}"
export EXL3_HIP_GR_MIX_Q8="${EXL3_HIP_GR_MIX_Q8:-1}"

exec .venv/bin/python examples/chat.py -m "$MODEL_DIR" -mode qwen35 \
  -mtp -ndt "$NDT" -dds -dc "$DC" -cs "$CACHE" ${CQ:+-cq "$CQ"} -tps "$@"
