#!/usr/bin/env bash
# Batched generation: trades per-stream latency for ~2.2x aggregate throughput.
#
# On this GPU one stream is latency-limited near 47 tok/s, but ~32 ms of every MTP
# verification forward is row-independent (dense weights and mixers are read once no matter
# how many token positions ride along), so concurrent sequences share it. Measured on a
# 16-prompt queue, chat template, stop conditions, Q4 cache, greedy:
#
#   batch  1 ->  34.2 tok/s      batch  5 ->  74.0   <- best
#   batch  4 ->  66.1            batch  6 ->  56.1   <- see the 16-row rule below
#   batch  8 ->  68.0            batch 10 ->  72.7
#
# THE 16-ROW RULE: the grouped-MoE kernel processes expert rows in chunks of 16
# (MOE_PREFILL_ROWS_PER_CHUNK). A batch of B sequences at -ndt K puts B*(K+1) rows in one
# forward, so pick B so that B*(K+1) lands ON or JUST UNDER a multiple of 16. At the default
# -ndt 2 that is 3 rows per sequence, so B=5 gives exactly 15 rows -- one full chunk -- and
# B=6 gives 18, which is a full chunk plus a nearly-empty second one and costs 24%.
#
# Use this for agent fan-out, eval sweeps and offline jobs, and keep the QUEUE DEEPER THAN
# THE BATCH: aggregate throughput needs the batch kept full, so 4 prompts at -b 4 drains to
# ~51 tok/s while 16 prompts at -b 5 sustains 74. For one interactive chat use run.sh --
# batching makes each individual reply slower.
#
#   bash scripts/batch.sh -f scripts/example_prompts.txt
#   bash scripts/batch.sh -p "Explain X" -p "Summarize Y"
#   bash scripts/batch.sh -f prompts.txt -o out.jsonl --max-tokens 1024
#   bash scripts/batch.sh -f prompts.txt -b 10          # 30 rows, also good (72.7)
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/exllamav3-amd}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-EXL3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$REPO_DIR"
# shellcheck disable=SC1091
source env.sh                          # LD_PRELOAD of Ubuntu's HSA runtime + venv PATH
export EXL3_MOE_CFG="${EXL3_MOE_CFG:-2}"
export EXL3_HIP_PREFILL_MIN_ROWS="${EXL3_HIP_PREFILL_MIN_ROWS:-2}"
export EXL3_HIP_SKINNY_GEMM="${EXL3_HIP_SKINNY_GEMM:-1}"
export EXL3_HIP_GR_MIX_Q8="${EXL3_HIP_GR_MIX_Q8:-1}"
export REPO_DIR

# Same preflight as run.sh: without the WMMA path this silently runs ~5x slower.
.venv/bin/python - <<'PY'
import torch  # load libtorch before the extension (RPATH)
import exllamav3_ext as e, sys
if not (bool(e.exl3_gemv_supported(0)) and int(e.exl3_gemv_wmma_family(0)) == 2):
    print(f"HIP GEMV inactive: supported={e.exl3_gemv_supported(0)} "
          f"family={e.exl3_gemv_wmma_family(0)} (want True, 2). Re-run scripts/setup.sh.",
          file=sys.stderr)
    sys.exit(2)
PY

exec .venv/bin/python "$HERE/batch_generate.py" -m "$MODEL_DIR" "$@"
