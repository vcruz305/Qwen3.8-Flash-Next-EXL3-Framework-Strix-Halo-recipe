#!/usr/bin/env bash
# Chat with Qwen3.8-Flash-Next on Strix Halo through exllamav3's own engine, with the measured-
# best settings: MTP self-speculation (ndt=3, dynamic, confidence 0.6), grouped-MoE k-split 2,
# MTP verify batches routed to the expert-deduplicating kernel.
#
#   bash scripts/run.sh                       # interactive chat (sampling, temp 0.8)
#   bash scripts/run.sh -prompt "..."         # one prompt, exit
#   GREEDY=1 bash scripts/run.sh -prompt "..." # greedy: this is what the README 35-47 t/s numbers are
#   NDT=2 DC=0.4 bash scripts/run.sh          # the higher-acceptance / slightly slower-mean point
#   CACHE=262144 bash scripts/run.sh          # the model's full window
#   CACHE=32768 CQ= bash scripts/run.sh       # short context, fp16 cache (+2 tok/s)
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/exllamav3-amd}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-EXL3}"
NDT="${NDT:-3}"
DC="${DC:-0.6}"
CACHE="${CACHE:-204800}"   # 200k default; the full 262144 also fits (both need the Q4 cache below)
CQ="${CQ:-4}"                   # Q4 KV cache (-2 tok/s vs fp16). CQ= (empty) -> fp16, which only loads up to ~100k
GCS="${GCS:-512}"               # prefill chunk; 512 measured faster than chat.py's 2048 default (460 vs 414 t/s at 16k)
GREEDY="${GREEDY:-0}"

cd "$REPO_DIR"
# shellcheck disable=SC1091
source env.sh                          # LD_PRELOAD of Ubuntu's HSA runtime + venv PATH. Not optional.
export EXL3_MOE_CFG="${EXL3_MOE_CFG:-2}"
export EXL3_HIP_PREFILL_MIN_ROWS="${EXL3_HIP_PREFILL_MIN_ROWS:-2}"
# Defaults already on in the fork on ROCm; listed so they are visible:
export EXL3_HIP_SKINNY_GEMM="${EXL3_HIP_SKINNY_GEMM:-1}"
export EXL3_HIP_GR_MIX_Q8="${EXL3_HIP_GR_MIX_Q8:-1}"

# Fail fast if the HIP GEMV path is not the one we built. Reconstruct+hgemm + MTP lands at
# ~8-10 t/s — the number people report when they skip setup.sh or shadow a stale .so.
.venv/bin/python - <<'PY'
import exllamav3_ext as e, sys
ok = bool(e.exl3_gemv_supported(0)) and int(e.exl3_gemv_wmma_family(0)) == 2
if not ok:
    print("HIP GEMV inactive: exl3_gemv_supported=%s wmma_family=%s (want True, 2).\n"
          "You will get ~8-10 t/s (reconstruct + MTP) instead of 35-47 greedy.\n"
          "Rebuild: cd ~/exllamav3-amd && source env.sh build && "
          "EXL3_HIP_DEFINES=EXL3_HIP_STG_PAD .venv/bin/python -m pip install --no-build-isolation --no-deps . && "
          "cp .venv/lib/python3.12/site-packages/exllamav3_ext.cpython-*.so ." % (
              e.exl3_gemv_supported(0), e.exl3_gemv_wmma_family(0)), file=sys.stderr)
    sys.exit(2)
PY

extra=()
if [[ "$GREEDY" == "1" ]]; then extra+=(-temp 0); fi
if [[ -n "${CQ}" ]]; then extra+=(-cq "$CQ"); fi

exec .venv/bin/python examples/chat.py -m "$MODEL_DIR" -mode qwen35 \
  -mtp -ndt "$NDT" -dds -dc "$DC" -cs "$CACHE" -gcs "$GCS" -tps "${extra[@]}" "$@"
