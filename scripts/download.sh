#!/usr/bin/env bash
# Pull turboderp's Qwen3.8-Flash-Next EXL3 pack (3.05 bpw, ~80 GB) to ~/models.
# Do NOT run vllm-exl3's prepare_pack.sh / regenerate_safetensors_index.py on this copy: the
# native engine wants ngram_embedding.safetensors OUT of the index (it is loaded file-backed,
# which is what lets an 80 GB pack fit in 61.4 GiB of GPU-visible memory).
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/exllamav3-amd}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-EXL3}"
REV="${REV:-3.05bpw_h5_ng5}"
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
"$REPO_DIR/.venv/bin/hf" download turboderp/Qwen3.8-Flash-Next-exl3 --revision "$REV" --local-dir "$MODEL_DIR"
du -sh "$MODEL_DIR"
ls "$MODEL_DIR"
