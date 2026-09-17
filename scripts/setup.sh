#!/usr/bin/env bash
# One-shot setup of the AMD exllamav3 fork on a Strix Halo box (Ubuntu 26.04, gfx1151).
# Idempotent. Takes ~10 minutes, most of it the 1.6 GB rocm-sdk-devel wheel and the HIP build.
#
#   bash scripts/setup.sh                # clone to ~/exllamav3-amd, build, smoke-test
#   REPO_DIR=/path bash scripts/setup.sh # elsewhere
#
# What it encodes (each one cost hours the first time; see README "Five traps"):
#   1. torch 2.10.0+rocm7.0 wheel (ROCm 6.4 has no gfx1151 code object)
#   2. Ubuntu's libhsa-runtime64-1 preloaded over torch's (torch's segfaults on first alloc)
#   3. hipcc + device bitcode borrowed from AMD's gfx1151 nightly rocm-sdk-devel wheel
#   4. pip (not uv) for the extension build, --no-build-isolation
#   5. build env vars never exported at runtime (env.sh guards them behind `build`)
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/exllamav3-amd}"
REPO_URL="${REPO_URL:-https://github.com/vcruz305/exllamav3-amd.git}"
BRANCH="${BRANCH:-main}"   # main == strix-halo (the fork's default)
PY="${PY:-3.12}"

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

sudo apt-get install -y --no-install-recommends build-essential cmake ninja-build libhsa-runtime64-1 git

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# Runtime venv: official ROCm 7.0 torch (stable). Needs the HSA preload at runtime.
[[ -d .venv ]] || uv venv --python "$PY" .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/rocm7.0
uv pip install --python .venv/bin/python numpy pydantic safetensors tokenizers rich pyyaml regex \
  ninja packaging wheel setuptools pip pillow marisa-trie datasets huggingface_hub hf_transfer \
  pyperclip prompt_toolkit

# Build-only venv: AMD gfx1151 nightly SDK for hipcc + bitcode + headers (_rocm_sdk_devel).
if [[ ! -d .venv-gfx1151 ]]; then
  uv venv --python "$PY" .venv-gfx1151
  UV_HTTP_TIMEOUT=600 uv pip install --python .venv-gfx1151/bin/python \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --prerelease=allow "rocm[libraries,devel]"
fi

# Root-level convenience symlinks the harnesses expect
for f in env.sh bench_mtp.py load_qwen.py; do [[ -e $f ]] || ln -s "tools/strix_halo/$f" "$f"; done

# Build the HIP extension with the LDS bank-conflict fix (the +10% one). Separate subshell so
# the SDK paths never leak into the runtime shell below.
(
  source env.sh build
  export PYTHONPATH="$PWD" EXL3_HIP_DEFINES="EXL3_HIP_STG_PAD"
  .venv/bin/python -m pip install --no-build-isolation --no-deps .
  # The repo-root .so shadows the installed one when running from the checkout; keep them equal
  cp .venv/lib/python$PY/site-packages/exllamav3_ext.cpython-*.so . 2>/dev/null || true
)

# Smoke test: arch really compiled in, real compute works, fast path active (family 2 = gfx11.5)
source env.sh
.venv/bin/python - <<'PY'
import torch, exllamav3_ext as e
assert "gfx1151" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
a = torch.randn(512, 512, dtype=torch.float16, device="cuda"); (a @ a).sum().item(); torch.cuda.synchronize()
print("GPU compute OK;", torch.__version__, torch.version.hip)
print("exl3_gemv_supported:", e.exl3_gemv_supported(0), "wmma_family:", e.exl3_gemv_wmma_family(0), "(want True, 2)")
PY
echo
echo "Done. Next: bash scripts/download.sh, then bash scripts/run.sh"
