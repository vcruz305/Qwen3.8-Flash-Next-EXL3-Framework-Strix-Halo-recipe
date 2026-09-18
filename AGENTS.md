# AGENTS.md — for coding agents setting this up

You are setting up Qwen3.8-Flash-Next (EXL3, 80 GB) on an AMD Strix Halo machine
(Ryzen AI MAX+ 395, Radeon 8060S, `gfx1151`) using the `exllamav3-amd` runtime fork. This
file is the contract: what to run, what "done" looks like, and the failure modes that
will waste your context window if you debug them from first principles. Read it whole
before acting. Every rule here cost real hours; none is stylistic.

## Do this, in order

```bash
git clone https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-Framework-Strix-Halo-recipe.git
cd Qwen3.8-Flash-Next-EXL3-Framework-Strix-Halo-recipe
bash scripts/setup.sh        # ~10 min. Needs sudo once (apt). Idempotent; rerun on failure.
bash scripts/download.sh     # ~80 GB from Hugging Face to ~/models/Qwen3.8-Flash-Next-EXL3
GREEDY=1 bash scripts/run.sh -prompt "Explain gradient descent in two sentences." -no_think
```

`run.sh` without `GREEDY=1` uses chat.py's default sampler (temp 0.8). That is the
interactive path and lands at **25–37 t/s** with 37–63% draft acceptance. The README's
35–47 / 41.3-mean numbers are **greedy**. Do not compare them. The verification command
above sets `GREEDY=1` (`-temp 0`) so you are measuring the same thing the README quotes.

**Done means** the last command prints a coherent answer followed by a line like
`Generate: N tokens at 35-47 t/s - Draft: X / Y accepted (60-90%)`. Anything under
~30 t/s *with GREEDY=1 and the draft line present*, or a missing draft line, is a
misconfiguration — see "Speed is wrong" below. **~8–12 t/s with a draft line is the
reconstruct+hgemm fallback with MTP still on** (skipped rebuild, stale repo-root `.so`,
or GPU still held by another process). Do not declare success on "it produced text".

`setup.sh` ends with a self-check that must print `exl3_gemv_supported: True wmma_family: 2`.
If it prints `False`, stop and read Trap 1–2 below; do not proceed to the download.

## Preconditions to verify before you start

- `lsb_release -ds` → Ubuntu 26.04 (24.04 probably works, untested). Kernel ≥ 6.14 with
  in-tree `amdgpu`. `grep gfx_target_version /sys/class/kfd/kfd/topology/nodes/1/properties`
  must say `110501`. If the node index differs, search all nodes.
- ≥ 96 GB system RAM and the iGPU allowed ≥ 60 GB of it. After setup:
  `~/exllamav3-amd/.venv/bin/python -c "import torch;print(torch.cuda.get_device_properties(0).total_memory/2**30)"`
  must print ≥ 60. If not, the fix is in **BIOS** (iGPU memory allocation), not software.
  Tell the user; you cannot fix it from a shell.
- ~100 GB free on the disk holding `~/models` and `~/exllamav3-amd`.
- `/dev/kfd` and `/dev/dri/renderD128` readable by the user. Test with
  `python3 -c "open('/dev/kfd','rb')"`, not by checking group membership — the nodes often
  carry ACLs that make `render`/`video` membership unnecessary.
- Do **not** install ROCm from AMD's apt repo. Ubuntu 26.04 has none, and you do not need
  it: the torch wheel ships the runtime and the build borrows a toolchain from a pip wheel.

## The five traps (each looks like broken hardware)

1. **`torch.cuda.is_available()` lies.** ROCm 6.4 wheels have no gfx1151 code object;
   `is_available()` is `True`, then every kernel dies with `HIP error: invalid device
   function`. Check `torch.cuda.get_arch_list()` contains `gfx1151`. `setup.sh` installs
   `torch==2.10.0+rocm7.0`, which does. Never set `HSA_OVERRIDE_GFX_VERSION` — it does not
   rescue 6.4 and it corrupts kernel selection on 7.0.
2. **torch's bundled HSA runtime segfaults on the first allocation** (exit 139,
   `segfault ... in libhsa-runtime64.so`). Even `torch.zeros(4, device="cuda")`. No env var
   avoids it. `env.sh` fixes it with
   `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1` (Ubuntu's package). **Every
   shell that runs the model must `source ~/exllamav3-amd/env.sh` first.** If you write your
   own launcher and skip this, it segfaults and you will blame the driver.
3. **No `hipcc` in the torch wheel.** The extension build fails with the misleading
   `CUDA_HOME environment variable is not set`. `setup.sh` installs AMD's gfx1151 nightly
   `rocm[libraries,devel]` into a *separate* venv (`.venv-gfx1151`) purely as a toolchain.
   Use `_rocm_sdk_devel`, not `_rocm_sdk_core` (only devel has `hipsparse.h`, `thrust/`).
   The 1.6 GB wheel needs `UV_HTTP_TIMEOUT=600` or it times out mid-extraction and looks
   like a corrupt download.
4. **Device bitcode is at `lib/llvm/amdgcn/bitcode`**, not where clang probes →
   `cannot find ROCm device library`. `env.sh build` passes `--rocm-device-lib-path`.
   Build with `pip install --no-build-isolation --no-deps .`, **not `uv pip`** —
   `setup.py` imports its own package from the source tree and uv's isolated backend
   can't see it (`ModuleNotFoundError: No module named 'exllamav3'`).
5. **Build env is poison at runtime.** `source env.sh build` exports SDK paths whose HSA
   runtime overrides the preload and re-triggers Trap 2 — or, more confusingly, makes model
   load fail with a spurious `Insufficient VRAM`. **Build in one shell, run in a fresh
   one.** `setup.sh` does the build in a subshell for this reason.

## Speed is wrong

Expected **with `GREEDY=1`**: 35–47 t/s single stream, `Draft: ... accepted (60–90%)`.
Expected **without it** (`run.sh` default, temp 0.8): 25–37 t/s, acceptance 37–63%. That
is not a bug. Reproduced 2026-09-17 across 13 published-`run.sh` invocations.

| You see | Cause |
|---|---|
| ~4–5 t/s, no `Draft:` line | WMMA path inactive *and* MTP off. Wrong branch or you invoked `chat.py` yourself without `-mtp`. |
| **~8–12 t/s with a `Draft:` line** | **HIP GEMV inactive, MTP still on.** Reconstruct + hgemm is 4.3 t/s; MTP roughly doubles it. `run.sh` now aborts before load if `wmma_family != 2`. If you bypassed it: stale repo-root `exllamav3_ext*.so` (it shadows site-packages — copy the venv `.so` over it), or another process still holds the GPU (`hipblaslt 'out of memory'` at load, then a wedged 8 t/s run). Wait 10 s, `GREEDY=1 bash scripts/run.sh ...` once. Reproduced at 9.94 t/s on a GPU that had just been released. |
| ~15–20 t/s, no `Draft:` line | MTP off. `run.sh` passes `-mtp -ndt 3 -dds -dc 0.6`; if you wrote your own invocation, add them. |
| **25–37 t/s with a draft line, no `GREEDY=1`** | **Normal.** You are sampling. Do not "fix" this. |
| ~30 t/s *with* `GREEDY=1` and a draft line | `EXL3_MOE_CFG=2` / `EXL3_HIP_PREFILL_MIN_ROWS=2` not exported. `run.sh` sets them. |
| Numbers vary ±25% run to run | You are sampling. Benchmark greedy (`GREEDY=1` or `bench_mtp.py -g`); acceptance is deterministic only under greedy. |
| One prompt at 47, another at 35 | Normal. Draft acceptance is content-dependent. Quote `prompt_sweep.py`'s six-prompt mean, never a single prompt. |

**Do not attempt further kernel optimisation** unless the user explicitly asks. Every bulk
kernel already streams at this GPU's practical single-kernel rate (~135 GB/s); the fork's
`README.strix-halo.md` documents nine measured null results (int8 GEMV, CPU offload,
prefetch depth, reduce-pad, row-looped mixers, graph replay, …). Re-running them is the most
likely way to burn a day here. ~41 t/s mean is the ceiling for this pack on this GPU.

## Throughput vs latency: use batch.sh when you have more than one request

Single-stream decode is latency-limited near 47 tok/s greedy and that is a real wall.
Aggregate throughput is **not** capped there: ~32 ms of every verification forward is
row-independent, so concurrent sequences share it.

- `scripts/batch.sh -f prompts.txt` → **74 tok/s aggregate** (16-prompt queue, batch 5).
- **The 16-row rule:** the grouped-MoE kernel works in 16-row chunks. `batch × (ndt+1)` must
  land on or just under a multiple of 16. At `-ndt 2`, batch 5 = 15 rows = one full chunk;
  batch 6 = 18 rows and **loses 24%**. Do not "optimise" by raising the batch by one.
- **Queue deeper than batch**, or the batch drains and you measure ~51 instead of 74.
- Per-sequence latency *falls* as batch rises. For a single interactive reply use `run.sh`.

If a user asks for "more tok/s", establish first whether they mean one stream (47 is near
the wall) or total work done (batch it). These are different machines.

## Context length facts (don't re-measure these)

- fp16 KV cache: loads up to `-cs 106496`; 114688+ OOMs **at load** (transient — steady state
  would fit). Q4 cache (`-cq 4`): the model's full **262,144** loads (58.3 GiB) and decodes at
  30.9 tok/s with the cache completely full. **`run.sh` defaults to 200k + Q4**; `CACHE=262144` for the full window, `CACHE=32768 CQ=` for a short fp16 cache.
- Cold prefill is 450–500 tok/s with `-gcs 512` (run.sh default). A 262k prompt is ~9–10
  minutes. Do not report a "hang" before that. Don't be fooled by 900–1,500 tok/s prefill
  on repetitive text — that is the n-gram path recognising repeats, not the real rate.
- Decode vs depth is flat (36 → 31 tok/s from empty to full 262k). If you see a cliff, it is
  something else.

## Things that will bite a custom script

- **MTP caches**: `Cache(..., max_history=N)` must equal `num_draft_tokens` on **both** the
  trunk and draft caches, or: `RuntimeError: recurrent_state must be [num_slots,
  max_history + 1, ...]`. `-ndt ≥ 6` OOMs (MTP costs ~8 GiB). Copy `scripts/bench_mtp.py`.
- **Silent failures**: the generator swallows job exceptions into `res["error"]`. If you get
  zero tokens and no traceback, print that key inside `gen.iterate()` before touching
  anything else.
- **The pack must stay as published.** `ngram_embedding.safetensors` (32.6 GB) is *not* in
  `model.safetensors.index.json` and must not be added — the engine loads it file-backed,
  which is the only reason an 80 GB pack fits in 61 GiB. The `prepare_pack.sh` /
  `regenerate_safetensors_index.py` tools in the sibling **DGX Spark** recipes are for the
  vLLM plugin path and are wrong here. If you already ran them, re-download or use
  `make_native_view.sh` from that recipe.
- **`pip install .` copies the Python tree** into site-packages. Scripts run from the repo
  root import the live tree; `eval/ppl.py` and `tests/` import the installed copy. After
  editing any `.py`, reinstall or your fix "works in one harness and not the other".
  Likewise the repo-root `exllamav3_ext*.so` shadows the installed one — `setup.sh` copies
  the fresh build over it; if you rebuild by hand, do the same or you benchmark old code.
- **`chat.py` mode**: `-mode qwen35`. There is no `auto`. It also needs `pyperclip` and
  `prompt_toolkit` (installed by `setup.sh`).
- **Cosmetic noise you should ignore**: `Resource leak detected by SharedSignalPool, ~620
  Signals leaked` at exit; pages of `_POSIX_C_SOURCE` redefinition warnings from Triton's
  first JIT; `amdgpu.ids: No such file or directory`. `hipblaslt 'out of memory'` at load
  right after another process exited is a workspace race — wait 5 s and retry once before
  investigating.

## Operational hygiene on a shared box

- Never `pkill -f "python.*something"` over ssh — the pattern matches the ssh session's own
  shell and kills it first. Use the PID.
- Never `while pgrep -f <name>` in the same shell command that contains `<name>` — it matches
  itself and spins forever.
- Write remote logs on the remote (`> /tmp/x.log 2>&1`), never `ssh ... | grep | tail`: when
  the connection drops the pipe loses everything.
- The model takes ~20 s to load; back-to-back runs need `sleep 6` between them.

## What not to change

`scripts/env.sh` — every line is one of the five traps. `setup.sh`'s venv split (runtime vs
build SDK). The `EXL3_HIP_DEFINES="EXL3_HIP_STG_PAD"` build flag (+10%, the LDS bank-conflict
fix). The pack contents.

## Where the detail lives

- Runtime fork, engineering log, 40+ harnesses:
  https://github.com/vcruz305/exllamav3-amd (`README.strix-halo.md`, `tools/strix_halo/`)
- This recipe's `README.md`: results, tuning table, hardware, troubleshooting matrix.
- Same pack on NVIDIA DGX Spark (different fork, different traps — do not mix instructions):
  https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe

If you hit something not covered here, the fork's `README.strix-halo.md` almost certainly
covers it. Check there before inventing a workaround.
