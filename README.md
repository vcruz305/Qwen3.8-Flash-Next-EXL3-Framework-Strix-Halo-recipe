# Qwen3.8-Flash-Next EXL3 on one Framework Desktop (AMD Strix Halo)

Runs [turboderp's Qwen3.8-Flash-Next EXL3 pack](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
(revision `3.05bpw_h5_ng5`, about 80 GB) on a single AMD Ryzen AI MAX+ 395 machine
(Radeon 8060S iGPU, **gfx1151 / RDNA 3.5**, 128 GB unified LPDDR5X) through the
[exllamav3-amd](https://github.com/vcruz305/exllamav3-amd) runtime fork — the native
ExLlamaV3 engine with a HIP decode backend ported from RDNA4 to RDNA 3.5.

Measured on `framework2` (Framework Desktop, Ubuntu 26.04), 2026-09-17, greedy, one stream,
512 new tokens per prompt:

| | |
|---|---|
| Decode, MTP self-speculation, six-prompt **mean** (**greedy**) | **41.3 tok/s** (ndt=3, dc=0.6) · 40.6 (ndt=2) |
| Decode, best prompt (greedy) | **47.5 tok/s** |
| Decode, worst prompt (greedy, prose, 63% draft acceptance) | 35.6 tok/s |
| Decode, **`run.sh` default** (sampling temp 0.8, 200k Q4, 13 runs) | **25–37 tok/s** (acceptance 37–63%) — this is what interactive chat prints |
| **Aggregate, batched** (`batch.sh`, 16-prompt queue, batch 5) | **74.0 tok/s** (2.2× one stream; see [Batched generation](#4-batched-generation-agent-fan-out-eval-sweeps-offline-jobs)) |
| Decode, no speculation | ~19 tok/s |
| Reconstruct + hgemm fallback **with MTP still on** | **~8–10 tok/s** (the "I followed AGENTS.md and got 8 t/s" report) |
| Stock AMD fork on this GPU (reconstruct + hgemm, no MTP) | 4.3 tok/s |
| Perplexity, 20 × 1024 tokens | 4.2259 — unchanged by every kernel change below |
| TTFT, short prompt | 0.25 s |
| **Max context** | **262,144 tokens — the model's full window — with a Q4 KV cache** (`-cq 4`); 98,304 with fp16 cache |
| Decode with the cache **full** (262,005-token cold prompt, Q4) | **30.9 tok/s**, 63% acceptance |
| Cold prefill (random tokens, chunk 512) | **450–500 tok/s**; 315–350 was the pre-fix rate (hipblaslt fp32-out GEMM at 6 TFLOP/s). Repetitive text still hits 900–1,500 via the n-gram path |
| TTFT, 262k cold prompt | 833 s (13.9 min) |
| GPU memory at load (model + MTP head + 200k Q4 cache) | ~57.7 GiB of 61.4 GiB visible |

That is **11× the fork's stock fallback path** and about 60–65% of what the same pack does
on a DGX Spark GB10 ([sibling recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe):
71–73 code / 46.5 prose). Everything is bandwidth: this part streams ~135 GB/s per kernel
against a 236 GB/s DRAM peak, and decode is 4.9 GiB of weight reads per verify step.

**Using a coding agent to set this up?** Point it at [AGENTS.md](AGENTS.md) first — the
contract, the definition of done, and the traps in the order an agent will hit them.

## Contents

- [Quick start](#quick-start)
- [Batched generation](#4-batched-generation-agent-fan-out-eval-sweeps-offline-jobs)
- [What the runtime fork changes](#what-the-runtime-fork-changes)
- [Tuning knobs and what they measured](#tuning-knobs-and-what-they-measured)
- [Context length](#context-length)
- [The five traps](#the-five-traps)
- [How things were measured](#how-things-were-measured)
- [Hardware, model, memory](#hardware-model-memory)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Related repositories, credits, license](#related-repositories)

## Quick start

Three scripts, all idempotent. About 10 minutes of setup plus the 80 GB download.

### Prerequisites

- A Strix Halo machine (Framework Desktop, or any Ryzen AI MAX+ 395 / 390 box) with
  **≥ 96 GB of RAM** (128 GB tested). Check the GPU-visible allocation — the pack needs
  ~58 GiB of it:
  ```bash
  python3 -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 2**30)"   # after setup
  ```
  If it reports less than ~60 GiB, raise the iGPU memory limit in BIOS (Framework: "GPU
  memory allocation" → 64 GB or higher). The VRAM carve-out itself can stay at 512 MB; the
  rest is GTT/unified.
- **Ubuntu 26.04** with the stock **7.0 kernel** (`amdgpu` in-tree; KFD reports
  `gfx_target_version 110501`). No ROCm apt repo needed — Ubuntu 26.04 has none, and the
  torch wheel brings its own runtime libs.
- ~100 GB free on NVMe for the pack plus the two venvs.
- `sudo` for one `apt-get install` (build tools and `libhsa-runtime64-1`).

### 1. Build the runtime

```bash
git clone https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-Framework-Strix-Halo-recipe.git
cd Qwen3.8-Flash-Next-EXL3-Framework-Strix-Halo-recipe
bash scripts/setup.sh
```

This clones [`exllamav3-amd`](https://github.com/vcruz305/exllamav3-amd) branch `main`
to `~/exllamav3-amd`, creates a runtime venv (torch `2.10.0+rocm7.0`, Python 3.12) and a
build-only venv (AMD's gfx1151 nightly `rocm-sdk-devel` for `hipcc` and device bitcode),
builds the HIP extension with `EXL3_HIP_DEFINES="EXL3_HIP_STG_PAD"` (~90 s, 37 translation
units), and ends with a smoke test that must print:

```
GPU compute OK; 2.10.0+rocm7.0 7.0.51831
exl3_gemv_supported: True wmma_family: 2 (want True, 2)
```

`wmma_family: 2` means the RDNA 3.5 WMMA GEMV path is active. `False` means you are on the
5-tok/s fallback; see Troubleshooting.

### 2. Download the pack

```bash
bash scripts/download.sh          # ~80 GB to ~/models/Qwen3.8-Flash-Next-EXL3
```

Leave the pack exactly as published. **Do not** run the vLLM-side pack-prep tools from the
Spark recipes on this copy: the native engine loads `ngram_embedding.safetensors` (32.6 GB)
separately and file-backed, and that is what fits an 80 GB pack into 61 GiB.

### 3. Run

```bash
bash scripts/run.sh                                   # interactive chat, prints tok/s per reply
bash scripts/run.sh -prompt "Explain gradient descent in two sentences."
GREEDY=1 bash scripts/run.sh -prompt "Explain gradient descent in two sentences." -no_think
```

`run.sh` is `examples/chat.py` with the measured-best flags:
`-mode qwen35 -mtp -ndt 3 -dds -dc 0.6 -cs 204800 -cq 4 -gcs 512 -tps` and `EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2`.
It **samples at temp 0.8** (chat.py's default) unless `GREEDY=1` (`-temp 0`). Interactive
speed is 25–37 t/s; the 35–47 / 41.3-mean table above is greedy only. `run.sh` aborts
before load if `wmma_family != 2` so you cannot silently land on the 8 t/s fallback.
**Default context is 200k tokens with a Q4 KV cache** (see [Context length](#context-length));
`CACHE=262144` gives the full window, `CACHE=32768 CQ=` a short fp16 cache that is ~2 tok/s faster.
`NDT=2 DC=0.4 bash scripts/run.sh` gives the higher-acceptance point (better on the easiest
prompts, ~2% lower mean). Any other `chat.py` flag passes through.

### 4. Batched generation (agent fan-out, eval sweeps, offline jobs)

One stream is latency-limited near 47 tok/s, but ~32 ms of every verification forward is
row-independent, so concurrent sequences share it — **aggregate throughput roughly doubles**:

```bash
bash scripts/batch.sh -f scripts/example_prompts.txt        # 74 tok/s aggregate
bash scripts/batch.sh -p "Explain X" -p "Summarize Y" -o out.jsonl
```

Measured on a 16-prompt queue (chat template, stop conditions, Q4 cache, greedy, 512 max):

| batch | rows in forward | aggregate tok/s |
|---|---|---|
| 1 | 3 | 34.2 |
| 4 | 12 | 66.1 |
| **5** | **15** | **74.0** |
| 6 | 18 | 56.1 ⚠ |
| 8 | 24 | 68.0 |
| 10 | 30 | 72.7 |

**The 16-row rule.** The grouped-MoE kernel processes expert rows in chunks of 16
(`MOE_PREFILL_ROWS_PER_CHUNK`). A batch of B sequences at `-ndt K` puts `B*(K+1)` rows into
one forward, so choose B such that `B*(K+1)` lands **on or just under a multiple of 16**. At
the default `-ndt 2` (3 rows/sequence) B=5 gives exactly 15 rows — one full chunk — while
B=6 gives 18, a full chunk plus a nearly-empty second one, and **loses 24%**. This is why
the default is 5 and not 6.

**Keep the queue deeper than the batch.** Aggregate throughput requires the batch stay full.
Four prompts at `-b 4` drains as sequences finish and yields ~51 tok/s; sixteen prompts at
`-b 5` sustains 74. Per-sequence latency falls as batch rises, so this is the wrong tool for
one interactive reply — use `run.sh` for that.

### 5. Benchmark (optional)

```bash
cd ~/exllamav3-amd && source env.sh
python bench_mtp.py -n 512 -ndt 3 -dds -g -dc 0.6          # one prompt, greedy: tok/s + acceptance
NDT=3 DC=0.6 python tools/strix_halo/prompt_sweep.py       # the six-prompt mean/median/min/max
python eval/ppl.py -m ~/models/Qwen3.8-Flash-Next-EXL3 -r 20 -l 1024   # must print 4.2259
```

Benchmark **greedy**. With sampling, draft acceptance — and so tok/s — swings 20–30% between
identical runs and A/B comparisons are meaningless. And quote the six-prompt distribution,
not a peak: acceptance is content-dependent and a single favourable prompt overstates real
throughput by ~30%.

## What the runtime fork changes

[`vcruz305/exllamav3-amd`](https://github.com/vcruz305/exllamav3-amd) (`main`; `strix-halo` is the same history) is
based on `sdougbrown/exllamav3` branch `integration` @ `991f1a0` (the community AMD/HIP port,
whose fast path is hard-gated to RDNA4 gfx1200/1201). Cumulative, each step measured
independently on this box, PPL identical throughout:

| # | Change | tok/s |
|---|---|---|
| — | stock fork on gfx1151: `exl3_gemv_supported()` false → reconstruct + hgemm | 4.3 |
| 1 | **RDNA 3.5 WMMA GEMV**: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` with the gfx11.5 fragment layouts (16 halves/lane A and B, *interleaved* C rows `2r + (lane>>4)`), verified on hardware with an fp32 oracle before any kernel code | 11.8 |
| 2 | Grouped-MoE kernels un-gated (they carry no RDNA4 assumption; "gfx12" in the name is historical) | 16.7 |
| 3 | Fused router gate un-gated | 17.9 |
| 4 | **MTP self-speculation** — the pack ships 6,200 `mtp.*` tensors; `-mtp -ndt 2 -dds` | 29.2 |
| 5 | `EXL3_MOE_CFG=2`: grouped-MoE k-split narrowed from 16 to 4 warps for a 20-CU part | 30.2 |
| 6 | `EXL3_HIP_PREFILL_MIN_ROWS=2`: MTP verify batches take the expert-*deduplicating* kernel (31% of expert reads in a 3-row batch are duplicates) | 32.7 mean |
| 7 | **LDS bank-conflict fix** (`EXL3_HIP_STG_PAD`): WMMA staging buffer inner extent 4 → 5 dwords, 8-way conflict → 2-way | 34.9 mean / 41.1 peak |
| 8 | Skinny fp16 GEMM replacing hipblaslt's split-K for the GDN `b/a` projections (60 µs → 24 µs × 72 calls) | 36.1 mean / 44.4 peak |
| 9 | **Int8 hyperconnection mixer weights** (`gr_mix_q8`): the only fp16 bulk in an EXL3 model, 22% of decode bytes | 40.0 mean / 46.3 peak |
| 10 | Block-wide scan in the MoE prefill metadata kernel (was one thread over 512 experts, ×48 layers) | 40.6 mean / 47.5 peak |
| 11 | ndt=3 with confidence 0.6 (re-swept once kernels got faster) | **41.3 mean** |

Also ported and kept, default off because they measured as nulls on this GPU: int8 GEMV
(`EXL3_INT8_GEMV`), CPU expert offload (`--moe_cpu_split`, a net *loss* on unified memory),
row-looped mixer kernels, a deeper B-prefetch ring, and the reduce-buffer pad. The fork's
[`README.strix-halo.md`](https://github.com/vcruz305/exllamav3-amd/blob/main/README.strix-halo.md)
has the full table, the null-result reasoning, and the 40+ profiling harnesses under
`tools/strix_halo/`.

## Tuning knobs and what they measured

| Knob | Default in `run.sh` | Measured |
|---|---|---|
| `-ndt` (draft tokens) | 3 | 2/3/4/5 → 40.6 / 41.3 / 39.7 / 39.0 mean. Acceptance decays faster than batching pays above 3. ≥ 6 OOMs (MTP costs ~8 GiB; `max_history` grows the recurrent state). |
| `-dc` (draft confidence) | 0.6 | Dead knob at ndt=2 (0.2–0.8 all 39.7–41.0). Matters at ndt=3: 0.5/0.6/0.7 → 43.3/44.0/43.6 on the easy prompt. |
| `EXL3_MOE_CFG` | 2 | 0/1/2 → 27.9 / 27.8 / 30.2 (with MTP). Stops mattering once `MIN_ROWS=2` routes MTP batches away from the decode kernel; keep 2 for the non-MTP path. |
| `EXL3_HIP_PREFILL_MIN_ROWS` | 2 | 17 (decode kernel for MTP batches) is faster per launch but loses end to end (44.3 vs 46.3) — it re-reads duplicate experts. |
| `EXL3_HIP_GR_MIX_Q8` | 1 | 0 → fp16 mixer weights, −11% decode. Int4 was tried and rejected: 10× the error on a gate applied 97× per forward. |
| `EXL3_HIP_SKINNY_GEMM` | 1 | 0 → hipblaslt for the small GDN projections, −2.5%. |
| `EXL3_BLOCK_GRAPH` | unset | Graph replay: neutral under MTP at every speed tested (47.0 vs 47.4 last). Not launch-bound. |
| `EXL3_INT8_GEMV` | unset | 1/2 → within 0.3 tok/s of off. m≤2 GEMVs are 6% of decode. |
| `-cs` (cache tokens) | 204800 | Memory, not speed. fp16 loads to 106k; **`-cq 4` loads the full 262,144**. See [Context length](#context-length). |
| `-cq` (cache quant) | 4 | `4` → 262k fits, −2 tok/s. `8` → 131k tested. |

## Context length

Measured 2026-09-17 with `tools/strix_halo/ctx_sweep.py` (MTP ndt=3 dc=0.6, greedy, 128 new
tokens). "Fill" rows use a **random-token prompt** sized to leave exactly 128 + draft slots free,
so they are true cold prefills with zero prefix reuse and the cache at 100%.

| Cache | Config | Loads? | Resident after load |
|---|---|---|---|
| 32,768 | fp16 | yes | 56.4 GiB |
| 98,304 | fp16 | yes | 58.3 GiB |
| 106,496 | fp16 | yes | 58.5 GiB |
| 114,688 | fp16 | **OOM** at load | — |
| 131,072 | Q8 (`-cq 8`) | yes | 57.7 GiB |
| **262,144** | **Q4 (`-cq 4`)** | **yes** | **58.3 GiB** |
| 393,216 | Q4 | loads (59.7 GiB) — beyond `max_position_embeddings`, for the record only |
| 524,288 | Q4 | OOM at load | — |

The fp16 cache costs ~30 KiB/token here (3 of every 4 layers are gated-delta-net recurrent
state, only every 4th is full attention), so the OOM at 112k is a **load-time transient**, not
steady state — the model itself is 55.5 GiB and a 100k fp16 cache steady-states at 58.5. Q4
shrinks the attention KV by 4× and makes the full 262k window fit with 3 GiB to spare.

Decode barely cares about depth. Q4 cache, 262,144 configured:

| Prompt tokens | Cold TTFT | Prefill tok/s | Decode tok/s | Acceptance | Peak GiB |
|---|---|---|---|---|---|
| 1,024 | 3.0 s | 345 | 36.1 | 62% | 58.7 |
| 8,192 | 23 s | 351 | 32.9 | 68% | 59.5 |
| 32,768 | 75 s | 435 | 33.1 | 66% | 59.5 |
| 65,536 | 101 s | 649 | 32.7 | 64% | 59.5 |
| 131,072 | 203 s | 646 | 36.0 | 73% | 59.6 |
| 200,000 | 219 s | 915 | 37.5 | 79% | 59.7 |
| 250,000 | 164 s | 1,526 | 34.4 | 72% | 59.7 |
| **262,005 (full, random)** | **833 s** | **314** | **30.9** | 63% | 59.5 |

fp16 cache, 98,304 configured: 38.7 / 38.0 / **37.0 tok/s at 1k / 32k / 98,165 (full)**.

Read it as: Q4 cache costs ~2 tok/s against fp16 at equal depth (36.1 vs 38.7 at 1k); going
from an empty cache to a full 262k one costs another ~5 (36 → 31); and the prefill numbers
above 32k on the repeated-README prompt are inflated by the pack's n-gram / prefix machinery
recognising repetition — **315–350 tok/s is the honest cold prefill rate**, which makes a full
262k prompt a 14-minute wait. For interactive use at long context, run at 131k (`-cs 131072
-cq 4`) or below; for a one-shot 200k+ document the machine can do it, budget the TTFT.

`run.sh` defaults to 200k with Q4; `CACHE=262144 bash scripts/run.sh` for the full window.

## The five traps

Each of these looks like broken hardware and each is encoded in `scripts/env.sh` /
`scripts/setup.sh`. Don't "simplify" them.

1. **ROCm 6.4 wheels have no gfx1151 code object.** `torch.cuda.is_available()` returns
   `True`, device properties say `gfx1151`, then every kernel dies with
   `HIP error: invalid device function`. `HSA_OVERRIDE_GFX_VERSION` does not rescue it.
   Use the `rocm7.0` wheel and check `torch.cuda.get_arch_list()`, never `is_available()`.
   Once on 7.0, `HSA_OVERRIDE_GFX_VERSION` must stay **unset**.
2. **torch's bundled HSA runtime segfaults on the first allocation** (`segfault ... in
   libhsa-runtime64.so`, exit 139), even `torch.zeros(4, device="cuda")`. No env var avoids
   it. Fix: `apt install libhsa-runtime64-1` and
   `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1` for every run. (AMD's
   gfx1151-native nightly torch does not have this bug, at the cost of being a nightly;
   `env.sh` documents both.)
3. **The torch wheel has no `hipcc`**, and reports `ROCM_HOME=None`, so the extension build
   fails with the misleading `CUDA_HOME environment variable is not set`. Borrow the
   toolchain from AMD's gfx1151 nightly `rocm[libraries,devel]` wheel in a separate venv —
   `_rocm_sdk_devel`, not `_rocm_sdk_core` (only `devel` has `hipsparse.h` and `thrust/`).
   `UV_HTTP_TIMEOUT=600`: the 1.6 GB wheel times out at uv's 30 s default mid-extraction.
4. **Device bitcode is in a non-standard place** (`lib/llvm/amdgcn/bitcode`), so every
   compile says `cannot find ROCm device library` until
   `--rocm-device-lib-path` is passed. And use `pip --no-build-isolation`, not `uv pip`, to
   build: `setup.py` imports its own package from the source tree.
5. **The SDK paths are build-only.** Exporting `ROCM_HOME`/`ROCM_PATH` at runtime re-triggers
   trap 2 (the SDK's own HSA runtime wins over the preload). `env.sh` gates them behind
   `source env.sh build`. **Build and bench in separate shells** — a bench in a build shell
   fails model load with a spurious "Insufficient VRAM".

## How things were measured

- Decode tok/s: `bench_mtp.py` — `(tokens − 1) / (wall − TTFT)` over 512 greedy tokens,
  in-process, MTP acceptance reported alongside. Six-prompt distribution via
  `prompt_sweep.py` (prompts listed in the script). All numbers are one stream.
- Fidelity for every kernel change: `eval/ppl.py -r 20 -l 1024` (4.225935 throughout), plus
  greedy token-ID A/B in separate processes (`tools/strix_halo/greedy_ab.py`) and
  teacher-forced per-position logit deltas with near-tie analysis
  (`tools/strix_halo/tie_check.py`). A kernel that changes summation order legitimately flips
  a coin-flip token; a flip with a multi-logit gap would be a defect. None were.
- Where time goes: `tools/strix_halo/phase_prof.py` — the trunk verify forward is 85–88% of
  every speculation round, the MTP head 10–12%, host ≈ 1%. `bytes_by_module.py` — 4.9 GiB
  of weight reads per verify step: routed experts 47%, dense EXL3 GEMVs 40%, mixers 13%.
- GPU utilisation from outside the process (`gpu_busy.py`, reads `gpu_busy_percent`,
  windowed to decode): 92% busy at 49% of DRAM bandwidth → kernels were latency-bound,
  which is what the LDS and k-split fixes addressed.

## Hardware, model, memory

| | |
|---|---|
| Machine | Framework Desktop, AMD Ryzen AI MAX+ 395 (Strix Halo), 128 GB LPDDR5X-8000 |
| GPU | Radeon 8060S, `gfx1151` (RDNA 3.5), wave32, **20 CUs**, 61.4 GiB GPU-visible |
| Measured bandwidth | 236 GB/s on a 2 GiB copy; ~130–145 GB/s practical for a single ≤ 64 MiB kernel |
| fp16 matmul | ~31 TFLOP/s (4096³) |
| OS / kernel | Ubuntu 26.04 LTS, 7.0.0 |
| torch | 2.10.0+rocm7.0 (HIP 7.0.51831), Python 3.12 |
| Build SDK | AMD gfx1151 nightly `rocm` 7.13.0a (hipcc, bitcode, headers) — build only |
| Model | `turboderp/Qwen3.8-Flash-Next-exl3` rev `3.05bpw_h5_ng5`, 80 GB, 48 layers, 512 experts top-10, MTP head |
| Resident at load | 49.8 GiB weights (n-gram table file-backed) + ~8 GiB MTP head and caches = 57.8 GiB |

## Known limitations

- **Single stream.** Concurrency was not measured; this is a desktop chat setup, not a server.
  For an OpenAI-compatible endpoint point [TabbyAPI](https://github.com/vcruz305/tabbyAPI) at
  the same venv — untested on this GPU.
- **~41 tok/s is the ceiling for this pack on this GPU with kernel work.** Every bulk kernel
  now streams at the part's practical rate; the gap to DRAM peak is spread over ~1,300 launches
  per verify step. Getting to 50 needs fewer bytes (a lower-bit expert re-quant) or a better
  drafter, not faster kernels.
- Cold prefill is 450–500 tok/s on random tokens with `-gcs 512` (was 315–350 before the
  fp16-accumulate workaround for hipblaslt's 6 TFLOP/s fp32-out GEMM). GB10 still does
  ~1,100. A 262k prompt is ~9–10 minutes. There is no prefix cache across processes.
- The pack's n-gram table (32.6 GB) stays on disk and is page-cached; the first prompt after
  a cold boot pays for that (~0 MiB read during decode once warm).
- `hipblaslt 'out of memory'` at load immediately after another process exited is a
  workspace race, not a leak: wait 5 s and retry.
- Cosmetic: `Warning: Resource leak detected by SharedSignalPool, ~620 Signals leaked` at
  exit, and pages of `_POSIX_C_SOURCE` redefinition warnings from Triton's first JIT. Both
  harmless.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `HIP error: invalid device function` | ROCm 6.4 torch. `pip list \| grep torch` must say `+rocm7.0`; `get_arch_list()` must include `gfx1151`. |
| exit 139 / segfault in `libhsa-runtime64.so` on first alloc | Trap 2 — `LD_PRELOAD` missing. `source env.sh` every shell. |
| `exl3_gemv_supported: False` after setup | Wrong branch (`integration` instead of `main`) or stale install. `git -C ~/exllamav3-amd branch`; rebuild. |
| `RuntimeError: exl3_moe_gfx12_k3 requires gfx1200/gfx1201` | Stock fork gate; you are not on `main`. |
| `Insufficient VRAM in split for model and cache` | Either a build shell (trap 5), `-ndt ≥ 6`, or the GPU memory limit in BIOS is below ~60 GB. |
| `RuntimeError: recurrent_state must be [num_slots, max_history + 1, ...]` | Custom script with `-mtp`: `Cache(max_history=)` must equal `num_draft_tokens` on **both** caches. `bench_mtp.py` shows the pattern. |
| Zero tokens generated, no error | Print `res["error"]` from `gen.iterate()`; the generator swallows job errors into the result dict. |
| `CUDA_HOME environment variable is not set` during build | Trap 3 — not in `source env.sh build`, or `.venv-gfx1151` missing. |
| `cannot find ROCm device library` | Trap 4 — `HIPCC_COMPILE_FLAGS_APPEND` not set; use `env.sh build`. |
| `ModuleNotFoundError: No module named 'exllamav3'` during build | Built with `uv pip`; use `pip --no-build-isolation`. |
| PPL fine but `bench_mtp.py` dies with `cannot convert float NaN` | A build that shrank `HIP_MMA_STG_WARPS`. PPL does not exercise every path; always run both. |
| Slow decode, **8–12 t/s, `Draft:` line present** | HIP GEMV inactive (reconstruct + MTP). `run.sh` now refuses to start. Stale repo-root `.so`, skipped `setup.sh`, or GPU still held (`hipblaslt 'out of memory'` then a wedged run — wait 10 s and retry). |
| Slow decode, 15–20 tok/s, no `Draft:` line | MTP off (`-mtp` missing). |
| Slow decode, 25–37 t/s, `Draft:` line, no `GREEDY=1` | **Normal sampling.** Do not "fix". Compare against greedy (`GREEDY=1`) or `bench_mtp.py -g`. |
| `run.sh` exits 2: `HIP GEMV inactive` | `wmma_family != 2`. Rebuild per the error text; copy the venv `.so` over the repo-root shadow. |

## Related repositories

| Repo | Role |
|---|---|
| [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) | the pack this recipe runs |
| **[vcruz305/exllamav3-amd](https://github.com/vcruz305/exllamav3-amd)** (`main`) | **the runtime fork this recipe builds**: RDNA 3.5 WMMA GEMV, un-gated grouped MoE, LDS fix, skinny GEMM, int8 mixers, all harnesses. |
| [sdougbrown/exllamav3](https://github.com/sdougbrown/exllamav3) `integration` | the community AMD/HIP port the fork is based on (remote `upstream`, @ `991f1a0`) |
| [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) | my *NVIDIA* fork (aarch64 + GB10 tuning) — different base, used by the Spark recipes, not this one |
| [Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe) | the same pack on a DGX Spark GB10 via vLLM + vllm-exl3 and the native engine; sibling this recipe is modeled on |
| [Qwen3.8-Flash-Next-EXL3-SGLang-DGX-Spark-recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-SGLang-DGX-Spark-recipe) | same pack, SGLang, GB10 |

## Credits and upstream work

**ExLlamaV3 by Turboderp ([@turboderp](https://github.com/turboderp-org/exllamav3)).** The
EXL3 trellis format, the quantization method, the engine, the MTP drafting support, and the
`Qwen3.8-Flash-Next-exl3` pack itself are theirs. MIT, Copyright (c) 2025 Turboderp.

**The HIP decode backend by [@sdougbrown](https://github.com/sdougbrown/exllamav3)**
(`integration` branch) — the grouped-MoE kernels, fused router, HIP GEMV scaffolding and
CPU-offload handoff that the RDNA 3.5 port builds on.

**AMD** for the gfx1151 nightly wheels (`rocm.nightlies.amd.com/v2/gfx1151`) that make a
from-source HIP build possible on Ubuntu 26.04 without a ROCm apt repo.

## License

MIT for the scripts and notes in this repo (see `LICENSE`). Weights are **not** redistributed
here; pull them from Hugging Face and respect turboderp's pack license. ExLlamaV3 and the
AMD fork have their own licenses.
