#!/usr/bin/env python
"""Batched generation for Qwen3.8-Flash-Next on Strix Halo.

Why this exists: on this GPU a single stream is latency-limited near 47 tok/s, but
~32 ms of every verification forward is row-independent (dense weights + mixers are
read once no matter how many token positions ride along). Concurrent sequences share
that cost, so aggregate throughput keeps climbing well past the single-stream number:

    batch 1 -> 47 tok/s     batch 3 -> 70     batch 4 -> 79     batch 5 -> 86

Per-stream latency falls as batch rises, so this is the right tool for agent fan-out,
offline batch jobs and eval sweeps -- not for one interactive chat (use run.sh).

    bash scripts/batch.sh -f prompts.txt
    bash scripts/batch.sh -p "Explain X" -p "Summarize Y" -b 4
    bash scripts/batch.sh -f prompts.txt -o out.jsonl --raw --max-tokens 1024
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.expanduser(os.environ.get("REPO_DIR", "~/exllamav3-amd")))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler, ComboSampler

ap = argparse.ArgumentParser(description="Batched generation (aggregate-throughput mode)")
ap.add_argument("-m", "--model", default=os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
ap.add_argument("-f", "--file", help="prompt file, one prompt per line (blank lines skipped)")
ap.add_argument("-p", "--prompt", action="append", default=[], help="a prompt; repeatable")
ap.add_argument("-b", "--batch", type=int, default=5,
                help="max concurrent sequences (default 5 = 15 rows at ndt=2, exactly one "
                     "16-row MoE chunk; 6 costs 24%% -- see the 16-row rule in batch.sh)")
ap.add_argument("-o", "--out", help="write results as JSONL to this path")
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("-cs", "--cache", type=int, default=16384, help="cache tokens per generator")
ap.add_argument("-cq", "--cache-quant", type=int, default=4,
                help="KV cache bits (4 recommended; 0 = fp16)")
ap.add_argument("-ndt", type=int, default=2, help="MTP draft tokens (2 is the measured best)")
ap.add_argument("--dc", type=float, default=None,
                help="draft confidence -> enables dynamic draft sizing; omit for static")
ap.add_argument("--raw", action="store_true", help="no chat template, raw completion")
ap.add_argument("--think", action="store_true", help="allow the model's thinking block")
ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy (default, reproducible)")
args = ap.parse_args()

prompts = list(args.prompt)
if args.file:
    with open(os.path.expanduser(args.file)) as fh:
        prompts += [ln.strip() for ln in fh if ln.strip()]
if not prompts:
    ap.error("no prompts: pass -f FILE or -p PROMPT")

config = Config.from_directory(os.path.expanduser(args.model))
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)

cache_kw = {}
if args.cache_quant:
    from exllamav3.cache import CacheLayer_quant
    cache_kw = dict(layer_type=CacheLayer_quant, k_bits=args.cache_quant, v_bits=args.cache_quant)

t0 = time.time()
cache = Cache(model, max_num_tokens=args.cache, max_history=args.ndt, **cache_kw)
model.load(progressbar=False)
draft = Model.from_config(config, component="mtp")
dcache = Cache(draft, max_num_tokens=args.cache, max_history=args.ndt, **cache_kw)
draft.load(progressbar=False)

gen_kw = dict(num_draft_tokens=args.ndt)
if args.dc is not None:
    gen_kw.update(dynamic_draft_tokens=True, draft_confidence=args.dc)
gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                draft_model=draft, draft_cache=dcache,
                max_batch_size=max(args.batch, 1), **gen_kw)

sampler = GreedySampler() if args.temp <= 0 else ComboSampler(temperature=args.temp)

# Same prompt format run.sh uses (-mode qwen35), implemented inline so this needs no
# `transformers` install. Reasoning-aware ChatML; add_bos is False for this format.
SYSTEM = os.environ.get("SYSTEM_PROMPT", "You are a helpful AI assistant.")


def encode(p):
    if args.raw:
        return tokenizer.encode(p, add_bos=True)
    ctx = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
           f"<|im_start|>user\n{p}<|im_end|>\n"
           f"<|im_start|>assistant\n")
    if not args.think:
        # Pre-close the thinking block, exactly as chat_templates.PromptFormat_qwen35 does
        # when thinking is off. Without this the model emits a <think> monologue and a
        # batch job spends most of its token budget on it.
        ctx += "<think>\n\n</think>\n\n"
    return tokenizer.encode(ctx, add_bos=False, encode_special_tokens=True)


# Without stop conditions every sequence runs to max_tokens, which wastes the budget and
# leaves trailing turn markers in the output.
STOP = [tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>"), "<|im_end|>"]
stop_conditions = None if args.raw else STOP


print(f"loaded in {time.time()-t0:.1f}s | {len(prompts)} prompts | batch<={args.batch} | "
      f"ndt={args.ndt}{'' if args.dc is None else f' dc={args.dc}'} | "
      f"cache={args.cache}{'' if not args.cache_quant else f' q{args.cache_quant}'} | "
      f"{'greedy' if args.temp <= 0 else f'temp={args.temp}'}", flush=True)

results = {}
for i, p in enumerate(prompts):
    results[i] = {"prompt": p, "text": "", "tokens": 0}
    job_kw = {} if stop_conditions is None else {"stop_conditions": stop_conditions}
    gen.enqueue(Job(input_ids=encode(p), max_new_tokens=args.max_tokens,
                    sampler=sampler, identifier=i, **job_kw))

acc = rej = 0
errors = []
t0 = time.time()
while gen.num_remaining_jobs():
    for r in gen.iterate():
        idx = r.get("identifier")
        if r.get("error"):
            errors.append((idx, r["error"]))
            continue
        if idx is None or idx not in results:
            continue
        if r.get("text"):
            results[idx]["text"] += r["text"]
        if r.get("eos"):
            # authoritative per-job totals arrive once, at eos
            results[idx]["tokens"] = r.get("new_tokens") or results[idx]["tokens"]
            results[idx]["eos_reason"] = r.get("eos_reason")
            acc += r.get("accepted_draft_tokens") or 0
            rej += r.get("rejected_draft_tokens") or 0
torch.cuda.synchronize()
wall = time.time() - t0

total = sum(v["tokens"] for v in results.values())
print(f"\n{len(prompts)} sequences, {total} tokens in {wall:.2f}s")
print(f"aggregate      {total/wall:.1f} tok/s")
print(f"per sequence   {total/len(prompts)/wall:.1f} tok/s  (latency view)")
if acc + rej:
    print(f"draft accept   {100*acc/(acc+rej):.1f}%  ({acc} accepted / {acc+rej} proposed)")
print(f"peak GPU       {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
for idx, err in errors:
    print(f"[error] seq {idx}: {err}", file=sys.stderr)

if args.out:
    with open(os.path.expanduser(args.out), "w") as fh:
        for i in sorted(results):
            fh.write(json.dumps({"index": i, **results[i]}) + "\n")
    print(f"wrote {args.out}")
else:
    for i in sorted(results):
        print(f"\n--- [{i}] {results[i]['prompt'][:70]}\n{results[i]['text'].strip()}")

sys.exit(1 if errors else 0)
