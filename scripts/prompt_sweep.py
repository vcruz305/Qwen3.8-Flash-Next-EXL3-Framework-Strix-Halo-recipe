#!/usr/bin/env python
"""Prompt-averaged MTP throughput. Speculative decode is content-dependent:
predictable text accepts more drafts, so a single prompt can over- or
under-state the real figure. Runs a fixed set and reports the distribution.
"""
import os, subprocess, sys, re, statistics
from pathlib import Path

REPO = Path(os.path.expanduser("~/exllamav3-amd"))
PROMPTS = [
    "Explain gradient descent in two sentences:",
    "Write a short technical summary of how GPUs execute matrix multiplication:",
    "List the first ten prime numbers and explain what makes a number prime:",
    "Describe the difference between a process and a thread:",
    "Write a Python function that reverses a linked list, with comments:",
    "Summarize the causes of the 1929 stock market crash:",
]
N = int(os.environ.get("NTOK", "512"))

env = os.environ.copy()
env.setdefault("EXL3_MOE_CFG", "2")
env.setdefault("EXL3_HIP_PREFILL_MIN_ROWS", "2")

rates, accs = [], []
print(f"{'tok/s':>8} {'accept':>8}  prompt")
print("-" * 78)
for p in PROMPTS:
    r = subprocess.run(
        [str(REPO / ".venv/bin/python"), str(REPO / "bench_mtp.py"),
         "-n", str(N), "-ndt", "2", "-dds", "-g", "-p", p],
        capture_output=True, text=True, env=env, timeout=1800)
    out = r.stdout + r.stderr
    m = re.search(r"decode:\s+([0-9.]+) tok/s", out)
    a = re.search(r"acceptance=([0-9.]+)%", out)
    if m:
        rates.append(float(m.group(1)))
        accs.append(float(a.group(1)) if a else 0.0)
        print(f"{rates[-1]:>8.2f} {accs[-1]:>7.1f}%  {p[:56]}")
    else:
        print(f"{'FAIL':>8} {'':>8}  {p[:56]}")

if rates:
    print("-" * 78)
    print(f"mean   {statistics.mean(rates):.2f} tok/s   "
          f"median {statistics.median(rates):.2f}   "
          f"min {min(rates):.2f}   max {max(rates):.2f}")
    print(f"mean acceptance {statistics.mean(accs):.1f}%")
