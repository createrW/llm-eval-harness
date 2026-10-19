#!/usr/bin/env python
"""Quick decode/scoring throughput calibration for sizing cloud GPU time (docs/cloud_compute_options.md).

Not part of the harness: it uses bf16 and random token ids on purpose. Run with `uv run --extra local python scripts/bench_decode.py`
on the laptop (MPS) or a rented GPU (set DEV=cuda) and compare tok/s.
"""

import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

dev = os.environ.get(
    "DEV",
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
)
name = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(dev).eval()


def sync():
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()
    return


def gen(bs, prompt_len, new):
    ids = torch.randint(1000, 30000, (bs, prompt_len), device=dev)
    att = torch.ones_like(ids)
    sync()
    t = time.time()
    with torch.inference_mode():
        out = m.generate(
            input_ids=ids,
            attention_mask=att,
            max_new_tokens=new,
            min_new_tokens=new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    sync()
    dt = time.time() - t
    n = (out.shape[1] - prompt_len) * bs
    return n, dt


gen(1, 32, 8)  # warmup
for bs, pl, new in [(1, 256, 128), (8, 256, 128), (32, 256, 128)]:
    n, dt = gen(bs, pl, new)
    print(
        f"generate bs={bs} prompt={pl} new={new}: {n} tok in {dt:.1f}s -> {n / dt:.0f} tok/s aggregate, {dt / new * 1000:.0f} ms/step"
    )
# HellaSwag-style scoring: forward passes over ~120-token sequences
for bs in (1, 16):
    x = torch.randint(1000, 30000, (bs, 120), device=dev)
    with torch.inference_mode():
        m(input_ids=x)
        sync()
        t = time.time()
        for _ in range(10):
            torch.log_softmax(m(input_ids=x).logits.float(), -1)
            sync()
    dt = (time.time() - t) / 10
    print(f"score bs={bs} len=120: {dt * 1000:.0f} ms/forward -> {bs / dt:.0f} seq/s")
