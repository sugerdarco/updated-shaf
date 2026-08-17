#!/usr/bin/env python3
"""Batched single-model baseline / candidate generator.

Same output as run_single.py (persists item + full generation, in the sample's
ORIGINAL order so the three models stay index-aligned for EWC), but generates in
left-padded batches grouped by answer type — ~5-10x faster on GPU. Greedy decoding,
so results match the unbatched runner up to negligible numerical noise.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring
from prompts import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only: pad on the left so the last token aligns
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()

    items = [json.loads(l) for l in open(args.sample) if l.strip()]
    N = len(items)
    gens = [""] * N
    bodies, maxnew = [None] * N, [0] * N
    groups = defaultdict(list)
    for i, it in enumerate(items):
        b, mn = build_prompt(it)
        bodies[i], maxnew[i] = b, mn
        groups[mn].append(i)

    t0, done = time.time(), 0
    for mn, idxs in sorted(groups.items()):
        idxs.sort(key=lambda i: len(bodies[i]))  # length-homogeneous batches -> minimal padding
        bs = args.batch if mn <= 64 else max(8, args.batch // 2)  # long gens -> smaller batch
        for s in range(0, len(idxs), bs):
            chunk = idxs[s:s + bs]
            batch_ids = []
            for i in chunk:
                e = tok.apply_chat_template([{"role": "user", "content": bodies[i]}],
                                            add_generation_prompt=True)
                batch_ids.append(e["input_ids"] if hasattr(e, "keys") else e)
            enc = tok.pad({"input_ids": batch_ids}, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"].to(args.device)
            attn = enc["attention_mask"].to(args.device)
            with torch.no_grad():
                out = model.generate(input_ids, attention_mask=attn, max_new_tokens=mn,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
            for j, i in enumerate(chunk):
                gens[i] = texts[j]
            done += len(chunk)
        print(f"[{done}/{N}] type-budget {mn} done ({time.time()-t0:.0f}s)", flush=True)

    correct, by, results = 0, {}, []
    for i, it in enumerate(items):
        ok = bool(scoring.score_item(it, gens[i]))
        correct += ok
        d = by.setdefault(it["task"], [0, 0])
        d[0] += ok
        d[1] += 1
        results.append({"item": it, "correct": ok, "gen": gens[i]})
    acc = correct / N * 100
    summary = {"model": args.model, "batched": True, "accuracy": acc, "correct": correct,
               "total": N, "by_task": {k: {"correct": v[0], "total": v[1]} for k, v in by.items()},
               "seconds": round(time.time() - t0, 1)}
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=2)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
