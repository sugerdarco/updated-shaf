#!/usr/bin/env python3
"""Single-model baseline on the mixed 100-prompt sample.

This establishes the number the SAHF ensemble must beat: each model is run alone,
greedy-decoded, with its own chat template (its realistic best), and scored by the
same unified scorer the ensemble uses.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring
from prompts import build_prompt


def encode(tok, body, chat, device):
    if chat and getattr(tok, "chat_template", None):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": body}],
            add_generation_prompt=True, return_tensors="pt",
        )
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    else:
        ids = tok(body, return_tensors="pt").input_ids
    return ids.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sample", default="eval/sample_100.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-chat", action="store_true", help="raw prompt, no chat template")
    args = ap.parse_args()

    chat = not args.no_chat
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to(args.device).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    items = [json.loads(l) for l in open(args.sample) if l.strip()]
    results, correct, by_task = [], 0, {}
    t0 = time.time()
    for i, it in enumerate(items):
        body, max_new = build_prompt(it)
        ids = encode(tok, body, chat, args.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        ok = bool(scoring.score_item(it, gen))
        correct += ok
        d = by_task.setdefault(it["task"], [0, 0])
        d[0] += ok
        d[1] += 1
        results.append({"item": it, "correct": ok, "gen": gen})
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(items)}] running acc={correct/(i+1)*100:.1f}%", flush=True)

    acc = correct / len(items) * 100
    summary = {"model": args.model, "chat": chat, "accuracy": acc, "correct": correct,
               "total": len(items),
               "by_task": {k: {"correct": v[0], "total": v[1]} for k, v in by_task.items()},
               "seconds": round(time.time() - t0, 1)}
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=2)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
