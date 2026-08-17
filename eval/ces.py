#!/usr/bin/env python3
"""Cross-Endorsement Selection (CES) — novel answer-level reconciliation.

Each agent answers independently (clean, chat-templated generation — no byte-bloat).
Every candidate answer is then re-scored by the WHOLE ensemble: agent i's
length-normalized log-probability of candidate j, given agent i's own template and
tokenizer. The candidate with the highest total endorsement (sum over agents) wins.

An answer that several heterogeneous models independently find likely is more
trustworthy than any single model's own pick, so this exploits their
complementarity (oracle ceiling ~82%) without ever averaging distributions and
without peeking at gold. It is distinct from DeePEn (relative-representation
averaging) and from plain majority voting (which ignores cross-model endorsement).

Candidates are reused from run_single.py outputs; only the scoring passes run here.
"""
import argparse
import json
import os
import sys
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring
from prompts import build_prompt


def load(model, device):
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(device).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, m, device


def templ_ids(tok, body):
    enc = tok.apply_chat_template([{"role": "user", "content": body}],
                                  add_generation_prompt=True, return_tensors="pt")
    return enc["input_ids"] if hasattr(enc, "keys") else enc


@torch.no_grad()
def logprob(tm, body, answer):
    """Length-normalized logprob of `answer` given `body`, under one model."""
    tok, model, device = tm
    if not answer.strip():
        return -1e9
    p = templ_ids(tok, body).to(device)
    a = tok(answer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if a.shape[1] == 0:
        return -1e9
    ids = torch.cat([p, a], dim=1)
    logits = model(ids).logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    start = p.shape[1]
    tgt = ids[0, start:]
    pr = logp[start - 1: ids.shape[1] - 1]
    return float(pr.gather(1, tgt[:, None]).squeeze(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_sheaf.yaml")
    ap.add_argument("--sample", default="eval/sample_100.jsonl")
    ap.add_argument("--cands", nargs="+", required=True, help="run_single.py output JSONs")
    ap.add_argument("--devices", default="cuda:2,cuda:3,cuda:7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hybrid", action="store_true",
                    help="Endorsement-Weighted Consensus: hard agreement (vote) for "
                         "discrete answers, soft likelihood endorsement for free-form")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    devs = args.devices.split(",")
    print(f"Loading {len(cfg['models'])} scorers on {devs} ...", flush=True)
    tms = [load(n, d) for n, d in zip(cfg["models"], devs)]
    cand_data = [json.load(open(c)) for c in args.cands]
    items = [json.loads(l) for l in open(args.sample) if l.strip()]

    import collections

    def majority(preds):
        valid = [p for p in preds if p is not None]
        if not valid:
            return None
        counts = collections.Counter(valid)
        best = max(counts.values())
        winners = {k for k, c in counts.items() if c == best}
        for p in valid:  # candidate-file order = priority for ties
            if p in winners:
                return p
        return valid[0]

    correct, by, results = 0, {}, []
    t0 = time.time()
    for i, it in enumerate(items):
        body, _ = build_prompt(it)
        cands = [cd["results"][i]["gen"] for cd in cand_data]
        endorse = [sum(logprob(tm, body, c) for tm in tms) for c in cands]
        if args.hybrid and it["type"] in ("mc", "number"):
            # discrete: aggregate endorsement by the explicit predicted answer.
            # Primary key = how many agents predict it (hard agreement / vote),
            # tie-break = best single-candidate endorsement for that answer.
            agg = {}
            for j, p in enumerate([scoring.predict_item(it, c) for c in cands]):
                if p is None:
                    continue
                cnt, es = agg.get(p, (0, -1e18))
                agg[p] = (cnt + 1, max(es, endorse[j]))
            winner = max(agg, key=lambda p: agg[p]) if agg else None
            ok = bool(scoring.score_prediction(it, winner))
            best = -1
        else:
            # free-form: pick the most ensemble-endorsed full answer
            best = max(range(len(cands)), key=lambda j: endorse[j])
            ok = bool(scoring.score_item(it, cands[best]))
        correct += ok
        d = by.setdefault(it["task"], [0, 0])
        d[0] += ok
        d[1] += 1
        results.append({"item": it, "chosen": best, "correct": ok,
                        "gen": cands[best] if best >= 0 else "(vote)"})
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(items)}] acc={correct/(i+1)*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    acc = correct / len(items) * 100
    summary = {"method": "CES", "accuracy": acc, "correct": correct, "total": len(items),
               "by_task": {k: {"correct": v[0], "total": v[1]} for k, v in by.items()},
               "seconds": round(time.time() - t0, 1)}
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=2)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
