#!/usr/bin/env python3
"""Batched Endorsement-Weighted Consensus (EWC) — for large samples.

Same method as ces.py --hybrid:
  discrete answers -> hard agreement (majority vote), endorsement breaks ties
  free-form        -> most ensemble-endorsed full answer
but the endorsement log-probs are computed in right-padded, length-sorted batches
(teacher forcing), and discrete prompts that already have a >=2 vote majority skip
GPU scoring entirely. ~10-20x faster than ces.py on 10k+ sets.

Reuses per-model generations from run_single_batched.py (index-aligned).
"""
import argparse
import collections
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
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(device).eval()
    return tok, m, device


def templ_ids(tok, body):
    e = tok.apply_chat_template([{"role": "user", "content": body}], add_generation_prompt=True)
    return e["input_ids"] if hasattr(e, "keys") else e


@torch.no_grad()
def score_pairs(tm, bodies, answers, batch=16):
    """Length-normalized logprob of each answer given its body, batched."""
    tok, model, device = tm
    tok.padding_side = "right"
    res = [0.0] * len(bodies)
    order = sorted(range(len(bodies)), key=lambda k: len(bodies[k]) + len(answers[k]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        seqs, starts, alens = [], [], []
        for k in idx:
            p = templ_ids(tok, bodies[k])
            a = tok(answers[k], add_special_tokens=False).input_ids if answers[k].strip() else []
            seqs.append((p + a) if a else (p + [tok.pad_token_id]))
            starts.append(len(p))
            alens.append(len(a))
        enc = tok.pad({"input_ids": seqs}, return_tensors="pt")
        ii = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        logp = torch.log_softmax(model(ii, attention_mask=am).logits.float(), dim=-1)
        for b, k in enumerate(idx):
            l = alens[b]
            if l == 0:
                res[k] = -1e9
                continue
            st = starts[b]
            tgt = ii[b, st:st + l]
            pr = logp[b, st - 1:st + l - 1]
            res[k] = pr.gather(1, tgt[:, None]).squeeze(1).mean().item()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_sheaf.yaml")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--devices", default="cuda:2,cuda:3,cuda:7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    devs = args.devices.split(",")
    items = [json.loads(l) for l in open(args.sample) if l.strip()]
    N = len(items)
    cand_data = [json.load(open(c)) for c in args.cands]
    bodies = [build_prompt(it)[0] for it in items]
    cands = [[cd["results"][i]["gen"] for cd in cand_data] for i in range(N)]

    # decide which prompts need GPU endorsement (openqa + discrete ties)
    disc_major = [None] * N
    need = []
    for i, it in enumerate(items):
        if it["type"] in ("mc", "number"):
            preds = [scoring.predict_item(it, cands[i][j]) for j in range(len(cands[i]))]
            valid = [p for p in preds if p is not None]
            cnt = collections.Counter(valid)
            if valid and max(cnt.values()) >= 2:
                disc_major[i] = cnt.most_common(1)[0][0]
                continue
        for j in range(len(cands[i])):
            need.append((i, j))
    print(f"{N} prompts | {len(need)//3 if need else 0} need endorsement "
          f"| {sum(x is not None for x in disc_major)} settled by majority", flush=True)

    endorse = [[0.0] * len(cand_data) for _ in range(N)]
    t0 = time.time()
    print(f"loading {len(cfg['models'])} scorers on {devs} ...", flush=True)
    for m_idx, (name, dev) in enumerate(zip(cfg["models"], devs)):
        tm = load(name, dev)
        lp = score_pairs(tm, [bodies[i] for i, j in need], [cands[i][j] for i, j in need], args.batch)
        for (i, j), v in zip(need, lp):
            endorse[i][j] += v
        del tm
        torch.cuda.empty_cache()
        print(f"  scorer {m_idx+1}/{len(cfg['models'])} done ({time.time()-t0:.0f}s)", flush=True)

    correct, by, results = 0, {}, []
    for i, it in enumerate(items):
        if disc_major[i] is not None:
            ok = scoring.score_prediction(it, disc_major[i])
            chosen = -1
        elif it["type"] in ("mc", "number"):
            agg = {}
            for j in range(len(cands[i])):
                p = scoring.predict_item(it, cands[i][j])
                if p is None:
                    continue
                c, e = agg.get(p, (0, -1e18))
                agg[p] = (c + 1, max(e, endorse[i][j]))
            win = max(agg, key=lambda p: agg[p]) if agg else None
            ok = scoring.score_prediction(it, win)
            chosen = -1
        else:
            chosen = max(range(len(cands[i])), key=lambda j: endorse[i][j])
            ok = scoring.score_item(it, cands[i][chosen])
        correct += bool(ok)
        d = by.setdefault(it["task"], [0, 0])
        d[0] += bool(ok)
        d[1] += 1
        results.append({"item": it, "correct": bool(ok), "chosen": chosen})

    acc = correct / N * 100
    summary = {"method": "EWC-batched", "accuracy": acc, "correct": correct, "total": N,
               "by_task": {k: {"correct": v[0], "total": v[1]} for k, v in by.items()},
               "seconds": round(time.time() - t0, 1)}
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=2)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
