#!/usr/bin/env python3
"""Answer the reviewer: voting vs EWC at full scale + McNemar significance."""
import collections
import json
import sys
from math import erfc, sqrt

sys.path.insert(0, "eval")
import scoring

files = ["eval/out/full_mistral.json", "eval/out/full_llama.json", "eval/out/full_yi.json"]
D = [json.load(open(f)) for f in files]
E = json.load(open("eval/out/ewc_full.json"))
n = len(D[0]["results"])
items = [D[0]["results"][i]["item"] for i in range(n)]

single_ok = [[bool(scoring.score_item(items[i], D[m]["results"][i]["gen"])) for i in range(n)]
             for m in range(3)]
single_acc = [sum(c) / n * 100 for c in single_ok]
best_m = max(range(3), key=lambda m: single_acc[m])
best_ok = single_ok[best_m]


def vote_choice(preds):
    valid = [p for p in preds if p is not None]
    if not valid:
        return None
    cnt = collections.Counter(valid)
    best = max(cnt.values())
    win = {k for k, c in cnt.items() if c == best}
    for p in valid:
        if p in win:
            return p
    return valid[0]


def openqa_idx(shorts):
    cnt = collections.Counter(s for s in shorts if s)
    if cnt and max(cnt.values()) >= 2:
        win = {k for k, c in cnt.items() if c == max(cnt.values())}
        for idx, s in enumerate(shorts):
            if s in win:
                return idx
    return 0


vote_ok, byv = [], {}
for i in range(n):
    it = items[i]
    gens = [D[m]["results"][i]["gen"] for m in range(3)]
    if it["type"] in ("mc", "number"):
        preds = [scoring.predict_item(it, g) for g in gens]
        ok = bool(scoring.score_prediction(it, vote_choice(preds)))
    else:
        shorts = [scoring._norm(scoring.predict_openqa(it, g)) for g in gens]
        ok = bool(scoring.score_openqa(it, gens[openqa_idx(shorts)]))
    vote_ok.append(ok)
    d = byv.setdefault(it["task"], [0, 0]); d[0] += ok; d[1] += 1

vote_acc = sum(vote_ok) / n * 100
ewc_ok = [bool(r["correct"]) for r in E["results"]]
ewc_acc = sum(ewc_ok) / n * 100


def mcnemar(a, b):
    b_ = sum(1 for i in range(n) if a[i] and not b[i])
    c_ = sum(1 for i in range(n) if b[i] and not a[i])
    stat = (abs(b_ - c_) - 1) ** 2 / (b_ + c_) if (b_ + c_) else 0.0
    p = erfc(sqrt(stat / 2)) if stat > 0 else 1.0
    return b_, c_, stat, p


print(f"n = {n}")
print(f"single accs   : mistral={single_acc[0]:.2f} llama={single_acc[1]:.2f} yi={single_acc[2]:.2f}")
print(f"BEST single   : model{best_m} = {single_acc[best_m]:.2f}%")
print(f"VOTING (40k)  : {vote_acc:.2f}%   " + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(byv.items())))
print(f"EWC (40k)     : {ewc_acc:.2f}%")
print()
b_, c_, s, p = mcnemar(ewc_ok, best_ok)
print(f"EWC vs BEST-single : EWC-only-right={b_}  best-only-right={c_}  chi2={s:.2f}  p={p:.2e}")
b_, c_, s, p = mcnemar(ewc_ok, vote_ok)
print(f"EWC vs VOTING      : EWC-only-right={b_}  vote-only-right={c_}  chi2={s:.2f}  p={p:.4f}")
print(f"  (delta EWC-voting = {ewc_acc-vote_acc:+.2f}pp = {sum(ewc_ok)-sum(vote_ok):+d} items)")
