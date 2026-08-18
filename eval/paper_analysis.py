#!/usr/bin/env python3
"""All the numbers the paper needs, from the saved 40k generations (CPU only):
baselines, voting, CES, oracle, branch decomposition, McNemar significance, and the
open-QA length-confound ablation."""
import argparse
import collections
import json
import sys
from math import erfc, sqrt

sys.path.insert(0, "eval")
import scoring

_ap = argparse.ArgumentParser()
_ap.add_argument("--cands", nargs=3,
                 default=["eval/out/full_mistral.json", "eval/out/full_llama.json", "eval/out/full_yi.json"])
_ap.add_argument("--ces", default="eval/out/ewc_full.json")
_ap.add_argument("--names", default="mistral,llama,yi")
_a = _ap.parse_args()
_names = _a.names.split(",")

D = [json.load(open(f)) for f in _a.cands]
E = json.load(open(_a.ces))
n = len(D[0]["results"])
items = [D[0]["results"][i]["item"] for i in range(n)]
gens = [[D[m]["results"][i]["gen"] for m in range(3)] for i in range(n)]


def mcnemar(a, b):
    b_ = sum(1 for i in range(n) if a[i] and not b[i])
    c_ = sum(1 for i in range(n) if b[i] and not a[i])
    stat = (abs(b_ - c_) - 1) ** 2 / (b_ + c_) if (b_ + c_) else 0.0
    return b_, c_, stat, (erfc(sqrt(stat / 2)) if stat > 0 else 1.0)


# ---- per-model + best single ----
single_ok = [[bool(scoring.score_item(items[i], gens[i][m])) for i in range(n)] for m in range(3)]
single_acc = [sum(c) / n * 100 for c in single_ok]
best_m = max(range(3), key=lambda m: single_acc[m])
best_ok = single_ok[best_m]

# ---- voting (discrete + open-QA priority pick) and CES (=EWC) ----
def vote_choice(preds):
    valid = [p for p in preds if p is not None]
    if not valid:
        return None
    cnt = collections.Counter(valid)
    win = {k for k, c in cnt.items() if c == max(cnt.values())}
    return next(p for p in valid if p in win)


vote_ok = []
for i in range(n):
    it = items[i]
    if it["type"] in ("mc", "number"):
        vote_ok.append(bool(scoring.score_prediction(it, vote_choice([scoring.predict_item(it, g) for g in gens[i]]))))
    else:
        shorts = [scoring._norm(scoring.predict_openqa(it, g)) for g in gens[i]]
        cnt = collections.Counter(s for s in shorts if s)
        idx = 0
        if cnt and max(cnt.values()) >= 2:
            win = {k for k, c in cnt.items() if c == max(cnt.values())}
            idx = next(j for j, s in enumerate(shorts) if s in win)
        vote_ok.append(bool(scoring.score_openqa(it, gens[i][idx])))
ces_ok = [bool(r["correct"]) for r in E["results"]]

print(f"n={n}")
print("single: " + " ".join(f"{_names[m]}={single_acc[m]:.2f}" for m in range(3)) +
      f" | best={_names[best_m]} {single_acc[best_m]:.2f}")
print(f"voting={sum(vote_ok)/n*100:.2f}  CES={sum(ces_ok)/n*100:.2f}  oracle={sum(any(single_ok[m][i] for m in range(3)) for i in range(n))/n*100:.2f}")
print("--- significance ---")
for name, a in (("CES vs best", best_ok), ("CES vs voting", vote_ok)):
    b_, c_, s, p = mcnemar(ces_ok, a)
    print(f"{name:14s}: CES+={b_} other+={c_} chi2={s:.2f} p={p:.2e}")

# ---- decomposition: gain vs best single, split by branch ----
disc = [i for i in range(n) if items[i]["type"] in ("mc", "number")]
oq = [i for i in range(n) if items[i]["type"] == "openqa"]
print("--- decomposition (items CES-correct minus best-correct) ---")
print(f"discrete: voting_gain={sum(vote_ok[i]-best_ok[i] for i in disc):+d}  ces_gain={sum(ces_ok[i]-best_ok[i] for i in disc):+d}")
print(f"open-QA : voting_gain={sum(vote_ok[i]-best_ok[i] for i in oq):+d}  ces_gain={sum(ces_ok[i]-best_ok[i] for i in oq):+d}  (CES-over-voting on oq = {sum(ces_ok[i]-vote_ok[i] for i in oq):+d})")

# ---- length-confound ablation on open-QA ----
def sel(i, key):
    return bool(scoring.score_openqa(items[i], gens[i][max(range(3), key=lambda j: key(gens[i][j]))]))


wc = lambda s: len(s.split())
end_ok = {i: bool(scoring.score_openqa(items[i], gens[i][E["results"][i]["chosen"]])) if E["results"][i]["chosen"] >= 0 else ces_ok[i] for i in oq}
long_acc = sum(sel(i, wc) for i in oq) / len(oq) * 100
short_acc = sum(sel(i, lambda s: -wc(s)) for i in oq) / len(oq) * 100
end_acc = sum(end_ok[i] for i in oq) / len(oq) * 100
sm_acc = [sum(scoring.score_openqa(items[i], gens[i][m]) for i in oq) / len(oq) * 100 for m in range(3)]
chosen_len = [wc(gens[i][E["results"][i]["chosen"]]) for i in oq if E["results"][i]["chosen"] >= 0]
cand_len = [wc(gens[i][j]) for i in oq for j in range(3)]
print("--- open-QA length-confound ablation (n_oq={}) ---".format(len(oq)))
print(f"per-model oq acc: {[round(x,2) for x in sm_acc]}")
print(f"select longest  : {long_acc:.2f}   select shortest: {short_acc:.2f}")
print(f"endorsement sel : {end_acc:.2f}")
print(f"mean words: chosen={sum(chosen_len)/len(chosen_len):.1f}  all-candidates={sum(cand_len)/len(cand_len):.1f}")
