#!/usr/bin/env python3
"""Paired bootstrap 95% CIs from the shipped per-prompt correctness data.
Headline = CES - best-single (all prompts). Residual = endorsement - best-single-model on open-QA."""
import json, random, sys

CLS = lambda t: "disc" if t in ("mc", "number") else "oq"


def ci(deltas, B=4000, seed=0):
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(B):
        s = sum(deltas[rng.randrange(n)] for _ in range(n))
        means.append(s / n * 100)
    means.sort()
    return sum(deltas) / n * 100, means[int(0.025 * B)], means[int(0.975 * B)]


def analyze(tag, path):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    cor = [r["correct"] for r in rows]
    ces = [r["ces"] for r in rows]
    typ = [r["type"] for r in rows]
    M = len(cor[0])
    # best single overall
    single = [sum(cor[i][m] for i in range(n)) for m in range(M)]
    bm = max(range(M), key=lambda m: single[m])
    head = [ces[i] - cor[i][bm] for i in range(n)]
    d, lo, hi = ci(head)
    print(f"{tag}  headline CES-bestSingle: {d:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
    # open-QA residual
    oq = [i for i in range(n) if CLS(typ[i]) == "oq"]
    oq_single = [sum(cor[i][m] for i in oq) for m in range(M)]
    bmo = max(range(M), key=lambda m: oq_single[m])
    res = [ces[i] - cor[i][bmo] for i in oq]
    d, lo, hi = ci(res)
    print(f"{tag}  open-QA endorsement-bestModel: {d:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")


analyze("2023-era", sys.argv[1])
analyze("modern  ", sys.argv[2])
