#!/usr/bin/env python3
"""Answer-type router baseline vs CES, from the shipped per-prompt correctness data.

Router = pick ONE model for discrete (mc/number) and ONE for open-QA. Two versions:
  - dev-split: choose the per-type best model on a random half, apply to the other half
    (no task labels at test time, no test-set peeking), averaged over seeds;
  - test-optimal: choose per-type best model on all data (mildly optimistic upper bound).
"""
import collections
import json
import random
import statistics
import sys
from math import erfc, sqrt

CLS = lambda t: "disc" if t in ("mc", "number") else "oq"


def mcnemar(a, b):
    B = sum(1 for i in range(len(a)) if a[i] and not b[i])
    C = sum(1 for i in range(len(a)) if b[i] and not a[i])
    s = (abs(B - C) - 1) ** 2 / (B + C) if (B + C) else 0.0
    return B, C, (erfc(sqrt(s / 2)) if s > 0 else 1.0)


def analyze(tag, path, seeds=range(20)):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    typ = [r["type"] for r in rows]
    cor = [r["correct"] for r in rows]      # [m0, m1, m2]
    ces = [r["ces"] for r in rows]
    M = len(cor[0])
    single = [sum(cor[i][m] for i in range(n)) / n * 100 for m in range(M)]

    def best_per_type(idxs):
        acc = collections.defaultdict(lambda: [[0, 0] for _ in range(M)])
        for i in idxs:
            c = CLS(typ[i])
            for m in range(M):
                acc[c][m][0] += cor[i][m]
                acc[c][m][1] += 1
        return {c: max(range(M), key=lambda m: acc[c][m][0] / max(1, acc[c][m][1])) for c in ("disc", "oq")}

    # test-optimal (all data)
    bo = best_per_type(range(n))
    router_opt = [cor[i][bo[CLS(typ[i])]] for i in range(n)]
    opt = sum(router_opt) / n * 100
    B, C, p = mcnemar(ces, router_opt)

    # dev-split, averaged
    dev_accs = []
    for seed in seeds:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        dev, test = set(idx[: n // 2]), idx[n // 2:]
        bp = best_per_type(dev)
        dev_accs.append(sum(cor[i][bp[CLS(typ[i])]] for i in test) / len(test) * 100)

    print(f"{tag}:")
    print(f"  best_single={max(single):.2f}  CES={sum(ces)/n*100:.2f}")
    print(f"  type-router[dev-split, {len(list(seeds))} seeds]={statistics.mean(dev_accs):.2f} "
          f"(+/-{statistics.pstdev(dev_accs):.2f})")
    print(f"  type-router[test-optimal]={opt:.2f}   picks={bo}")
    print(f"  CES vs test-optimal router: CES+={B} router+={C}  p={p:.3f}  (CES-router={sum(ces)/n*100-opt:+.2f})")


analyze("2023-era  [correct=mistral,llama2,yi]", sys.argv[1])
analyze("modern    [correct=qwen,llama3.1,mistral]", sys.argv[2])
