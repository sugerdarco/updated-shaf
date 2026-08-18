#!/usr/bin/env python3
"""Reliability-weighted majority voting vs CES, fully from the shipped export
(eval/results_export/*.jsonl with per-model 'pred' + 'correct'), so it is independently
verifiable — no gitignored generations needed.

Dev-split (20 seeds): on a random half, weight each model by its per-answer-type accuracy.
Discrete: weighted majority over extracted answers. Open-QA (voting ill-defined on free text):
defer to the most-reliable model (its 'correct' flag).
"""
import collections, json, random, statistics, sys

CLS = lambda t: "disc" if t in ("mc", "number") else "oq"


def run(tag, path, seeds=range(20)):
    rows = [json.loads(l) for l in open(path)]
    n, M = len(rows), len(rows[0]["correct"])
    ces = sum(r["ces"] for r in rows) / n * 100

    def weights(dev):
        acc = collections.defaultdict(lambda: [[0, 0] for _ in range(M)])
        for i in dev:
            c = CLS(rows[i]["type"])
            for m in range(M):
                acc[c][m][0] += rows[i]["correct"][m]
                acc[c][m][1] += 1
        return {c: [acc[c][m][0] / max(1, acc[c][m][1]) for m in range(M)] for c in ("disc", "oq")}

    accs = []
    for seed in seeds:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        dev, test = set(idx[: n // 2]), idx[n // 2:]
        w = weights(dev)
        best_oq = max(range(M), key=lambda m: w["oq"][m])
        c = 0
        for i in test:
            r = rows[i]
            if CLS(r["type"]) == "disc":
                sc = collections.defaultdict(float)
                for m in range(M):
                    p = r["pred"][m]
                    if p is not None:
                        sc[str(p)] += w["disc"][m]
                if sc:
                    win = max(sc, key=lambda k: sc[k])
                    # a model is correct iff its own pred won and that model was correct
                    c += any(str(r["pred"][m]) == win and r["correct"][m] for m in range(M))
            else:
                c += r["correct"][best_oq]
        accs.append(c / len(test) * 100)
    print(f"{tag}: CES={ces:.2f}  weighted-vote[dev-split, {len(list(seeds))} seeds]="
          f"{statistics.mean(accs):.2f} (+/-{statistics.pstdev(accs):.2f})")


run("2023-era", sys.argv[1])
run("modern  ", sys.argv[2])
