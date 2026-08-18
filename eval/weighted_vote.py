#!/usr/bin/env python3
"""Reliability-weighted majority voting baseline vs CES (dev-split, no test peeking).

On a random dev half, estimate each model's per-answer-type accuracy; use it as its vote
weight. Discrete: weighted majority over extracted answers. Open-QA (voting ill-defined on
free text): defer to the most-reliable model. Averaged over 20 seeds.
"""
import collections, json, random, statistics, sys
sys.path.insert(0, "eval")
import scoring

CLS = lambda t: "disc" if t in ("mc", "number") else "oq"


def run(tag, cand_files, ces_file, seeds=range(20)):
    D = [json.load(open("eval/out/" + f)) for f in cand_files]
    E = json.load(open("eval/out/" + ces_file))
    n, M = len(D[0]["results"]), len(cand_files)
    items = [D[0]["results"][i]["item"] for i in range(n)]
    gens = [[D[m]["results"][i]["gen"] for m in range(M)] for i in range(n)]
    preds = [[scoring.predict_item(items[i], gens[i][m]) for m in range(M)] for i in range(n)]
    ces = sum(bool(E["results"][i]["correct"]) for i in range(n)) / n * 100

    def weights(dev):
        acc = collections.defaultdict(lambda: [[0, 0] for _ in range(M)])
        for i in dev:
            c = CLS(items[i]["type"])
            for m in range(M):
                ok = (scoring.score_prediction(items[i], preds[i][m]) if c == "disc"
                      else scoring.score_openqa(items[i], gens[i][m]))
                acc[c][m][0] += ok
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
            it = items[i]
            if CLS(it["type"]) == "disc":
                sc = collections.defaultdict(float)
                for m in range(M):
                    if preds[i][m] is not None:
                        sc[preds[i][m]] += w["disc"][m]
                if sc:
                    c += scoring.score_prediction(it, max(sc, key=lambda k: sc[k]))
            else:
                c += scoring.score_openqa(it, gens[i][best_oq])
        accs.append(c / len(test) * 100)
    print(f"{tag}: CES={ces:.2f}  weighted-vote[dev-split, {len(list(seeds))} seeds]="
          f"{statistics.mean(accs):.2f} (+/-{statistics.pstdev(accs):.2f})")


run("2023-era", ["full_mistral.json", "full_llama.json", "full_yi.json"], "ewc_full.json")
run("modern  ", ["mod_qwen.json", "mod_llama31.json", "mod_mistral.json"], "ces_modern2.json")
