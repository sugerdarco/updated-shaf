#!/usr/bin/env python3
"""Export a compact, independently-verifiable per-prompt correctness table for a run,
so the headline numbers can be recomputed from the repo without the full generations.

Each output line: {task, type, gold, correct: [m0, m1, m2], ces}. Recompute any number:
best-single = max over models of mean(correct[m]); CES = mean(ces); etc.
"""
import argparse, json, sys
sys.path.insert(0, "eval")
import scoring

ap = argparse.ArgumentParser()
ap.add_argument("--cands", nargs=3, required=True)
ap.add_argument("--ces", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

D = [json.load(open(f)) for f in a.cands]
E = json.load(open(a.ces))
n = len(D[0]["results"])
with open(a.out, "w") as f:
    for i in range(n):
        it = D[0]["results"][i]["item"]
        disc = it["type"] in ("mc", "number")
        rec = {
            "task": it["task"], "type": it["type"], "gold": it["gold"],
            # discrete answer each model extracted (for recomputing voting variants); null for open-QA
            "pred": [scoring.predict_item(it, D[m]["results"][i]["gen"]) if disc else None for m in range(3)],
            "correct": [int(bool(scoring.score_item(it, D[m]["results"][i]["gen"]))) for m in range(3)],
            "ces": int(bool(E["results"][i]["correct"])),
        }
        f.write(json.dumps(rec) + "\n")
print(f"wrote {n} rows -> {a.out}")
