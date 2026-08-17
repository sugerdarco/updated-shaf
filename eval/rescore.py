#!/usr/bin/env python3
"""Re-score saved eval result files with the current scorer (no model re-run).

Requires result files written by run_single.py / run_ensemble.py, which persist
the full item (gold) and full generation per prompt.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring

for path in sys.argv[1:]:
    d = json.load(open(path))
    by = collections.defaultdict(lambda: [0, 0])
    c = 0
    for r in d["results"]:
        it = r["item"]
        ok = bool(scoring.score_item(it, r["gen"]))
        c += ok
        by[it["task"]][0] += ok
        by[it["task"]][1] += 1
    n = len(d["results"])
    tasks = " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(by.items()))
    print(f"{os.path.basename(path):26s} acc={c/n*100:5.1f}% ({c}/{n})  {tasks}")
