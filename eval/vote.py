#!/usr/bin/env python3
"""Answer-level voting ensemble.

Reconciles the agents in ANSWER space instead of byte space: each model answers
independently (clean chat-templated generation, no byte-bloat), then a per-prompt
majority vote picks the consensus answer. Ties break toward the earlier file, so
pass the models in priority order (strongest first).

Reuses the generations already saved by run_single.py, so it needs no GPU.

    python eval/vote.py out/single_llama.json out/single_mistral.json out/single_yi.json
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring


def vote(items_type, preds):
    """preds is in model-priority order; None means that model abstained."""
    valid = [p for p in preds if p is not None]
    if not valid:
        return None
    key = (lambda p: scoring._norm(str(p))) if items_type == "openqa" else (lambda p: p)
    counts = collections.Counter(key(p) for p in valid)
    best = max(counts.values())
    winners = {k for k, c in counts.items() if c == best}
    for p in valid:  # priority order -> first model whose answer is a top vote
        if key(p) in winners:
            return p
    return valid[0]


def main():
    files = sys.argv[1:]
    data = [json.load(open(f)) for f in files]
    n = len(data[0]["results"])
    correct = 0
    by = collections.defaultdict(lambda: [0, 0])
    for i in range(n):
        item = data[0]["results"][i]["item"]
        preds = [scoring.predict_item(d["results"][i]["item"], d["results"][i]["gen"]) for d in data]
        voted = vote(item["type"], preds)
        ok = bool(scoring.score_prediction(item, voted))
        correct += ok
        by[item["task"]][0] += ok
        by[item["task"]][1] += 1
    tasks = " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(by.items()))
    print(f"VOTE({','.join(os.path.basename(f).replace('single_','').replace('.json','') for f in files)})"
          f" acc={correct/n*100:5.1f}% ({correct}/{n})  {tasks}")


if __name__ == "__main__":
    main()
