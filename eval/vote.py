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


def vote_choice(preds):
    """Majority vote over discrete predictions (MC letters / numbers).

    preds is in model-priority order; None = abstain; ties break to priority."""
    valid = [p for p in preds if p is not None]
    if not valid:
        return None
    counts = collections.Counter(valid)
    best = max(counts.values())
    winners = {k for k, c in counts.items() if c == best}
    for p in valid:
        if p in winners:
            return p
    return valid[0]


def openqa_choice_idx(shorts):
    """Pick which model's answer to trust for open QA (no gold peeking).

    WARNING: majority voting is ill-defined on free-form text. When <2 models agree
    (near-always) this falls back to candidate index 0, so the open-QA voting score
    is ORDER-DEPENDENT and is NOT a fair baseline — listing a weak model first tanks
    it, listing a strong one inflates it. Do not compare CES against this on open-QA;
    use endorsement vs the best single model instead (see eval/paper_analysis.py's
    "HONEST measures"). Kept only for the discrete-answer voting reference."""
    counts = collections.Counter(s for s in shorts if s)
    if counts:
        best = max(counts.values())
        if best >= 2:
            winners = {k for k, c in counts.items() if c == best}
            for idx, s in enumerate(shorts):
                if s in winners:
                    return idx
    return 0


def main():
    files = sys.argv[1:]
    data = [json.load(open(f)) for f in files]
    n = len(data[0]["results"])
    correct = 0
    by = collections.defaultdict(lambda: [0, 0])
    for i in range(n):
        item = data[0]["results"][i]["item"]
        gens = [d["results"][i]["gen"] for d in data]
        if item["type"] in ("mc", "number"):
            preds = [scoring.predict_item(item, g) for g in gens]
            ok = bool(scoring.score_prediction(item, vote_choice(preds)))
        else:  # open QA: score the chosen model's full generation
            shorts = [scoring._norm(scoring.predict_openqa(item, g)) for g in gens]
            ok = bool(scoring.score_openqa(item, gens[openqa_choice_idx(shorts)]))
        correct += ok
        by[item["task"]][0] += ok
        by[item["task"]][1] += 1
    tasks = " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(by.items()))
    print(f"VOTE({','.join(os.path.basename(f).replace('single_','').replace('.json','') for f in files)})"
          f" acc={correct/n*100:5.1f}% ({correct}/{n})  {tasks}")


if __name__ == "__main__":
    main()
