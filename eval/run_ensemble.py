#!/usr/bin/env python3
"""SAHF cross-tokenizer ensemble on the mixed 100-prompt sample.

Same prompts and same unified scorer as run_single.py, so the ensemble accuracy is
directly comparable to the single-model baselines.
"""
import argparse
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring
from prompts import build_prompt

from pipeline.stage_0_agent_ensemble import DirectHFAgent as HFAgent
from pipeline.stage_2_divergence_gate import GateThresholds
from prefix_tree_build.prefix_tree import BytePrefixTree
from runners.sheaf_orchestrator import SheafOrchestrator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_sheaf.yaml")
    ap.add_argument("--sample", default="eval/sample_100.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--devices", default="cuda:7,cuda:3,cuda:2")
    ap.add_argument("--limit", type=int, default=0, help="only first N prompts (debug)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dtype = getattr(torch, cfg["dtype"])
    devs = args.devices.split(",")
    print(f"Loading {len(cfg['models'])} agents on {devs} ...", flush=True)
    agents = [HFAgent(n, device=d, dtype=dtype) for n, d in zip(cfg["models"], devs)]

    tree = None
    tp = cfg["sheaf"].get("tree_path")
    if tp and os.path.exists(tp):
        tree = BytePrefixTree.load(tp)
        print(f"Loaded prefix tree: {tp} ({tree.n_nodes:,} nodes)", flush=True)
    th = GateThresholds(entropy=cfg["gate"]["entropy_threshold"],
                        divergence=cfg["gate"]["divergence_threshold"])

    items = [json.loads(l) for l in open(args.sample) if l.strip()]
    if args.limit:
        items = items[: args.limit]
    correct, by_task, results = 0, {}, []
    t0 = time.time()
    for i, it in enumerate(items):
        body, max_new = build_prompt(it)
        orch = SheafOrchestrator(
            agents, th, max_new_bytes=max_new,
            mad_multiplier=cfg["gate"]["mad_multiplier"], k=cfg["sheaf"]["top_k"],
            min_support=cfg["sheaf"].get("min_support"), tree=tree, logger=None,
        )
        full, hist = orch.generate(body)
        gen = full[len(body):] if full.startswith(body) else full
        ok = bool(scoring.score_item(it, gen))
        correct += ok
        d = by_task.setdefault(it["task"], [0, 0])
        d[0] += ok
        d[1] += 1
        results.append({"task": it["task"], "type": it["type"], "correct": ok,
                        "steps": len(hist), "gen": gen[:600]})
        print(f"[{i+1}/{len(items)}] {it['task']:8s} ok={int(ok)} "
              f"acc={correct/(i+1)*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    acc = correct / len(items) * 100
    summary = {"accuracy": acc, "correct": correct, "total": len(items),
               "by_task": {k: {"correct": v[0], "total": v[1]} for k, v in by_task.items()},
               "seconds": round(time.time() - t0, 1)}
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=2)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
