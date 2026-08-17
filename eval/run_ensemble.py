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
from chat_agent import ChatTemplateAgent
from deleg_orchestrator import DelegatingOrchestrator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_sheaf.yaml")
    ap.add_argument("--sample", default="eval/sample_100.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--devices", default="cuda:7,cuda:3,cuda:2")
    ap.add_argument("--limit", type=int, default=0, help="only first N prompts (debug)")
    ap.add_argument("--confidence-weight", action="store_true",
                    help="weight each agent's byte vote by its section peakedness")
    ap.add_argument("--conf-power", type=float, default=1.0)
    ap.add_argument("--sharpen", type=float, default=1.0,
                    help="exponent on fused next-byte dist before decode (>1 sharpens)")
    ap.add_argument("--weights", default="", help="comma per-agent global weights, e.g. 1,1.5,1")
    ap.add_argument("--chat", action="store_true", help="wrap each agent's context in its chat template")
    ap.add_argument("--mode", default="fuse", choices=["fuse", "delegate", "agree_delegate"],
                    help="fuse=byte averaging (base); agree_delegate=AGED (consensus when agents "
                         "agree, most-confident agent otherwise); delegate=always most-confident")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dtype = getattr(torch, cfg["dtype"])
    devs = args.devices.split(",")
    AgentCls = ChatTemplateAgent if args.chat else HFAgent
    print(f"Loading {len(cfg['models'])} agents ({AgentCls.__name__}) on {devs} ...", flush=True)
    agents = [AgentCls(n, device=d, dtype=dtype) for n, d in zip(cfg["models"], devs)]

    tree = None
    tp = cfg["sheaf"].get("tree_path")
    if tp and os.path.exists(tp):
        tree = BytePrefixTree.load(tp)
        print(f"Loaded prefix tree: {tp} ({tree.n_nodes:,} nodes)", flush=True)
    th = GateThresholds(entropy=cfg["gate"]["entropy_threshold"],
                        divergence=cfg["gate"]["divergence_threshold"])
    weights = [float(x) for x in args.weights.split(",")] if args.weights else None
    print(f"fusion knobs: confidence_weight={args.confidence_weight} "
          f"conf_power={args.conf_power} sharpen={args.sharpen} weights={weights}", flush=True)

    items = [json.loads(l) for l in open(args.sample) if l.strip()]
    if args.limit:
        items = items[: args.limit]
    correct, by_task, results = 0, {}, []
    t0 = time.time()
    for i, it in enumerate(items):
        body, max_new = build_prompt(it)
        if args.chat:
            for a in agents:
                a.set_prompt(body)
        common = dict(
            weights=weights, max_new_bytes=max_new,
            mad_multiplier=cfg["gate"]["mad_multiplier"], k=cfg["sheaf"]["top_k"],
            min_support=cfg["sheaf"].get("min_support"), tree=tree, logger=None,
        )
        if args.mode == "fuse":
            orch = SheafOrchestrator(agents, th, **common)
        else:
            orch = DelegatingOrchestrator(
                agents, th, delegate_on_agree=(args.mode == "delegate"), **common)
        for pl in (orch._path_a, orch._path_b):
            pl.reconciler.confidence_weight = args.confidence_weight
            pl.reconciler.conf_power = args.conf_power
            pl.reconciler.sharpen = args.sharpen
        full, hist = orch.generate(body)
        gen = full[len(body):] if full.startswith(body) else full
        ok = bool(scoring.score_item(it, gen))
        correct += ok
        d = by_task.setdefault(it["task"], [0, 0])
        d[0] += ok
        d[1] += 1
        results.append({"item": it, "correct": ok, "steps": len(hist), "gen": gen})
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
