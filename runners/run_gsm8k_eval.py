import os
import json
import re
import yaml
import time
from pathlib import Path
import torch

from pipeline.stage_0_agent_ensemble import DirectHFAgent as HFAgent
from pipeline.stage_2_divergence_gate import GateThresholds
from utils.logger import RunLogger
from prefix_tree_build.prefix_tree import BytePrefixTree
from runners.sheaf_orchestrator import SheafOrchestrator

def extract_number(text):
    # try to find numbers in the text, taking the last one typically as the final answer
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if matches:
        return float(matches[-1])
    return None

def main():
    cfg_path = "config_sheaf.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    dtype = getattr(torch, cfg["dtype"])
    devices = ["cuda:2", "cuda:3", "cuda:7"]
    agents = [HFAgent(name, device=dev, dtype=dtype) for name, dev in zip(cfg["models"], devices)]
    
    tree = BytePrefixTree.load(cfg["sheaf"]["tree_path"])
    thresholds = GateThresholds(
        entropy=cfg["gate"]["entropy_threshold"],
        divergence=cfg["gate"]["divergence_threshold"]
    )
    
    correct = 0
    total = 0
    
    results = []
    
    print("Starting GSM8K Evaluation...")
    
    with open("dataset/gsm8k_sample.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            question = data["question"]
            ans_str = data["answer"]
            
            # Ground truth answer
            gt_match = re.search(r"####\s*(-?\d+\.?\d*)", ans_str)
            if not gt_match: continue
            gt_ans = float(gt_match.group(1))
            
            # Formatting as few-shot or zero-shot. We will use a simple zero-shot prompt.
            prompt = f"Question: {question}\nAnswer: Let's think step by step.\n"
            
            orchestrator = SheafOrchestrator(
                agents, thresholds,
                max_new_bytes=cfg["sheaf"].get("max_new_bytes", 256),
                mad_multiplier=cfg["gate"]["mad_multiplier"],
                k=cfg["sheaf"].get("top_k", 16),
                tree=tree,
                logger=RunLogger(out_dir=cfg.get("out_dir", "out"), run_label=f"gsm8k_eval_p{total}")
            )
            
            output_text, _ = orchestrator.generate(prompt)
            pred_ans = extract_number(output_text)
            
            is_correct = (pred_ans == gt_ans)
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "question": question,
                "ground_truth": gt_ans,
                "prediction": pred_ans,
                "correct": is_correct,
                "output": output_text
            })
            
            print(f"[{total}] Correct: {is_correct} | Pred: {pred_ans} | GT: {gt_ans}")
            
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\nFinal Accuracy: {acc:.2f}% ({correct}/{total})")
    
    with open("gsm8k_eval_results.json", "w") as f:
        json.dump({"accuracy": acc, "correct": correct, "total": total, "details": results}, f, indent=2)

if __name__ == "__main__":
    main()
