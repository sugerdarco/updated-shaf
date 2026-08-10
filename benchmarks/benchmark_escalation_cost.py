"""
Benchmark: Escalation Cost
Measures latency overhead (in ms) when triggering Stage 6 Outlier Check and Stage 7
Geometric Median escalation versus fast path mean fusion.
"""

import time
import torch
import numpy as np

from pipeline.stage_5_fast_mean_fusion import fast_mean_fusion
from pipeline.stage_6_and_7_robust_escalation import detect_outliers, weiszfeld_geometric_median

def run_escalation_benchmark(n_agents=3, dim=32000, n_trials=50):
    print(f"--- Benchmarking Escalation Latency Cost ---")
    print(f"Agents: {n_agents}, Dimension: {dim}, Trials: {n_trials}")

    # Generate mock amplitude vectors
    psi_list = [torch.randn(dim) for _ in range(n_agents)]
    psi_list = [p / torch.norm(p, p=2) for p in psi_list]

    # Measure Stage 5 Mean Fusion
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = fast_mean_fusion(psi_list)
    t_stage5 = ((time.perf_counter() - t0) / n_trials) * 1000

    # Measure Stage 6 + 7 Escalation
    fused_mean = fast_mean_fusion(psi_list)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        mask, _ = detect_outliers(psi_list, fused_mean)
        med, _ = weiszfeld_geometric_median(psi_list, max_iter=20)
    t_escalation = ((time.perf_counter() - t0) / n_trials) * 1000

    print(f"Stage 5 (Fast Mean Fusion): {t_stage5:.3f} ms / step")
    print(f"Stage 6+7 (Escalation Tier): {t_escalation:.3f} ms / step")
    print(f"Escalation Overhead: +{t_escalation - t_stage5:.3f} ms")

if __name__ == "__main__":
    run_escalation_benchmark()
