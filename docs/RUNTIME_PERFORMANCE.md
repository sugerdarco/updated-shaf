# SAHF Runtime Performance & Latency Specifications

This document outlines runtime performance metrics, latency overheads, and memory characteristics of the decoupled SAHF cross-tokenizer pipeline.

## 1. Latency Breakdown per Decoding Step

| Stage / Component | Operation | Typical Latency | Notes |
|---|---|---|---|
| **Stage 1** | Logit-to-Amplitude Conversion | ~0.1 ms | $\psi = \sqrt{P}$, vectorized NumPy/PyTorch |
| **Stage 2** | Divergence Gate | ~0.05 ms | Heuristic entropy & pairwise spread check |
| **Stage 3** | Fast Passthrough | < 0.01 ms | Direct sampling when gate opens |
| **Stage 5** | Chordal Mean Fusion | ~0.3 ms | Vectorized spherical mean |
| **Stage 6 & 7** | Outlier Check & Geometric Median | ~1.5 - 3.0 ms | Triggered only upon divergence escalation |
| **Stage 8** | Sheaf Reconciliation | ~2.0 - 5.0 ms | Per-node byte tree traversal & projection |

## 2. Memory & Scaling Behavior

- **Dynamic `topk_union` Tree**: Consumes < 5 MB RAM per step, re-building trie over top-k logits dynamically.
- **Prebuilt Full-Vocabulary Tree**: ~50–150 MB RAM artifact (`prefix_tree.npz`), static construction pre-run.
- **Breakdown Point Guarantee**: 50% adversarial tolerance when running $\ge 3$ heterogeneous agents.
