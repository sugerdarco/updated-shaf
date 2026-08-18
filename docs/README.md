# SAHF Documentation

Deep-dive technical documentation for the SAHF cross-tokenizer multi-agent project.

## Files

```text
docs/
├── README.md                          # This index
├── DECOUPLED_ARCHITECTURE_SPEC.md     # Byte-fusion (Stage 0–8) cross-tokenizer architecture
├── RUNTIME_PERFORMANCE.md             # Latency/memory of the byte-fusion pipeline
└── EWC_METHOD.md                      # Endorsement-Weighted Consensus — the method that beats
                                       # the single-model baseline (accuracy)
```

## Two layers of the project

1. **Byte-fusion pipeline** (`pipeline/`, `prefix_tree_build/`, `runners/`) — the original
   SAHF architecture that reconciles mismatched tokenizers by fusing next-byte distributions
   on a shared byte-prefix tree. See `DECOUPLED_ARCHITECTURE_SPEC.md` and `RUNTIME_PERFORMANCE.md`.

2. **Accuracy evaluation & ensemble reconciliation** (`eval/`) — added to answer whether the
   ensemble actually *beats the best single model*. It does, but not via byte fusion (which
   collapses to ~36% through "byte-bloat"): the winning method is **Endorsement-Weighted
   Consensus**, specified in [`EWC_METHOD.md`](EWC_METHOD.md) and documented with usage in
   [`../eval/README.md`](../eval/README.md). Empirical write-ups live in
   [`../experiment_analysis/2026-08-18-*`](../experiment_analysis).

**Headline result (full DeePEn eval set, 40,604 prompts):** EWC 63.17% vs best single model
61.31%.
