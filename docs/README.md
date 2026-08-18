# SAHF Documentation

Deep-dive technical documentation for the SAHF cross-tokenizer multi-agent project.

## Files

```text
docs/
├── README.md                          # This index
├── DECOUPLED_ARCHITECTURE_SPEC.md     # Byte-fusion (Stage 0–8) cross-tokenizer architecture
├── RUNTIME_PERFORMANCE.md             # Latency/memory of the byte-fusion pipeline
├── CES_METHOD.md                      # Cross-Endorsement Selection — answer-space selector
│                                      # (renamed off "EWC"); the honest method spec
└── ANALYSIS.md                        # Honest evaluation: byte-fusion fails, voting is most of
                                       # the gain, significance + length ablation + prior art
```

## Two layers of the project

1. **Byte-fusion pipeline** (`pipeline/`, `prefix_tree_build/`, `runners/`) — the original
   SAHF architecture that reconciles mismatched tokenizers by fusing next-byte distributions
   on a shared byte-prefix tree. See `DECOUPLED_ARCHITECTURE_SPEC.md` and `RUNTIME_PERFORMANCE.md`.

2. **Accuracy evaluation & answer-space selection** (`eval/`) — added to test whether the
   ensemble *beats the best single model*. It does, but not via byte fusion (which collapses to
   ~36% through "byte-bloat"): a trivial answer-space selector wins, and **most of the gain is
   plain majority voting**. Method spec: [`CES_METHOD.md`](CES_METHOD.md); honest evaluation
   (decomposition, significance, length ablation, prior art): [`ANALYSIS.md`](ANALYSIS.md);
   usage: [`../eval/README.md`](../eval/README.md).

**Headline (full DeePEn eval set, 40,604 prompts):** best single 61.32% → voting 62.78% →
CES 63.17%; byte-fusion ~36%. This is framed as a **negative-result / analysis** study; see
`ANALYSIS.md` for the honest read and the open work (DeePEn reproduction, modern models).
