# Distribution-space fusion vs. answer-space selection for heterogeneous LLMs

An analysis of ensembling LLMs with **different tokenizers** (Mistral-7B-Instruct,
Llama-2-13B-chat, Yi-6B-Chat). The repo contains two ways to reconcile them and an honest
evaluation of both across six DeePEn benchmarks (40,604 prompts).

> **Framing (post-review):** this is a **negative-result / analysis** study, not a new-method
> paper. The headline finding is that an *elaborate byte-space distribution-fusion pipeline
> collapses*, while a *trivial answer-space selector beats it and the best single model* — most
> of that gain being plain majority voting. See [`docs/ANALYSIS.md`](docs/ANALYSIS.md) for the
> full, honest decomposition, significance tests, and prior-art positioning.

## Headline results (full 40,604-prompt eval)

| method | accuracy | vs best single |
|---|---:|---:|
| Best single model (Mistral-7B) | 61.32% | — |
| **Byte-space distribution fusion (SAHF)** | **~36%** | −25 (byte-bloat) |
| Answer-space **majority voting** | 62.78% | +1.46 (p ≪ 1e-3) |
| Answer-space **cross-endorsement selection (CES)** | 63.17% | +1.85 (p = 2e-28) |
| — CES over voting | | +0.39 (p = 3.9e-4) |
| oracle any-correct ceiling | 76.62% | |

~79% of the improvement over the best single model is **plain voting**; the only non-voting
component (likelihood endorsement, ≈ MBR) contributes +0.39pp, concentrated in open-QA and
verified *not* to be a length artifact.

## The two reconciliation strategies

- **Byte-space fusion** (`pipeline/`, `prefix_tree_build/`, `runners/`) — the original SAHF
  architecture: average next-*byte* distributions on a shared byte-prefix tree. Averaging bytes
  across mismatched tokenizers produces an argmax no model intended ("byte-bloat") → misspelled
  output → ~36%. Three rescues fail (`eval/deleg_orchestrator.py`).
  Spec: [`docs/DECOUPLED_ARCHITECTURE_SPEC.md`](docs/DECOUPLED_ARCHITECTURE_SPEC.md).
- **Answer-space selection** (`eval/`) — each model answers cleanly, then one answer is selected
  by voting (discrete) + likelihood endorsement / MBR (open-QA).
  Method: [`docs/CES_METHOD.md`](docs/CES_METHOD.md) · Usage: [`eval/README.md`](eval/README.md).

## Open work before submission

DeePEn reproduction (does the *paradigm* fail or just our byte-tree?), modern models
(Llama-3.x / Qwen2.5), MBR & LLM-Blender/PairRanker baselines, weighted-voting ablation, seeds/CIs.
Detailed in [`docs/ANALYSIS.md`](docs/ANALYSIS.md) §5.

## Layout

```text
pipeline/ prefix_tree_build/ runners/   byte-space fusion (SAHF stages 0–8)
eval/                                    answer-space selection + evaluation harness
baseline/                                single-model baseline metrics
experiment_analysis/                     dated experiment write-ups
docs/                                    architecture, method (CES), and ANALYSIS
tests/                                   unit + parity tests
```

Datasets follow DeePEn (Huang et al., 2024); see [`dataset/README.md`](dataset/README.md).
