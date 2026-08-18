# SAHF — Cross-Tokenizer Multi-Agent Ensemble

An ensemble of heterogeneous LLMs with **different tokenizers** (Mistral-7B-Instruct,
Llama-2-13B-chat, Yi-6B-Chat) made to answer together, and the question of whether doing
so **beats the best single model's accuracy**.

## Result

**Yes — with Endorsement-Weighted Consensus (EWC).** On the full DeePEn evaluation set
(40,604 prompts across ARC, MMLU, GSM8K, PIQA, TriviaQA, NQ):

| Method | Accuracy |
|---|---:|
| Best single model (Mistral-7B) | 61.31% |
| Original byte-fusion ensemble (SAHF) | ~36% (byte-bloat) |
| **EWC (this work)** | **63.17%** |
| — oracle per-task router (gold-free target) | 63.26% |
| — oracle any-correct ceiling | 76.63% |

EWC nearly matches an oracle that picks the best model per task — *without labels*. It also
beat the baseline on smaller samples (100-prompt +6, 1,200-prompt +2.7).

## The two layers

- **`pipeline/`, `prefix_tree_build/`, `runners/`** — the original SAHF byte-fusion
  architecture (Stages 0–8): reconcile mismatched tokenizers by fusing next-*byte*
  distributions on a shared byte-prefix tree. Mathematically elegant, but averaging bytes
  across mismatched tokenizers misspells words ("byte-bloat") and tanks accuracy. See
  [`docs/DECOUPLED_ARCHITECTURE_SPEC.md`](docs/DECOUPLED_ARCHITECTURE_SPEC.md).
- **`eval/`** — accuracy evaluation and the reconciliation method that actually wins by
  moving from *byte space* to *answer space*: each model answers cleanly, then the whole
  ensemble endorses one answer (hard agreement for discrete answers, soft cross-tokenizer
  likelihood for free-form). See [`eval/README.md`](eval/README.md) and
  [`docs/EWC_METHOD.md`](docs/EWC_METHOD.md).

## Layout

```text
pipeline/            byte-fusion stages 0–8
prefix_tree_build/   shared byte-prefix tree (the fusion base space)
runners/             byte-fusion decode orchestrator
eval/                accuracy harness + EWC ensemble  (see eval/README.md)
baseline/            single-model baseline metrics (the target to beat)
experiment_analysis/ dated experiment write-ups
docs/                architecture + method specs
tests/               unit + parity tests
```

## Quickstart

```bash
pip install -r requirements.txt
# build a sample, run the 3 single-model baselines, then EWC over them:
#   see eval/README.md  ("Reproduce")
```

Datasets follow DeePEn ("Ensemble Learning for Heterogeneous LLMs with Deep Parallel
Collaboration"); see [`dataset/README.md`](dataset/README.md).
