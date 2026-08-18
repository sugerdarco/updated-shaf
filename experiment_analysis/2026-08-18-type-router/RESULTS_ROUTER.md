# Answer-type routing captures nearly all the ensemble gain
*Date: 2026-08-18 | both 40,604-prompt runs | from the shipped `eval/results_export/` correctness data*

Prompted by the second review. The shipped per-prompt data made one thing computable that the
earlier write-ups missed: **no single model is best at both answer types**, so the ensemble's
advantage over the best single model is mostly *answer-type routing*, not fusion or selection.

## Per-branch single-model accuracy (modern trio)

| model | discrete (n=18,371) | open-QA (n=22,233) |
|---|---:|---:|
| Qwen2.5-7B | **73.91** | 50.34 |
| Llama-3.1-8B | 71.41 | 65.42 |
| Mistral-7B-v0.3 | 62.05 | **65.44** |
| CES | 73.35 | 65.60 |

Qwen wins discrete by +2.5pp and loses open-QA by 15. CES's discrete branch tracks the best
discrete model, its open-QA branch tracks the best open-QA model — it is functioning as an
**implicit answer-type router**.

## An explicit type-router matches/beats CES

Router = one model for discrete (mc/number), one for open-QA. **Dev-split** version chooses the
per-type best model on a random half and applies it to the other half (no task labels or test
peeking), averaged over 20 seeds:

| | best single | CES | type-router (dev-split, 20 seeds) | type-router (test-optimal) |
|---|---:|---:|---:|---:|
| 2023-era | 61.32 | 63.17 | 62.86 ±0.19 | 62.84 |
| modern | 68.13 | 69.11 | **69.19 ±0.18** | 69.27 |

- **Modern: the router (69.19) beats CES (69.11)**; CES is not significantly better than the
  test-optimal router (p=0.28).
- **2023-era: CES (63.17) edges the router (62.86) by +0.31**, marginally significant (p=0.045).

## Decomposition of CES's gain over the best single model

| | best→router (type routing) | router→CES (voting + endorsement) |
|---|---:|---:|
| 2023-era | +1.54 | +0.31 |
| modern | +1.06 | −0.08 |

**Almost all of the ensemble's gain over the best single model is answer-type routing** — a
two-number heuristic needing no task labels at test time. Byte-fusion, majority voting, and
likelihood-endorsement selection add little beyond it. The honest residual for the novel
endorsement mechanism is unchanged: **+0.16 to +0.49pp on open-QA**, same sign twice, length-robust.

## Reliability-weighted voting also ties CES

The other standard baseline (each model weighted by its dev-half per-type accuracy; discrete =
weighted majority, open-QA = defer to the most-reliable model), 20 seeds:

| | CES | weighted-vote (dev-split) |
|---|---:|---:|
| 2023-era | 63.17 | 63.11 ±0.21 |
| modern | 69.11 | 69.16 ±0.21 |

Within seed noise. So **CES ≈ type-router ≈ weighted voting** — the likelihood-endorsement mechanism
has no distinct advantage over standard baselines. (`eval/weighted_vote.py`.)

## Thesis (updated)
On a mixed-benchmark suite of heterogeneous, mismatched-tokenizer LLMs, the ensemble's advantage
over its best single member is **primarily answer-type routing**, not distribution fusion or
answer selection. Byte-level fusion actively *destroys* accuracy (byte-bloat); answer-space
selection recovers it but reduces to routing plus a marginal open-QA endorsement effect.

Reproduce: `python eval/type_router.py eval/results_export/full40k_2023era.jsonl eval/results_export/full40k_modern.jsonl`.
