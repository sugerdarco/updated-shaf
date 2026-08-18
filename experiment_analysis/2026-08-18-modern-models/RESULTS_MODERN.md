# Modern-models robustness (CORRECTED)
*Date: 2026-08-18 | 40,604-prompt full eval | trio: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3*

> **Correction (supersedes the original version of this file).** The original write-up claimed an
> "inversion" on modern models — that plain voting *fails* (62.92%, below the best single model)
> while CES is *decisive* (+6.19pp over voting). **That finding was an artifact and is retracted.**
> An adversarial review pointed out, correctly, that the open-QA "voting" score is set by an
> order-dependent tie-break (`openqa_choice_idx` returns candidate index 0 when <2 models agree,
> which for free-form text is almost always), and the original modern run listed the **weakest**
> model (Qwen2.5) first. With a fair fallback the effect disappears (see below).

## Setup
2024-era models, three distinct tokenizers (Qwen BPE 152k, Llama-3 tiktoken 128k, Mistral 33k),
same 40,604-prompt eval and same CES / voting code.

## Single models and CES

| method | acc |
|---|---:|
| Qwen2.5-7B | 61.01% |
| Mistral-7B-v0.3 | 63.91% |
| **Llama-3.1-8B (best single)** | **68.13%** |
| **CES** | **69.11%** (+0.98 vs best single, McNemar p=1e-7) |
| best-model-per-task oracle | 69.64% |
| oracle any-correct ceiling | 80.89% |

CES beats the best single model by +0.98pp — but this is essentially a well-constructed **majority
vote**, not the endorsement mechanism.

## The voting baseline is order-dependent on open-QA (the retraction)

Voting's open-QA branch has no principled tie-break; its score is set by whichever model is listed
first:

| open-QA tie-break fallback | voting (full trio) | vs CES 69.11% |
|---|---:|---:|
| Qwen first (weakest) — *originally reported* | 62.92% | CES +6.19 |
| **Llama-3.1 first (best) — fair** | **69.25%** | **CES −0.14 (p=0.38, n.s.)** |
| Mistral first | 68.83% | CES +0.28 |

**With a fair fallback, voting (69.25%) ties/beats CES (69.11%).** The apparent "CES beats voting"
was entirely the weakest model being handed the open-QA fallback. Consistent with this, CES is
already *worse* than voting on the well-defined discrete branch in the modern run (−87 items).

## The honest measure of the novel component

Majority voting is not well-defined on free-form answers, so "voting fails on open-QA" is a
statement about a tie-break default, not about voting. The order-independent comparison is
**endorsement vs the best single model on open-QA**:

| | endorsement-oq | best-single-oq | Δ |
|---|---:|---:|---:|
| 2023-era trio | 65.77 | 65.28 | **+0.49** |
| modern trio | 65.60 | 65.44 | **+0.16** |

Small, **consistent in sign across two model generations**, and robust to the length-confound
ablation (endorsement 65.6 vs longest 61.2 / shortest 55.0). That consistency is a modest, real
result. **There is no regime-dependent inversion, and no +6pp effect.**

## Takeaway (corrected)

CES ≈ a well-constructed answer-space voting baseline. The genuinely non-voting component
(likelihood/MBR endorsement on open-QA) adds a small (+0.16 to +0.49pp), consistent, length-robust
improvement over the best single model — but it is marginal and does not, on its own, carry a
method-paper claim. This is the same conclusion the review reached on the 2023-era models, now
confirmed at modern scale. Next baseline required: **weighted majority voting** (the textbook fix
for a dominant model being outvoted).
