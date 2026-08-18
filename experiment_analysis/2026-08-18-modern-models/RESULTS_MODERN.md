# Modern-models robustness — the endorsement mechanism becomes decisive
*Date: 2026-08-18 | 40,604-prompt full eval | trio: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3*

Repeat of the full-dataset evaluation on 2024-era models with three genuinely different
tokenizers (Qwen BPE 152k, Llama-3 tiktoken 128k, Mistral 33k), answering the review's
"does this survive on modern models?" question. It does — and the finding **inverts** the
earlier 2023-era conclusion.

## Result

| method | acc | vs best single | test |
|---|---:|---:|---|
| Qwen2.5-7B | 61.01% | | |
| Mistral-7B-v0.3 | 63.91% | | |
| **Llama-3.1-8B (best single)** | **68.13%** | — | |
| **plain majority voting** | **62.92%** | **−5.21** | fails |
| **cross-endorsement selection (CES)** | **69.11%** | **+0.98** | McNemar p = 1.1e-7 |
| — CES *over voting* | | **+6.19** | p ≈ 0 (χ² = 1640) |
| best-model-per-task (oracle router) | 69.64% | | |
| oracle any-correct ceiling | 80.89% | | |

## The inversion (why this matters)

On the 2023-era complementary trio, CES beat voting by only +0.39pp — the review correctly
noted the novel part was marginal. Here **one model dominates** (Llama-3.1 68.1% vs 63.9 / 61.0),
which is the realistic modern regime, and the picture flips:

- **Plain voting collapses to 62.92% — 5.2 points *below* the best single model** — because the
  two weaker members outvote the strong one.
- **CES beats the best single model (69.11%)**, and CES-over-voting is **+6.19pp**.

Decomposition (items gained vs best single):

| branch | voting | CES | CES over voting |
|---|---:|---:|---:|
| discrete | +445 | +358 | −87 |
| **open-QA** | **−2561** | **+40** | **+2601** |

On open-QA, voting's non-agreement fallback picks a weak model and loses 2,561 items; the
likelihood-endorsement (MBR) branch recovers to +40 vs best single. **The endorsement mechanism
— the genuinely non-voting component — is what carries the ensemble past the best single model
here.** It is not a +0.3pp add-on on modern models; it is decisive.

## Not a length artifact (open-QA, n=22,233)

| selector | acc |
|---|---:|
| best single (Mistral/Llama tie ~65.4) | 65.44 |
| select longest | 61.15 |
| select shortest | 54.95 |
| **endorsement (CES)** | **65.60** |

Endorsement beats length-based selection by 4–11 points; chosen answers are only mildly longer
(6.3 vs 5.1 words). CES also ≈ the gold-free oracle per-task router (69.11 vs 69.64).

## Takeaway for the paper

The contribution's strength is **model-regime-dependent**, and that is itself the finding:
- complementary models (2023-era): answer-space *voting* already captures most of the gain;
- one dominant modern model: voting *fails*, and **likelihood-endorsement selection is the
  essential mechanism** that lets a heterogeneous ensemble still beat its best member.

Reproduce: `eval/run_single_batched.py` (each model) → `eval/ces_batched.py --config
config_modern2.yaml` → `eval/paper_analysis.py`.
