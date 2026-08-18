# Analysis: distribution-space fusion vs. answer-space selection across mismatched tokenizers

Honest analysis of the experiments in this repo, written after an adversarial review.
It replaces the earlier over-claimed "EWC beats the baseline" framing. All numbers are
recomputed from the saved 40,604-prompt generations by `eval/paper_analysis.py`.

## TL;DR

- The original **byte-level distribution fusion collapses to ~36%** vs ~61% best single —
  a mechanistically-diagnosed failure ("byte-bloat"). Three rescues (confidence weighting,
  sharpening, AGED delegation) fail (33–48%).
- A **trivial answer-space selector beats it and the best single model.** On 40,604 prompts:

  | method | acc | vs best single | test |
  |---|---:|---:|---|
  | best single (Mistral-7B) | 61.32% | — | |
  | **plain majority voting** | **62.78%** | +1.46 | McNemar p ≪ 1e-3 |
  | cross-endorsement selection (CES) | 63.17% | +1.85 | p = 2e-28 |
  | — CES *over voting* | | +0.39 | p = 3.9e-4 |
  | oracle any-correct ceiling | 76.62% | | |

- **On these 2023-era, complementary models, ~79% of the gain over the best single model is
  plain voting**; the non-voting likelihood-endorsement component adds only **+0.39pp** overall.
- **But that is regime-dependent — and the regime that matters flips the conclusion.** On a
  2024-era trio where one model dominates (Llama-3.1-8B 68.1% vs Mistral-v0.3 63.9% / Qwen2.5
  61.0%), **plain voting FAILS — 62.92%, five points *below* the best single model** — while
  **CES beats it (69.11%, p=1e-7)** and CES-over-voting is **+6.19pp (p≈0)**. Here the
  likelihood-endorsement branch is not a rounding error; it is the *decisive* mechanism (see §5).

**Implication for framing:** the honest contribution is two-fold — (1) the **negative result**
that distribution-space byte fusion collapses while a trivial answer-space baseline beats it and
the best single model; and (2) the **regime analysis**: answer-space *voting* suffices when
models are complementary, but **likelihood-endorsement selection is essential when one modern
model dominates** — exactly the realistic case. The DeePEn reproduction (below) tests whether the
distribution-space paradigm itself fails or only our byte-tree instantiation.

## 1. Why byte-space fusion fails (the negative result)

Averaging next-*byte* distributions from models with different tokenizers produces a fused
distribution whose argmax is a byte **no constituent model intended**, yielding misspellings
("secretedd", "Juiced Box Band") that destroy exact-match scoring. Rescues tried and failed:
confidence-weighting/sharpening the fusion (33–34%, *worse*), and **AGED** — use the byte tree
only as an agreement sensor and delegate divergent steps to the most-confident agent
(47–48%). A specific finding: **per-step leader switching zeroes out GSM8K** because it breaks
multi-step reasoning mid-chain. (`eval/deleg_orchestrator.py`.)

**Caveat that must be resolved before claiming the paradigm fails:** DeePEn does distribution
averaging in a *relative-representation* space and reports gains on these same six benchmarks.
Our 36% shows *our byte-tree instantiation* fails — not necessarily the paradigm. A DeePEn
reproduction on this setup is required and is listed as open work.

## 2. Answer-space selection

Each model answers independently and cleanly (its own chat template, greedy — no cross-model
interference). Then one answer is selected. Two selectors:

- **Voting** (`eval/vote.py`): majority vote on the extracted discrete answer; open-QA falls
  back to an agreement/priority pick. This is textbook plurality voting.
- **Cross-Endorsement Selection (CES)** (`eval/ces.py --hybrid`, batched `ces_batched.py`):
  discrete = the same majority vote (endorsement only breaks the rare 3-way tie); open-QA =
  pick the candidate maximizing summed length-normalized log-prob under every model. The
  open-QA branch is **likelihood-utility Minimum Bayes Risk / n-best rescoring** (Kumar &
  Byrne 2004; Model-Based MBR, Jinnai et al. 2023) applied across heterogeneous tokenizers.

### Decomposition (items gained over best single)

| branch | voting | CES | CES over voting |
|---|---:|---:|---:|
| discrete (mc/number) | +592 | +641 | +49 |
| open-QA | **+1** | **+109** | **+108** |

Voting captures the discrete gain almost entirely; on open-QA it gains *nothing*, and the
likelihood-endorsement branch is the only thing that helps there.

## 3. Is the open-QA gain a length artifact?

`score_openqa` credits any gold alias appearing anywhere in the generation, so selecting among
candidates of different verbosity could reward length. Ablation on the 22,233 open-QA prompts:

| open-QA selector | acc |
|---|---:|
| best single model (Mistral) | 65.28 |
| select **longest** candidate | 62.96 |
| select **shortest** candidate | 62.75 |
| **endorsement (CES)** | **65.77** |

Endorsement beats both length-based selectors by ~3 points and edges the best single model.
Chosen answers are only mildly longer (22.7 vs 19.3 words). **The gain is not explained by
length** — pure length selection underperforms it.

## 4. Prior art this sits next to

- **MBR / cross-model n-best rescoring** — the open-QA branch *is* likelihood-utility MBR.
  Closest relative; must be a baseline.
- **LLM-Blender / PairRanker (Jiang et al., ACL 2023)** — canonical heterogeneous-LLM
  candidate selection; GAC and others benchmark against it.
- **DeePEn (Huang et al., 2024)** — the distribution-space method this repo re-implements at
  byte level; also the paper that surveys rerank/selection ensembling.
- **Weighted majority voting** — the natural fix for the PIQA regression (76.1 → 71.1 when a
  dominant model is outvoted) is decades-old weighted voting, not novel.

## 5. Modern models: the endorsement mechanism becomes decisive

Repeating the full 40,604-prompt evaluation on a 2024-era trio (Qwen2.5-7B, Llama-3.1-8B,
Mistral-7B-v0.3 — three distinct tokenizers) inverts the §2 conclusion. Full write-up:
[`../experiment_analysis/2026-08-18-modern-models/RESULTS_MODERN.md`](../experiment_analysis/2026-08-18-modern-models/RESULTS_MODERN.md).

| method | acc | vs best single |
|---|---:|---:|
| best single (Llama-3.1-8B) | 68.13% | — |
| plain voting | 62.92% | **−5.21** (fails) |
| CES | 69.11% | **+0.98** (p = 1e-7) |
| — CES over voting | | **+6.19** (p ≈ 0) |

Here one model dominates, so majority voting is dragged *below* the best single model. On open-QA
voting loses 2,561 items vs best-single while endorsement is +40 — a +2,601-item swing. **The
likelihood-endorsement branch is decisive, not marginal, in this regime**, and again not a length
artifact (endorsement 65.6% vs longest 61.2% / shortest 55.0%). Contribution strength is
model-regime-dependent — which is itself a reportable finding.

## 6. Limitations / open work

1. **DeePEn reproduction** (pivotal — decides "our instantiation fails" vs "paradigm fails").
2. **Modern models** — results are on 2023-era Llama-2-13B / Mistral-7B-v0.2 / Yi-6B; needs
   Llama-3.x / Qwen2.5-class to show the complementarity survives.
3. **Ablations** — N, ensemble composition, and a weighted-voting baseline.
4. **Statistics** — McNemar reported here; add seeds/CIs and the MBR & PairRanker baselines.
5. **Naming** — the method was renamed off "EWC" (collides with Elastic Weight Consolidation).
   The `ces_batched == ces` parity holds only for odd N (even N can split votes).
