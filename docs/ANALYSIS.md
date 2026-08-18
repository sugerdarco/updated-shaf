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

- **~79% of the gain over the best single model is plain voting**; the non-voting
  likelihood-endorsement component adds only **+0.39pp** overall (2023-era), and this holds on
  modern models too (§5). CES ≈ a well-constructed answer-space voting baseline.
- The genuinely-novel component (open-QA likelihood/MBR endorsement) contributes a **small,
  consistent** improvement over the best single model on open-QA: **+0.49pp (2023-era), +0.16pp
  (modern)** — same sign across two model generations, robust to the length-confound ablation, but
  marginal. *(An earlier draft claimed a +6.19pp modern-models "inversion"; that was an artifact of
  an order-dependent open-QA voting tie-break and has been retracted — see §5.)*

**Implication for framing:** this is a **negative-result / analysis** paper. Its contribution is
(1) the mechanistically-diagnosed collapse of byte-level distribution fusion (byte-bloat) while a
trivial answer-space selector beats it and the best single model; and (2) the honest decomposition
showing that gain is mostly plain voting, with a small consistent open-QA endorsement effect on
top. It is **not** a method paper about a new decisive ensemble mechanism. The DeePEn reproduction
(below) narrows the negative result to our byte-tree instantiation rather than the paradigm.

## 1. Why byte-space fusion fails (the negative result)

Averaging next-*byte* distributions from models with different tokenizers produces a fused
distribution whose argmax is a byte **no constituent model intended**, yielding misspellings
("secretedd", "Juiced Box Band") that destroy exact-match scoring. Rescues tried and failed:
confidence-weighting/sharpening the fusion (33–34%, *worse*), and **AGED** — use the byte tree
only as an agreement sensor and delegate divergent steps to the most-confident agent
(47–48%). A specific finding: **per-step leader switching zeroes out GSM8K** because it breaks
multi-step reasoning mid-chain. (`eval/deleg_orchestrator.py`.)

**Narrowed by reproduction (§6):** DeePEn does distribution averaging in a *relative-representation*
space. We ran DeePEn's released code on our own models (scoped: GSM8K, 50 items, Llama-2 + Mistral).
It produces **correct, coherently-spelled** answers — no byte-bloat. (The 44% on 50 items is
indicative only, and the 13B single-model baseline is incomplete, so we claim **no** numeric gain.)
The qualitative point is what matters: relative-space fusion is **not** broken on our exact models,
so the byte-bloat collapse is specific to *our byte-tree instantiation* (averaging next-*byte*
distributions yields an argmax no model intended). See
[`../experiment_analysis/2026-08-18-deepen/RESULTS_DEEPEN.md`](../experiment_analysis/2026-08-18-deepen/RESULTS_DEEPEN.md).

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

## 5. Modern models (2024-era) — and a retracted claim

Repeating the full 40,604-prompt evaluation on a 2024-era trio (Qwen2.5-7B, Llama-3.1-8B,
Mistral-7B-v0.3 — three distinct tokenizers; Llama-3.1 dominant at 68.1%). Full write-up:
[`../experiment_analysis/2026-08-18-modern-models/RESULTS_MODERN.md`](../experiment_analysis/2026-08-18-modern-models/RESULTS_MODERN.md).

**Retraction.** An earlier draft reported that plain voting *fails* here (62.92%, below best single)
while CES is decisive (+6.19pp over voting). That was an **artifact**: the open-QA voting score is
set by an order-dependent tie-break (fallback to the first-listed candidate when <2 models agree —
near-always, for free text), and the run listed the *weakest* model (Qwen) first. With a fair
fallback the effect vanishes:

| open-QA tie-break | voting | vs CES 69.11% |
|---|---:|---:|
| Qwen first (weakest) — *retracted* | 62.92% | +6.19 |
| **Llama-3.1 first (fair)** | **69.25%** | **−0.14 (p=0.38, n.s.)** |

So **voting-done-right (69.25%) ties/beats CES (69.11%)**; CES is essentially a well-constructed
majority vote here (it is even worse than voting on the discrete branch, −87 items).

**The honest, order-independent measure** — endorsement vs the best single model, on open-QA —
gives **+0.16pp (modern)** and **+0.49pp (2023-era)**: small, same-sign across two model
generations, length-robust, but marginal. There is no regime-dependent inversion.

## 6. Limitations / open work

1. **Weighted majority voting — highest priority.** The textbook fix for a dominant model being
   outvoted (the exact §5 phenomenon). If weighted voting also handles the modern trio, the
   endorsement component has no room left as a distinct contribution. Not yet run.
2. **A principled open-QA voting baseline.** Plain voting is ill-defined on free text; the current
   order-dependent tie-break is not a fair baseline (§5). The honest comparison used here is
   endorsement vs the best single model on open-QA (+0.16 to +0.49pp).
3. **PairRanker / LLM-Blender and MBR** as selection baselines — cited (§4) but not run.
4. **DeePEn at scale** — done only scoped (GSM8K, 50 items, 2 models, 13B baseline incomplete);
   scale to full test sets + all six benchmarks + the missing baseline.
5. **Statistics / data** — McNemar reported; add seeds/CIs. Ship the raw 40k generation JSONs (or a
   hash-verified subset) — currently `eval/out/` is gitignored, so results aren't independently
   checkable from the repo alone.
6. **Naming / scope** — renamed off "EWC" (Elastic Weight Consolidation collision); the
   `ces_batched == ces` parity holds only for odd N. Models are 2023/2024-era; no scaling-law claim.
