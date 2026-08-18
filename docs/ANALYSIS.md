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
- The genuinely-novel component (open-QA likelihood/MBR endorsement) contributes a **small**
  improvement over the best single model on open-QA — but only one of two runs is distinguishable
  from zero (paired bootstrap, 4,000 resamples): **+0.49pp [+0.13, +0.84] (2023-era)** and
  **+0.16pp [−0.22, +0.51] (modern — CI crosses zero)**. Consistent in sign, length-robust, but
  marginal — which *supports* the thesis that endorsement adds little. (Headline CES-vs-best-single
  gains are solid: **+1.85 [+1.52, +2.18]** and **+0.98 [+0.62, +1.35]**.) *(An earlier draft
  claimed a +6.19pp modern "inversion" — an artifact of an order-dependent open-QA voting tie-break,
  retracted, see §5.)*
- **Sharper still: CES has no distinct advantage over trivial baselines.** Two dev-split (20-seed)
  baselines each match or beat it: an **answer-type router** ("best model for MC/numeric, best model
  for open-QA") — modern 69.19 ≥ CES 69.11 (n.s.), 2023-era 62.86 vs 63.17 (+0.31, p=0.045); and
  **reliability-weighted majority voting** — 2023-era 63.11 vs CES 63.17, modern 69.16 vs CES 69.11
  (both within ±0.2 seed noise). No single model is best at both answer types, so nearly all the
  ensemble gain over the best single model is answer-**type routing**. See §2b and
  [`../experiment_analysis/2026-08-18-type-router/RESULTS_ROUTER.md`](../experiment_analysis/2026-08-18-type-router/RESULTS_ROUTER.md).

**Implication for framing:** this is a **negative-result / analysis** paper. Its contribution is
(1) the mechanistically-diagnosed collapse of byte-level distribution fusion (byte-bloat) while a
trivial answer-space method beats it and the best single model; and (2) the honest decomposition
showing that gain is **mostly answer-type routing** (a two-number, label-free heuristic — §2b), with
the remainder split between within-type majority voting and a small open-QA endorsement effect
(+0.16 to +0.49pp, only one of two runs distinguishable from zero). It is **not** a method paper
about a new decisive ensemble mechanism. And (3) a **reusable methodological point**: *any* ensemble
evaluated on a mixed-task suite collects a free answer-**type routing** bonus that most ensembling
papers do not control for — the answer-type router (§2b) is a cheap, label-free control others
should adopt before claiming a fusion/selection gain. The DeePEn reproduction (below) narrows the
negative result to our byte-tree instantiation, not the paradigm.

## 1. Why byte-space fusion fails (the negative result)

Averaging next-*byte* distributions from models with different tokenizers produces a fused
distribution whose argmax is a byte **no constituent model intended**, yielding misspellings
("secretedd", "Juiced Box Band") that destroy exact-match scoring. Rescues tried and failed:
confidence-weighting/sharpening the fusion (33–34%, *worse*), and **AGED** — use the byte tree
only as an agreement sensor and delegate divergent steps to the most-confident agent
(47–48%). A specific finding: **per-step leader switching zeroes out GSM8K** because it breaks
multi-step reasoning mid-chain. (`eval/deleg_orchestrator.py`.)

**Narrowed by reproduction (§6):** DeePEn does distribution averaging in a *relative-representation*
space. We ran DeePEn's released code on our own models. On **full ARC-Challenge (1,172 items)**,
DeePEn's faithful method (learned weights, lr=0.5) scores **73.9% — tying the best single model
(73.8%)** — with every one of the 1,172 outputs a clean, valid letter (no byte-bloat). (Uniform-weight
fusion, lr=0.0, gets 64.2%, dragged toward the weaker model — a config caveat, not the method.) So
relative-space fusion is coherent **and** competitive at full-task scale; the byte-bloat collapse is
specific to *our byte-tree instantiation* (averaging next-*byte* distributions yields an argmax no
model intended). Notably it only *ties*, not beats, the best single model — consistent with §2b
(distribution fusion is not the win; answer-type routing is). See
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

## 2b. Most of the gain is answer-type routing, not selection

The shipped per-prompt data (`eval/results_export/`) shows **no single model is best at both answer
types**: on the modern trio Qwen wins discrete (73.9 vs Mistral 62.1) but loses open-QA (50.3 vs
Mistral 65.4). CES's discrete branch tracks the best discrete model and its open-QA branch tracks
the best open-QA model — it is an *implicit answer-type router*. An explicit router (one model for
mc/number, one for open-QA), chosen on a random dev half and applied to the other (20 seeds, no test
peeking):

| | best single | CES | type-router (dev-split) |
|---|---:|---:|---:|
| 2023-era | 61.32 | 63.17 | 62.86 ±0.19 |
| modern | 68.13 | 69.11 | **69.19 ±0.18** |

Decomposition of CES's gain over best single: **routing accounts for +1.54 of +1.85 (2023-era) and
+1.06 of +0.98 (modern — the router *is* CES)**. So the ensemble's advantage on a mixed suite is
primarily answer-type routing; fusion/voting/endorsement add little beyond it. Full write-up +
`eval/type_router.py` in
[`../experiment_analysis/2026-08-18-type-router/RESULTS_ROUTER.md`](../experiment_analysis/2026-08-18-type-router/RESULTS_ROUTER.md).

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

1. **Weighted majority voting — done.** Reliability-weighted vote (dev-split, 20 seeds) ties CES on
   both trios (2023-era 63.11 vs 63.17; modern 69.16 vs 69.11). Together with the type-router (§2b),
   this confirms CES has **no distinct room** over standard baselines. `eval/weighted_vote.py`.
2. **Answer-type router baseline — done (§2b).** Dev-split, matches/beats CES; folded into the
   thesis. A principled open-QA voting baseline is still ill-defined (the order-dependent tie-break
   is not fair, §5); endorsement vs best-single-on-open-QA is the honest comparison (+0.16–0.49pp).
3. **Trained rerankers (LLM-Blender / PairRanker) — out of scope, by declaration.** We restrict to
   **training-free** ensembling (as DeePEn does). PairRanker is trained on MixInstruct for
   instruction-following quality and is out of domain for MC-letter / integer answers (GAC report
   its fuser refusing a large share of QA questions), so running it off-the-shelf would buy a
   domain-mismatch argument rather than settle one. Cited as complementary (§4). MBR is the right
   framing for our open-QA endorsement branch (§2), not a separate baseline. With this boundary the
   baseline set is complete: plain / weighted / type-router / oracle-router / oracle-ceiling + length
   ablation.
4. **DeePEn at scale — the gating open item.** The §1 claim (our byte-tree instantiation fails, not
   the paradigm) is our *primary* contribution and currently rests on a scoped run (GSM8K, 50 items,
   2 models, 13B baseline incomplete). In progress: full GSM8K + a second task + the missing baseline.
5. **Statistics / data** — McNemar reported; add seeds/CIs. Compact per-prompt correctness for both
   40k runs is now shipped in `eval/results_export/` (per model + CES; recompute any headline number
   with `eval/export_results.py`'s schema). Full generations remain local (gitignored).
6. **Naming / scope** — renamed off "EWC" (Elastic Weight Consolidation collision); the
   `ces_batched == ces` parity holds only for odd N. Models are 2023/2024-era; no scaling-law claim.
