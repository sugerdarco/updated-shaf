# Endorsement-Weighted Consensus (EWC)

Formal specification of the ensemble-reconciliation method that beats the best
single model in the SAHF project. Implemented in `eval/ces.py --hybrid` (batched:
`eval/ces_batched.py`). Evaluated in `experiment_analysis/2026-08-18-*`.

## 1. Motivation — why byte-level fusion fails

The original SAHF pipeline (and DeePEn) reconcile heterogeneous, mismatched-tokenizer
models by **averaging their next-step distributions** in a shared space — for SAHF, a
byte-prefix tree; for DeePEn, a relative-representation space. On real mismatched
tokenizers this averaging manufactures **byte-bloat**: the mean of several models'
next-byte distributions has an argmax that is a byte *none* of them intended, so the
consensus text is misspelled ("secretedd", "Juiced Box Band"). Measured accuracy of
byte-fusion collapses to ~36% versus ~57% for the best single model.

Decode-time fixes were tried and rejected (kept in `eval/deleg_orchestrator.py`):
weighting/sharpening the byte fusion (33–34%, worse), and **AGED** — using the sheaf
tree only as an agreement sensor and delegating divergent steps to the most-confident
agent (47–48%; per-step leader switching breaks multi-step reasoning).

**Conclusion:** reconcile in **answer space**, not byte space.

## 2. Method

Let agents `a = 1..N` each answer a prompt independently and cleanly (greedy, with
their own chat template — no cross-model interference, no byte-bloat), producing
candidate answers `c_1 … c_N`. EWC selects one candidate by **whole-ensemble
endorsement**, measured under a single rule that adapts to the answer type:

**Discrete answers (multiple-choice, numeric) — hard endorsement.**
Each agent endorses the candidate whose *extracted answer* equals its own
(agreement / majority vote). The winner is the answer with the most votes; ties are
broken by the soft signal below. This is where the ensemble corrects an individual
model's mistakes when the other two agree.

**Free-form answers (open-QA) — soft endorsement.**
Voting is hopeless on free text (models phrase answers differently), so each agent
endorses candidate `c_j` by its **length-normalized log-probability of `c_j` under
its own tokenizer and chat template**:
```
endorse(c_j) = Σ_a  (1/|c_j|) Σ_t  log P_a( c_j[t] | prompt, c_j[<t] )
```
The most-endorsed candidate wins. An answer several heterogeneous models
independently find likely is more trustworthy than any one model's own pick.

The unified selection:
```
if type is discrete:  winner = argmax_answer ( votes(answer), max endorse of its candidates )
else:                 winner = argmax_j  endorse(c_j)
```

## 3. Relation to prior work

- **vs DeePEn / byte-fusion:** those *average* distributions in a shared space; EWC
  never averages — it *selects* a whole clean answer. That is what avoids byte-bloat.
- **vs majority voting:** voting ignores how much the other models endorse an answer
  and is useless on free-form text; EWC's soft endorsement handles open-QA and its
  hard endorsement recovers voting as the discrete special case.
- **vs a per-task model router:** EWC needs no gold and no task labels, yet on the
  full dataset it matches the *oracle* per-task router (63.17% vs 63.26%).

## 4. Results

| Scale | Best single | EWC | Δ |
|---|---:|---:|---:|
| 100-prompt sample | 57.0% | 63.0% | +6.0 |
| 1,200-prompt sample | 54.8% | 57.5% | +2.7 |
| Full 40,604 (DeePEn eval) | 61.31% | 63.17% | +1.85 |

Oracle any-correct ceiling on the full set is 76.6%. Per task, EWC beats every single
model on GSM8K, TriviaQA, and NQ.

## 5. Limitations & next step

EWC's hard-agreement voting **dilutes a strongly-dominant model** on discrete tasks:
on PIQA one model scores 76.1% but the 3-way vote drops to 71.1%, which is most of the
gap to DeePEn's +2.05% SOTA threshold at full scale. The clear next lever is
**reliability-weighted endorsement** — weight each agent's discrete vote by a gold-free
reliability estimate (e.g. its own answer confidence) instead of one-agent-one-vote.
