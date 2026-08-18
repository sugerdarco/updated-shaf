# Cross-Endorsement Selection (CES)

Specification of the answer-space selection method used in this repo. **Renamed from
"EWC"** (which collides with Elastic Weight Consolidation). For the honest evaluation,
decomposition, significance, and prior-art positioning, see [`ANALYSIS.md`](ANALYSIS.md);
this file defines the mechanism only.

> CES is **not** claimed as a large novel contribution. Its discrete branch is plain majority
> voting; its open-QA branch is likelihood-utility Minimum Bayes Risk / n-best rescoring
> applied across heterogeneous tokenizers. It is documented as the strong answer-space
> **baseline** in the analysis, and as the contrast to byte-space distribution fusion.

## Setup

`N` heterogeneous, mismatched-tokenizer models each answer a prompt independently and cleanly
(their own chat template, greedy decode — no cross-model interference, no byte-bloat),
producing candidate answers `c_1 … c_N`. CES selects one candidate.

## Selection rule

**Discrete answers (multiple-choice, numeric) — majority vote.**
Each model's extracted answer is a vote; the plurality wins. Endorsement (below) only breaks
the 3-way tie, which for N=3 is the sole case with no majority. (Precondition: the majority
rule and the tie-break agree only for odd N; even N can split.)

**Free-form answers (open-QA) — likelihood endorsement (MBR).**
Pick the candidate maximizing summed length-normalized log-probability under every model, each
scoring under its own tokenizer and chat template:
```
select  argmax_j  Σ_a  (1/|c_j|) Σ_t  log P_a( c_j[t] | prompt, c_j[<t] )
```
This is MBR decoding with a likelihood utility (Kumar & Byrne 2004; cf. Model-Based MBR,
Jinnai et al. 2023), across a heterogeneous ensemble.

## Where the contribution actually is

On the full 40,604-prompt evaluation, over plain voting CES adds **+0.39pp** overall
(McNemar p = 3.9e-4). The gain is concentrated entirely in open-QA, where voting gains nothing
(+1 item) but endorsement gains +108 items, and it is **not a length artifact** — endorsement
(65.77%) beats selecting the longest (62.96%) or shortest (62.75%) candidate. Full numbers and
caveats in [`ANALYSIS.md`](ANALYSIS.md).

## Implementation

`eval/ces.py --hybrid` (reference), `eval/ces_batched.py` (batched: right-pad teacher forcing +
vote-majority skip; reproduces the reference exactly for odd N). Baselines: `eval/vote.py`
(voting), `eval/run_single[_batched].py` (single models). Analysis: `eval/paper_analysis.py`.
