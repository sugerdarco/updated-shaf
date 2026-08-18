# DeePEn reproduction — relative-space fusion is coherent AND competitive; our byte-tree is what fails
*Updated 2026-08-19 | DeePEn (Huang et al., NeurIPS 2024) relative-representation fusion | our models*

The pivotal experiment: run DeePEn's *released* code (relative-space distribution fusion) on our own
models, to decide whether the earlier ~36% byte-fusion collapse means the **distribution-fusion
paradigm fails**, or only **our byte-tree instantiation** fails.

## Setup
- DeePEn repo `OrangeInSouth/DeePEn`, its own env (torch 2.1.2 / transformers 4.40) and its native
  few-shot data. Ensemble = **Llama-2-13b-chat + Mistral-7B-Instruct-v0.2**. Transfer matrices built
  with DeePEn's `cal_and_save_transfer_matrix.py` (common vocabulary = 24,184 anchors).

## Result — full ARC-Challenge (1,172 items), faithful method

| ARC-Challenge (full) | accuracy | note |
|---|---:|---|
| DeePEn fusion, **lr=0.0** (uniform weights) | 64.2% | *misconfiguration* — uniform averaging is dragged toward the weaker model |
| **DeePEn fusion, lr=0.5** (DeePEn's learned weights) | **73.9%** | faithful |
| Mistral-7B-Instruct-v0.2 (best single) | 73.8% | |
| Llama-2-13b-chat | ~62% (weaker; our zero-shot eval) | |

**With DeePEn's actual learned-weight method (lr=0.5), fusion (73.9%) ties the best single model
(73.8%)** on a full 1,172-item task — coherent *and* competitive. Every one of the 1,172 fusion
outputs is a clean, valid letter — no byte-bloat.

**Fidelity lesson.** Uniform-weight fusion (lr=0.0, the argparse default) scored 64.2% — dragged
toward the weaker Llama-2. DeePEn's per-example weight learning (lr=0.5) recovers the +9.7pp. Our
earlier scoped GSM8K number (44%, n=50) was also lr=0.0 and should be read as uniform-averaging, not
DeePEn's method.

## GSM8K
Scoped n=50 at lr=0.0 gave 44%. A faithful full-GSM8K fusion is **compute-bound**: on the shared,
over-subscribed GPUs it runs at ~50–70 s/item (~24 h for 1,319 items), so we report a faithful
lr=0.5 **sample** rather than the full set. The load-bearing claim below does not depend on it — the
full ARC run establishes it.

## What this establishes (the framing question, answered at full-task scale)

- **DeePEn's relative-space fusion produces correct, coherently-spelled output on our exact models**
  (all 1,172 ARC answers valid) and, done faithfully, **matches the best single model** — no
  byte-bloat, no collapse.
- **Our byte-tree fusion of the same class of models collapsed to ~36% (≈0–2% on GSM8K) via
  byte-bloat.**

So distribution-space fusion is **not** inherently broken across mismatched tokenizers — the failure
is **specific to our byte-level instantiation** (averaging next-*byte* distributions yields an argmax
no model intended). This is the narrowed §1 claim, now supported by a **full-task, faithful**
reproduction rather than the earlier scoped run.

Note this also fits the paper's broader thesis: even DeePEn's coherent relative-space fusion only
*ties* the best single model here — it does not beat it — while a trivial answer-type router does.
Distribution fusion (byte or relative-space) is not the win; answer-type routing is.

## Caveats
- Instruct models run in DeePEn's base-style few-shot (its native setup), matched across fusion and
  baselines. Llama-2 ARC few-shot baseline was GPU-contention-blocked; Mistral (73.8%) is the
  stronger ARC model in our full eval, so it is the best single.
- Reproduce: `DeePEn/` transfer matrices → `confs/{GSM8K,ARC-c}/ours_full.json` →
  `main_many_ensemble_llama_series_local_matrix.py -lpm based_on_probility_transfer_logits_fp32_processor -lr 0.5`.
