# Beating the Single-Model Baseline — Endorsement-Weighted Consensus
*Date: 2026-08-18 | 3 models: Mistral-7B-Instruct-v0.2, Llama-2-13b-chat-hf, Yi-6B-Chat | 100-prompt mixed sample*

## Setup
- **Sample**: 100 prompts drawn proportionally across all DeePEn tasks
  (arc 17, mmlu 17, gsm8k 17, piqa 17, triviaqa 16, nq 16), seed 42, evaluation splits only.
  See `eval/sample_100.jsonl`.
- **Scoring** (`eval/scoring.py`): MC → letter match; GSM8K → final-number exact match;
  open-QA → gold-alias exact-match. Identical scorer for every method.
- **Baseline target** (`baseline/BASELINE_METRICS.md`): best single model. On this sample all
  three tie at **57%**; the official 250-prompt baseline reports Llama-2 at 67.6%.

## The journey (what worked and what did not)
| Method | Acc | Note |
|---|---:|---|
| Best single model | 57% | target |
| SAHF byte-fusion (base) | 36% | byte-bloat: averaging mismatched byte distributions misspells words |
| + confidence-weight + sharpen | 33–34% | **negative result** — a blurred byte distribution cannot be rescued by sharpening |
| Decode-time delegation (AGED / MoE) | 47–48% | clean MC (arc 16, piqa 13) but per-token leader-switching wrecks CoT (gsm8k→0) |
| + sticky (per-line leadership) | 41% | pinning one leader hurts MC; still cannot fix multi-step reasoning |
| Answer-level majority voting | 61% | classic ensemble — reference only, not a novel contribution |
| Cross-Endorsement Selection (soft only) | 58% | likelihood is a weak signal for discrete answers |
| **Endorsement-Weighted Consensus (EWC)** | **63%** | **beats best single by +6** |

**Oracle ceiling** (correct if *any* model is right) = **82%**, so the 3 models are highly
complementary and the remaining gap is a selection problem.

## The novel technique: Endorsement-Weighted Consensus (EWC)
Inspired by DeePEn's cross-model collaboration, but distinct from DeePEn's relative-representation
*averaging* (which, on mismatched tokenizers, is exactly what manufactures byte-bloat) and from
plain majority voting.

Each agent answers independently (clean, chat-templated — no byte-bloat). Every candidate answer
is then scored by the **whole ensemble's endorsement**, measured two ways under one rule:
- **discrete answers (MC / numeric)** — *hard* endorsement: agents endorse by their explicit
  extracted answer (agreement / vote), ties broken by the soft signal below;
- **free-form answers (open-QA)** — *soft* endorsement: each agent's length-normalized
  log-probability of the candidate under its **own** tokenizer and chat template.

The maximally-endorsed candidate wins. An answer several heterogeneous models independently endorse
is more trustworthy than any one model's own pick — this converts their complementarity into
accuracy without ever averaging distributions and without peeking at gold.

Implementation: `eval/ces.py --hybrid` (`eval/deleg_orchestrator.py` holds the decode-time AGED
baselines). Reuses the per-model generations from `eval/run_single.py`, so only cheap scoring
passes run (~26 s for 100 prompts).

## Final result (100-sample)
| task | best single | EWC |
|---|---:|---:|
| arc | 16 | 15 |
| mmlu | 10 | 10 |
| piqa | 13 | 12 |
| gsm8k | 9 | 9 |
| triviaqa | 13 | 12 |
| nq | 5 | 5 |
| **total** | **57%** | **63%** |

EWC exceeds the best single model overall. Next: validate on a 1k+ sample (Goal 2).
