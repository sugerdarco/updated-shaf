# Full-Dataset Evaluation — Endorsement-Weighted Consensus
*Date: 2026-08-18 | 3 models: Mistral-7B-Instruct-v0.2, Llama-2-13b-chat-hf, Yi-6B-Chat | 40,604 prompts*

## Setup
- **Full DeePEn evaluation set** from VM2 `~/deepEn_dataset` (163 GB): every eval-split prompt across
  all six tasks — **40,604** total (ARC-C 1,172; MMLU 14,042; GSM8K 1,319; PIQA 1,838;
  TriviaQA 17,944; NQ 4,289 short-answer). Only {question, gold} was extracted (no bulk context copied).
- **Batched inference** (`run_single_batched.py`, B=16, length-sorted left-pad) — validated to
  reproduce unbatched greedy exactly; generation took ~68–72 min per model (3 GPUs in parallel).
- **Batched EWC scoring** (`ces_batched.py`) — validated to reproduce `ces.py --hybrid` exactly;
  15,411 discrete prompts settled by vote majority (no GPU), 25,193 scored by endorsement (~22 min).

## Results
| Model / method | Accuracy | ARC | MMLU | GSM8K | PIQA | TriviaQA | NQ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mistral-7B (best single) | **61.31%** | 70.0 | 54.4 | 40.5 | **76.1** | **72.8** | **33.7** |
| Yi-6B | 58.97% | **78.7** | **59.2** | 37.3 | 69.1 | 65.3 | 28.5 |
| Llama-2-13B | 56.87% | 62.4 | 46.4 | 35.9 | 59.0 | 72.2 | 31.2 |
| Oracle (any-correct ceiling) | 76.63% | | | | | | |
| Best-model-per-task (oracle router) | 63.26% | | | | | | |
| **EWC (novel)** | **63.17%** | 76.8 | 58.3 | **47.5** | 71.1 | 73.3 | 34.5 |

(cells are per-task accuracy %)

## Verdict
- **EWC beats the best single model at full scale: 63.17% vs 61.31% → +1.85 points.** The result the
  100- and 1,200-prompt experiments predicted holds on all 40,604 prompts.
- **EWC ≈ the oracle per-task model router (63.17% vs 63.26%)** — it recovers almost all of the gain
  a perfect "pick the right model for this task" selector would get, **without any gold labels**.
- Per task, EWC beats *every* single model on **GSM8K (+7.0)**, **TriviaQA**, and **NQ**; it trails the
  best single on ARC/MMLU/PIQA, where one model dominates and hard-agreement voting dilutes it.
- The margin (+1.85) is below the +2.05% DeePEn "SOTA" threshold at this scale (it cleared it on the
  1,200-sample at +2.7). The single clear lever to close that gap: **reliability-weighted endorsement
  for discrete answers** — PIQA alone (71.1 vs 76.1, one strongly-dominant model) accounts for most of
  the shortfall.

## Reproduce
```
python eval/run_single_batched.py --model <m> --device cuda:X --sample eval/full_eval.jsonl --out out/full_<m>.json --batch 16
python eval/ces_batched.py --sample eval/full_eval.jsonl --cands out/full_mistral.json out/full_llama.json out/full_yi.json --out out/ewc_full.json
```
