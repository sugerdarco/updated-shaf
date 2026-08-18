# SAHF Evaluation & Ensemble Reconciliation

This directory adds an **accuracy** evaluation to the SAHF project and the ensemble
method that beats the single-model baseline. The original pipeline measured only
throughput/escalation; nothing scored whether the ensemble was *correct*. This
directory closes that gap and answers the project's actual question:

> Can a multi-agent ensemble of heterogeneous, mismatched-tokenizer LLMs beat the
> best single model's accuracy?

**Answer: yes — but by moving reconciliation from *byte space* to *answer space*, and
mostly via plain voting.** Byte-level fusion collapses to ~36% ("byte-bloat"); a trivial
answer-space selector beats it and the best single model.

## Result at a glance (full 40,604-prompt eval)

| method | acc | vs best single |
|---|---:|---:|
| best single (Mistral-7B) | 61.32% | — |
| byte-space fusion (SAHF) | ~36% | −25 |
| **majority voting** (`vote.py`) | **62.78%** | +1.46 |
| **CES** (`ces.py --hybrid`) | **63.17%** | +1.85 |
| — CES over voting | | +0.39 |

**~79% of the gain over best-single is plain voting.** Read the honest decomposition,
significance tests, and length-confound ablation in
[`../docs/ANALYSIS.md`](../docs/ANALYSIS.md); the method (renamed off "EWC") is in
[`../docs/CES_METHOD.md`](../docs/CES_METHOD.md). Dated write-ups (which use the earlier
over-claimed framing) are in [`../experiment_analysis`](../experiment_analysis).

## The tasks & scoring

Six DeePEn tasks, evaluation splits only, one unified scorer (`scoring.py`):

| task | type | scoring |
|---|---|---|
| ARC-Challenge, MMLU, PIQA | multiple-choice | predicted letter == gold |
| GSM8K | numeric | final number == gold (exact) |
| TriviaQA, NQ | open-QA | any gold alias appears in the answer (normalized EM) |

`predict_item()` extracts a per-model answer; `score_item()` scores a generation;
`score_prediction()` scores an already-extracted answer (used by the ensemble).

## File map

Sampling & scoring (CPU only):
- `build_sample.py` — proportional sampler across all six tasks. Modes: default
  100-prompt plan, `--per-task N`, `--strat --cap C` (small tasks full / big tasks
  capped), `--full` (every eval-split prompt). Extracts only `{question, gold}`.
- `prompts.py` — shared prompt formatting + per-type generation budgets.
- `scoring.py` — unified predict/score functions.
- `rescore.py` — re-score saved result files offline (runners persist gold + full
  generation), so scorer changes never require re-running models.

Single-model baselines (the number to beat):
- `run_single.py` — chat-templated greedy, one prompt at a time.
- `run_single_batched.py` — same, in length-sorted left-padded batches (`--batch`,
  default 16). ~5× faster; batch=16 reproduces the unbatched greedy result exactly.

Byte-space ensemble (the original SAHF pipeline + decode-time variants — *do not*
beat the baseline, kept for the record):
- `run_ensemble.py` — drives `runners/sheaf_orchestrator`. Flags: `--chat`
  (chat-templated agents), `--confidence-weight/--conf-power/--sharpen` (byte-fusion
  tweaks), `--mode fuse|delegate|agree_delegate`, `--sticky`.
- `chat_agent.py` — `ChatTemplateAgent`: wraps each agent's shared context in its
  own chat template inside the byte loop.
- `deleg_orchestrator.py` — **AGED** (Agreement-Gated Expert Delegation): use the
  sheaf byte tree only as an agreement sensor; when agents diverge, delegate the
  step to the most-confident agent instead of averaging (kills byte-bloat but
  plateaus ~48% because per-step routing breaks multi-step reasoning).

Answer-space reconciliation (**the winner**):
- `ces.py` — Cross-Endorsement Selection; `--hybrid` = **EWC**.
- `ces_batched.py` — batched EWC (right-pad teacher forcing + vote-majority skip),
  reproduces `ces.py --hybrid` exactly; makes the 40k run tractable (~22 min).
- `vote.py` — plain answer-level majority voting. Reference baseline only; a
  textbook method, not the contribution.

## Reproduce

```bash
export HF_HOME=/path/to/hf-cache HF_TOKEN=...      # models: Mistral-7B-Instruct-v0.2,
export PYTHONPATH=$(pwd)                            # Llama-2-13b-chat-hf, Yi-6B-Chat

# 1. build a sample (or --full for the whole eval set)
python eval/build_sample.py --strat --cap 2000 --out eval/sample.jsonl

# 2. single-model baselines = candidates (one GPU each, in parallel)
python eval/run_single_batched.py --model mistralai/Mistral-7B-Instruct-v0.2 --device cuda:0 --sample eval/sample.jsonl --out eval/out/mistral.json
python eval/run_single_batched.py --model meta-llama/Llama-2-13b-chat-hf     --device cuda:1 --sample eval/sample.jsonl --out eval/out/llama.json
python eval/run_single_batched.py --model 01-ai/Yi-6B-Chat                   --device cuda:2 --sample eval/sample.jsonl --out eval/out/yi.json

# 3. EWC over those candidates (beats the best single above)
python eval/ces_batched.py --sample eval/sample.jsonl \
  --cands eval/out/mistral.json eval/out/llama.json eval/out/yi.json \
  --devices cuda:0,cuda:1,cuda:2 --out eval/out/ewc.json
```

`--full` builds the whole 40,604-prompt eval set; batched generation runs it in
~1.6 h across 3 GPUs, EWC scoring in ~22 min.
