#!/usr/bin/env python3
"""Build an evaluation prompt file from the DeePEn tasks.

Extracts only {task, type, question, choices?, gold} — never the bulky context in
the raw NQ/TriviaQA files — so a full-dataset prompt file is a few MB, not 163 GB.

Modes:
  (default)        the fixed 100-prompt plan (17/17/17/17/16/16)
  --per-task N     N per task
  --strat --cap C  small tasks in full, large tasks (mmlu/triviaqa/nq) capped at C
  --full           every eval-split prompt (no sampling)
"""
import argparse
import json
import os
import random
import re

SEED = 42
DATA = os.environ.get("DEEPEN_DIR", "/storage/riya/bhuvi-sahf-test/dataset/deepEn_dataset_1k")
NQ_MAX_SCAN = 6000   # overridden for --full/--strat
NQ_CAP = 1000
BIG = {"mmlu", "triviaqa", "nq"}


def _iter_jsonl(path, max_scan=None):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_scan is not None and i >= max_scan:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_arc():
    out = []
    for d in _iter_jsonl(f"{DATA}/allenai_ai2_arc_ARC-Challenge_test.jsonl"):
        labels, texts = d["choices"]["label"], d["choices"]["text"]
        if d["answerKey"] not in labels:
            continue
        out.append({"task": "arc", "type": "mc", "question": d["question"],
                    "choices": texts, "gold": labels.index(d["answerKey"])})
    return out


def load_mmlu():
    out = []
    for d in _iter_jsonl(f"{DATA}/cais_mmlu_all_test.jsonl"):
        out.append({"task": "mmlu", "type": "mc", "question": d["question"],
                    "choices": d["choices"], "gold": int(d["answer"])})
    return out


def load_gsm8k():
    out = []
    for d in _iter_jsonl(f"{DATA}/openai_gsm8k_main_test.jsonl"):
        m = re.search(r"####\s*(-?[\d,]+\.?\d*)", d["answer"])
        if not m:
            continue
        out.append({"task": "gsm8k", "type": "number", "question": d["question"],
                    "gold": float(m.group(1).replace(",", ""))})
    return out


def load_piqa():
    out = []
    for d in _iter_jsonl(f"{DATA}/piqa_dev.jsonl"):
        out.append({"task": "piqa", "type": "mc", "question": d["goal"],
                    "choices": [d["sol1"], d["sol2"]], "gold": int(d["label"])})
    return out


def load_triviaqa():
    out = []
    for d in _iter_jsonl(f"{DATA}/mandarjoshi_trivia_qa_rc_validation.jsonl"):
        ans = d.get("answer") or {}
        golds = [a for a in ([ans.get("value")] + (ans.get("aliases") or [])) if a]
        golds = list(dict.fromkeys(golds))
        if not golds:
            continue
        out.append({"task": "triviaqa", "type": "openqa", "question": d["question"],
                    "gold": golds})
    return out


def load_nq():
    out = []
    path = f"{DATA}/google-research-datasets_natural_questions_default_validation.jsonl"
    for d in _iter_jsonl(path, max_scan=NQ_MAX_SCAN):
        q = (d.get("question") or {}).get("text")
        texts = []
        for a in (d.get("annotations") or {}).get("short_answers") or []:
            for t in (a.get("text") or []):
                if t:
                    texts.append(t)
        if q and texts:
            out.append({"task": "nq", "type": "openqa", "question": q,
                        "gold": list(dict.fromkeys(texts))})
        if len(out) >= NQ_CAP:
            break
    return out


PLAN = [("arc", load_arc, 17), ("mmlu", load_mmlu, 17), ("gsm8k", load_gsm8k, 17),
        ("piqa", load_piqa, 17), ("triviaqa", load_triviaqa, 16), ("nq", load_nq, 16)]


def main():
    global NQ_MAX_SCAN, NQ_CAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/sample_100.jsonl")
    ap.add_argument("--per-task", type=int, default=0)
    ap.add_argument("--strat", action="store_true", help="small tasks full, big tasks capped at --cap")
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--full", action="store_true", help="every eval-split prompt, no sampling")
    args = ap.parse_args()
    if args.full or args.strat:
        NQ_MAX_SCAN, NQ_CAP = 10**9, 10**9  # scan the whole NQ val file

    rng = random.Random(SEED)
    sample = []
    for name, loader, n in PLAN:
        pool = loader()
        if args.full:
            pick = pool
        elif args.strat:
            want = args.cap if name in BIG else len(pool)
            pick = rng.sample(pool, min(want, len(pool)))
        else:
            want = args.per_task if args.per_task else n
            pick = rng.sample(pool, min(want, len(pool)))
        print(f"{name:9s} pool={len(pool):7d} picked={len(pick)}")
        sample.extend(pick)
    if not args.full:
        rng.shuffle(sample)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in sample:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"WROTE {len(sample)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
