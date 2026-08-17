#!/usr/bin/env python3
"""Build a 100-prompt evaluation sample drawn proportionally from every DeePEn task.

Six tasks, evaluation splits only (test/validation/dev), fixed seed, uniform-random
within each task. Counts sum to 100:

    arc 17 | mmlu 17 | gsm8k 17 | piqa 17 | triviaqa 16 | nq 16

Unified per-line schema:
    {"task","type","question","choices"?,"gold"}
      type "mc":     choices=[str,...], gold=int index
      type "number": gold=float
      type "openqa": gold=[alias,...]
"""
import argparse
import json
import os
import random
import re

SEED = 42
DATA = os.environ.get("DEEPEN_DIR", "/storage/riya/bhuvi-sahf-test/dataset/deepEn_dataset_1k")


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


def load_nq(max_scan=1500, cap=400):
    """NQ validation is ~470 MB with ~1 MB HTML per line; scan a bounded prefix and
    keep only questions that carry a short answer."""
    out = []
    path = f"{DATA}/google-research-datasets_natural_questions_default_validation.jsonl"
    for d in _iter_jsonl(path, max_scan=max_scan):
        q = (d.get("question") or {}).get("text")
        texts = []
        for a in (d.get("annotations") or {}).get("short_answers") or []:
            for t in (a.get("text") or []):
                if t:
                    texts.append(t)
        if q and texts:
            out.append({"task": "nq", "type": "openqa", "question": q,
                        "gold": list(dict.fromkeys(texts))})
        if len(out) >= cap:
            break
    return out


PLAN = [
    ("arc", load_arc, 17), ("mmlu", load_mmlu, 17), ("gsm8k", load_gsm8k, 17),
    ("piqa", load_piqa, 17), ("triviaqa", load_triviaqa, 16), ("nq", load_nq, 16),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/sample_100.jsonl")
    args = ap.parse_args()
    rng = random.Random(SEED)
    sample = []
    for name, loader, n in PLAN:
        pool = loader()
        pick = rng.sample(pool, min(n, len(pool)))
        print(f"{name:9s} pool={len(pool):6d} picked={len(pick)}")
        sample.extend(pick)
    rng.shuffle(sample)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in sample:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"WROTE {len(sample)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
