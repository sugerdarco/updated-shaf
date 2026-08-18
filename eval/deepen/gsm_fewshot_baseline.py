#!/usr/bin/env python3
"""Single-model few-shot GSM8K baseline matching DeePEn's exact prompt/demos,
so it is directly comparable to the DeePEn fusion number on the same 50 examples."""
import json, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, device = sys.argv[1], sys.argv[2]
D = "/storage/riya/riya_experiment/DeePEn/datasets/GSM/data"
demon = [json.loads(l) for l in open(f"{D}/demon_4.jsonl")]
demon_instr = "".join(
    f"Question: {d['question']}\nLet's think step by step\nAnswer:{d['answer']}\n" for d in demon)
tests = [json.loads(l) for l in open(f"{D}/test.cleand.head50.jsonl")]

tok = AutoTokenizer.from_pretrained(model_id, truncation_side="left")
m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()
device = m.device


def num(s):
    x = re.findall(r"-?\d[\d,]*\.?\d*", str(s).replace(",", ""))
    return float(x[-1]) if x else None


c = 0
for t in tests:
    prompt = demon_instr + f"Question: {t['question']}\nLet's think step by step\nAnswer:"
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=3500).input_ids.to(device)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("\n\n")[0]
    mm = re.search(r"answer is\s*\$?(-?\d[\d,]*\.?\d*)", gen, re.I)
    pred = num(mm.group(1)) if mm else num(gen)
    gold = num(t["answer"])
    c += (pred is not None and gold is not None and abs(pred - gold) < 1e-4)
print(f"BASELINE {model_id} GSM8K few-shot = {c}/{len(tests)} = {c/len(tests)*100:.1f}%")
