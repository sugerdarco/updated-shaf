"""Unified scoring for the mixed DeePEn 100-prompt eval.

Three answer types, one entry point `score_item(item, gen)` where `gen` is the
model's continuation (prompt already stripped):

  mc      choices=[str,...], gold=int index  -> letter/choice-text match
  number  gold=float                          -> last number in gen, exact match
  openqa  gold=[alias,...]                     -> any normalized alias substring
"""
import re

_ARTICLES = re.compile(r"\b(?:a|an|the)\b")
_PUNCT = re.compile(r"[^a-z0-9 ]")
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


def score_mc(item, gen: str) -> bool:
    n = len(item["choices"])
    gold = chr(65 + int(item["gold"]))
    L = f"A-{chr(64 + n)}"  # valid letter class, e.g. 'A-D' for 4 choices
    # 1. after an "answer" cue, first letter through any punctuation:
    #    matches "Answer: B", "Answer:: B", "Answer:* D", "the answer is C".
    m = re.findall(rf"answer\W*(?:is\W*)?([{L}])(?![A-Za-z])", gen, re.I)
    if m:
        return m[-1].upper() == gold
    # 2. "option/choice X"
    m = re.findall(rf"(?:option|choice)\W*([{L}])(?![A-Za-z])", gen, re.I)
    if m:
        return m[0].upper() == gold
    # 3. a letter tagged as a choice: "(B)", "B)", "C.", "D:"
    m = re.findall(rf"(?<![A-Za-z])\(?([{L}])[)\].:]", gen)
    if m:
        return m[0].upper() == gold
    # 4. a lone letter (start of answer or isolated by non-alphanumerics)
    m = re.findall(rf"(?<![A-Za-z0-9])([{L}])(?![A-Za-z0-9])", gen)
    if m:
        return m[0].upper() == gold
    # 5. fall back to the longest matching choice text
    ng = _norm(gen)
    best, blen = -1, 0
    for i, c in enumerate(item["choices"]):
        nc = _norm(c)
        if nc and nc in ng and len(nc) > blen:
            best, blen = i, len(nc)
    if best >= 0:
        return best == int(item["gold"])
    return False


def score_number(item, gen: str) -> bool:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", gen)
    if not nums:
        return False
    try:
        val = float(nums[-1].rstrip(".").replace(",", ""))
    except ValueError:
        return False
    return abs(val - float(item["gold"])) < 1e-4


def score_openqa(item, gen: str) -> bool:
    ng = _norm(gen)
    for a in item["gold"]:
        na = _norm(a)
        if na and na in ng:
            return True
    return False


def score_item(item, gen: str) -> bool:
    t = item["type"]
    if t == "mc":
        return score_mc(item, gen)
    if t == "number":
        return score_number(item, gen)
    return score_openqa(item, gen)
