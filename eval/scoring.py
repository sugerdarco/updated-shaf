"""Unified prediction + scoring for the mixed DeePEn 100-prompt eval.

Three answer types. `predict_item(item, gen)` returns the model's extracted answer
(letter / float / short-string), `score_item(item, gen)` returns correctness. The
prediction functions are reused by the answer-level voting ensemble (vote.py).
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


# ---- MC -----------------------------------------------------------------

def predict_mc(item, gen: str):
    """Return the predicted choice letter (e.g. 'B') or None."""
    n = len(item["choices"])
    L = f"A-{chr(64 + n)}"
    m = re.findall(rf"answer\W*(?:is\W*)?([{L}])(?![A-Za-z])", gen, re.I)
    if m:
        return m[-1].upper()
    m = re.findall(rf"(?:option|choice)\W*([{L}])(?![A-Za-z])", gen, re.I)
    if m:
        return m[0].upper()
    m = re.findall(rf"(?<![A-Za-z])\(?([{L}])[)\].:]", gen)
    if m:
        return m[0].upper()
    m = re.findall(rf"(?<![A-Za-z0-9])([{L}])(?![A-Za-z0-9])", gen)
    if m:
        return m[0].upper()
    ng = _norm(gen)
    best, blen = -1, 0
    for i, c in enumerate(item["choices"]):
        nc = _norm(c)
        if nc and nc in ng and len(nc) > blen:
            best, blen = i, len(nc)
    if best >= 0:
        return chr(65 + best)
    return None


def score_mc(item, gen: str) -> bool:
    return predict_mc(item, gen) == chr(65 + int(item["gold"]))


# ---- number -------------------------------------------------------------

def predict_number(item, gen: str):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", gen)
    if not nums:
        return None
    try:
        return float(nums[-1].rstrip(".").replace(",", ""))
    except ValueError:
        return None


def score_number(item, gen: str) -> bool:
    p = predict_number(item, gen)
    return p is not None and abs(p - float(item["gold"])) < 1e-4


# ---- open QA ------------------------------------------------------------

def predict_openqa(item, gen: str) -> str:
    """A short answer string: text after an 'Answer:' cue, first line, clipped."""
    g = gen.strip()
    m = re.search(r"answer\s*:?\s*(.+)", g, re.I | re.S)
    if m:
        g = m.group(1)
    g = g.strip().splitlines()[0] if g.strip() else ""
    return g[:100].strip()


def score_openqa(item, gen: str) -> bool:
    ng = _norm(gen)
    for a in item["gold"]:
        na = _norm(a)
        if na and na in ng:
            return True
    return False


# ---- dispatch -----------------------------------------------------------

def predict_item(item, gen: str):
    t = item["type"]
    if t == "mc":
        return predict_mc(item, gen)
    if t == "number":
        return predict_number(item, gen)
    return predict_openqa(item, gen)


def score_item(item, gen: str) -> bool:
    t = item["type"]
    if t == "mc":
        return score_mc(item, gen)
    if t == "number":
        return score_number(item, gen)
    return score_openqa(item, gen)


def score_prediction(item, pred) -> bool:
    """Score an already-extracted prediction (used by the voting ensemble)."""
    if pred is None:
        return False
    t = item["type"]
    if t == "mc":
        return pred == chr(65 + int(item["gold"]))
    if t == "number":
        return abs(float(pred) - float(item["gold"])) < 1e-4
    ns = _norm(str(pred))
    return any(_norm(a) and _norm(a) in ns for a in item["gold"])
