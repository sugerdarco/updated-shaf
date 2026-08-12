# utils/output_cleaner.py
#
# Post-generation text cleanup for the SAHF byte-fusion pipeline.
#
# Because 4 heterogeneous tokenizers are fused byte-by-byte, the output can
# contain several categories of artifact:
#
#   1. Repeated punctuation  - Yi tokenizer produces ''' or !!! runs.
#   2. Doubled letters       - Weiszfeld median creates "secretedd", "thee" etc.
#   3. Infinite repetition   - Self-reinforcing loops waste the byte budget.
#
# This module strips all three so evaluation scripts see clean matches.

import re

# ---------------------------------------------------------------------------
# 1. Repeated-punctuation collapse: ''' -> '   !!! -> !   ... -> .
# ---------------------------------------------------------------------------
_PUNCT_REPEAT_RE = re.compile(r"([^\w\s])\1+")


def collapse_punct_runs(text):
    return _PUNCT_REPEAT_RE.sub(r"\1", text)


# ---------------------------------------------------------------------------
# 2. Doubled/tripled letter collapse: "secretedd" -> "secreted"  "thee" -> "the"
#    Collapses 2+ identical consecutive letters at word-end positions, and
#    3+ identical consecutive letters anywhere.
# ---------------------------------------------------------------------------
_LETTER_TRIPLE_RE = re.compile(r"([a-zA-Z])\1{2,}")
_LETTER_DOUBLE_END_RE = re.compile(r"([a-zA-Z])\1(?=\s|[.,!?;:\'\"]|$)")  # doubled at word boundary


def collapse_triple_letters(text):
    # First collapse triples (secreteddd -> secretedd -> secreted)
    text = _LETTER_TRIPLE_RE.sub(r"\1\1", text)
    # Then collapse word-boundary doubles that are almost certainly byte artifacts
    # e.g. "secretedd " -> "secreted ", "thee " -> "the "
    text = _LETTER_DOUBLE_END_RE.sub(r"\1", text)
    return text


# ---------------------------------------------------------------------------
# 3. Repetition-loop truncation
# ---------------------------------------------------------------------------
def truncate_at_repetition(text, min_phrase_len=8, max_repeats=3):
    max_phrase = min(120, len(text) // max_repeats + 1)
    for phrase_len in range(min_phrase_len, max_phrase):
        for i in range(len(text) - phrase_len * max_repeats + 1):
            phrase = text[i:i + phrase_len]
            if text[i:i + phrase_len * max_repeats] == phrase * max_repeats:
                return text[:i + phrase_len].rstrip()
    return text


# ---------------------------------------------------------------------------
# 4. First-answer extraction for long outputs
# ---------------------------------------------------------------------------
def extract_first_answer(text, max_len=300):
    if len(text) <= max_len:
        return text.strip()
    for sep in ["\n\n", ". ", ".\n"]:
        idx = text.find(sep)
        if 0 < idx < max_len:
            return text[:idx + 1].strip()
    return text[:max_len].strip()


# ---------------------------------------------------------------------------
# Master cleaner - apply all steps in order
# ---------------------------------------------------------------------------
def clean_output(raw_text, prompt=""):
    """
    Apply all cleanup steps to a raw SAHF output string.

    Parameters
    ----------
    raw_text : full text returned by orchestrator.generate() (including prompt).
    prompt   : the original question, used to strip the re-echoed prefix.

    Returns
    -------
    Cleaned output text suitable for exact-match evaluation.
    """
    generated = raw_text[len(prompt):].lstrip() if raw_text.startswith(prompt) else raw_text

    generated = collapse_punct_runs(generated)
    generated = collapse_triple_letters(generated)
    generated = truncate_at_repetition(generated)
    generated = extract_first_answer(generated, max_len=300)

    return generated
